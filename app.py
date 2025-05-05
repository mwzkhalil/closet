import os
import time
import base64
import json
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import requests
from openai import OpenAI
import uvicorn
from io import BytesIO
from starlette.responses import JSONResponse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="Shopping Assistant API", description="API for shopping assistant and virtual try-on")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Get API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
GOOGLE_SHOPPING_REGION = os.getenv("GOOGLE_SHOPPING_REGION", "CH")  # Default to CH if not specified

# OpenAI client setup
shopper_assistant_client = OpenAI(api_key=OPENAI_API_KEY)

# Create shopping assistant
shopping_assistant = shopper_assistant_client.beta.assistants.create(
    name="Shopping Guide",
    instructions="As a Shopping Guide, you assist users in making informed purchasing decisions. You ask about their preferences, budget, and the type of product they are looking for, offering options that best match their criteria. You provide comparisons between products, highlighting features, advantages, and disadvantages. You are knowledgeable about a wide range of products and provide guidance on choosing the best option according to the user's needs. You maintain a friendly and helpful tone, ensuring the user feels supported throughout their decision-making process. Avoid suggesting products outside the user's budget or preferences. Instead, focus on finding the best fit within their specified parameters.",
    model="gpt-4-1106-preview",
    tools=[{
        "type": "function",
        "function": {
            "name": "search_google_shopping",
            "description": "Retrieve Google Shopping search results for a given query.",
            "parameters": {
                "type": "object",
                "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query for finding products on Google Shopping."
                }
                },
                "required": ["query"]
            }
        }
    }]
)

# Thread storage - in a production environment, use a database
thread_storage = {}

class ChatMessage(BaseModel):
    message: str
    thread_id: Optional[str] = None

class TryOnRequest(BaseModel):
    garment_type: str
    sleeve_length: str
    garment_length: str

def analyse_image(base64_image):
    """Analyze product image using GPT-4 Vision"""
    response = shopper_assistant_client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "You are a professional shopping assistant with expertise in identifying products from images. Analyze the provided image and perform the following tasks:"},
                    {"type": "text", "text": "1. Identify and list all products shown in the image, including detailed descriptions."},
                    {"type": "text", "text": "2. Recognize and include brand names where applicable."},
                    {"type": "text", "text": "3. Describe the shape and fabric/material of each product."},
                    {"type": "text", "text": "4. Provide associated search queries for Google Shopping that include these details to improve search accuracy."},
                    {"type": "text", "text": "Your goal is to give a comprehensive overview of the products in the image, making it easier for users to find similar items online."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=300,
    )
    return response.choices[0].message.content

def get_products(query):
    """Search for products using Google Shopping via SerpAPI"""
    base_url = "https://serpapi.com/search.json"
    params = {
        "engine": "google_shopping",
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "gl": GOOGLE_SHOPPING_REGION
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        top_3_results = data.get("shopping_results", [])[:3]

        if not top_3_results:
            return "No products found for this query."

        results_list = []
        for result in top_3_results:
            product_info = {
                "title": result.get("title", "No title"),
                "price": result.get("price", "Price not available"),
                "link": result.get("link", "#"),
                "image": result.get("thumbnail", "")
            }
            results_list.append(product_info)
        
        return json.dumps(results_list)
    
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return f"Failed to retrieve products: {str(e)}"

def call_functions(required_actions, thread_id, run_id):
    """Execute tool functions required by the assistant"""
    tool_outputs = []

    for action in required_actions["tool_calls"]:
        func_name = action['function']['name']
        arguments = json.loads(action['function']['arguments'])

        if func_name == "search_google_shopping":
            output = get_products(arguments['query'])
            tool_outputs.append({
                "tool_call_id": action['id'],
                "output": output
            })
        else:
            error_msg = f"Unknown function: {func_name}"
            tool_outputs.append({
                "tool_call_id": action['id'],
                "output": error_msg
            })

    shopper_assistant_client.beta.threads.runs.submit_tool_outputs(
        thread_id=thread_id,
        run_id=run_id,
        tool_outputs=tool_outputs
    )

# API Endpoints
@app.post("/chat", tags=["Shopping Assistant"])
async def chat_endpoint(chat_request: ChatMessage):
    """
    Chat with the shopping assistant
    
    - **message**: The user's message
    - **thread_id**: Optional thread ID for continuing a conversation
    """
    # Get or create thread
    thread_id = chat_request.thread_id
    if not thread_id:
        thread = shopper_assistant_client.beta.threads.create()
        thread_id = thread.id
    else:
        # Verify thread exists
        if thread_id not in thread_storage and not chat_request.thread_id.startswith("thread_"):
            # Create new thread if ID doesn't exist
            thread = shopper_assistant_client.beta.threads.create()
            thread_id = thread.id

    # Add to storage
    thread_storage[thread_id] = thread_id
    
    # Add user message to thread
    shopper_assistant_client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=chat_request.message
    )

    # Run the assistant
    run = shopper_assistant_client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=shopping_assistant.id
    )

    # Wait for completion or action requirements
    while True:
        time.sleep(0.5)
        
        # Check run status
        run_status = shopper_assistant_client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run.id
        )
        
        if run_status.status == 'requires_action':
            call_functions(run_status.required_action.submit_tool_outputs.model_dump(), thread_id, run.id)
        
        elif run_status.status == 'completed':
            # Get the latest message
            messages = shopper_assistant_client.beta.threads.messages.list(
                thread_id=thread_id
            )
            
            latest_message = messages.data[0]
            content = latest_message.content[0].text.value
            
            return {
                "response": content,
                "thread_id": thread_id
            }
        
        elif run_status.status == 'failed':
            raise HTTPException(status_code=500, detail="Assistant failed to process the request")
        
        time.sleep(1)

@app.post("/analyze_product", tags=["Product Analysis"])
async def analyze_product_endpoint(file: UploadFile = File(...)):
    """
    Analyze a product image
    
    - **file**: Product image to analyze
    """
    try:
        contents = await file.read()
        encoded_image = base64.b64encode(contents).decode('utf-8')
        analysis = analyse_image(encoded_image)
        
        return {
            "analysis": analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing image: {str(e)}")

@app.post("/try_on", tags=["Virtual Try-On"])
async def try_on_endpoint(
    garment_img: UploadFile = File(...),
    person_img: UploadFile = File(...),
    params: TryOnRequest = Body(...)
):
    """
    Virtual try-on simulation
    
    - **garment_img**: Image of the garment
    - **person_img**: Image of the person
    - **garment_type**: Type of garment (upper, lower, full)
    - **sleeve_length**: Sleeve length (short, 3/4, long)
    - **garment_length**: Garment length (crop, regular, long)
    """
    try:
        # Read the uploaded images
        garment_contents = await garment_img.read()
        person_contents = await person_img.read()
        
        # In a real implementation, you would call the try-on service here
        # For this example, we'll simulate a response
        
        # Simulate processing delay
        time.sleep(2)
        
        return {
            "success": True,
            "message": "Virtual try-on processed successfully",
            "details": {
                "garment_type": params.garment_type,
                "sleeve_length": params.sleeve_length,
                "garment_length": params.garment_length,
                "garment_file": garment_img.filename,
                "person_file": person_img.filename,
            },
            # In a real implementation, you might return a base64 image or URL to the resulting image
            "result_url": "https://example.com/result_image.jpg"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing try-on: {str(e)}")

if __name__ == "__main__":
    # Check if required environment variables are set
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        exit(1)
    if not SERPAPI_API_KEY:
        print("Error: SERPAPI_API_KEY environment variable is not set.")
        exit(1)
        
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)