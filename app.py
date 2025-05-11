import os
import time
import base64
import json
import uuid
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Body, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union
import requests
from openai import OpenAI
import uvicorn
from io import BytesIO
from starlette.responses import JSONResponse
from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError
import pandas as pd

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="Enhanced Shopping Assistant API", description="API for shopping assistant, virtual try-on, wardrobe analytics, and weather-based recommendations")

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
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")  # Weather API key

# AWS S3 Configuration
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# OpenAI client setup
shopper_assistant_client = OpenAI(api_key=OPENAI_API_KEY)

# AWS S3 client setup
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

# DynamoDB client for metadata if needed
dynamodb_client = boto3.resource(
    'dynamodb',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

# Create shopping assistant
shopping_assistant = shopper_assistant_client.beta.assistants.create(
    name="Shopping Guide",
    instructions="""As a Shopping Guide, you assist users in making informed purchasing decisions. 
    You ask about their preferences, budget, and the type of product they are looking for, offering options that best match their criteria. 
    You provide comparisons between products, highlighting features, advantages, and disadvantages. 
    You are knowledgeable about a wide range of products and provide guidance on choosing the best option according to the user's needs. 
    You also consider weather conditions when recommending clothing and other weather-sensitive items. 
    You maintain a friendly and helpful tone, ensuring the user feels supported throughout their decision-making process. 
    Avoid suggesting products outside the user's budget or preferences. 
    Instead, focus on finding the best fit within their specified parameters.""",
    model="gpt-4-1106-preview",
    tools=[
        {
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
        },
        {
            "type": "function",
            "function": {
                "name": "get_weather_forecast",
                "description": "Get weather forecast for a location to suggest appropriate products.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city or location for weather forecast."
                        }
                    },
                    "required": ["location"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_user_wardrobe_items",
                "description": "Retrieve wardrobe items for a user to make recommendations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id": {
                            "type": "string",
                            "description": "The user ID to retrieve wardrobe data for."
                        },
                        "category": {
                            "type": "string",
                            "description": "Optional category to filter (tops, bottoms, dresses, etc.)",
                            "default": "all"
                        }
                    },
                    "required": ["user_id"]
                }
            }
        }
    ]
)

# Thread storage - in a production environment, use a database
thread_storage = {}

# Define data models
class ChatMessage(BaseModel):
    message: str
    thread_id: Optional[str] = None
    user_id: str

class TryOnRequest(BaseModel):
    garment_type: str
    sleeve_length: str
    garment_length: str
    user_id: str

class WardrobeItem(BaseModel):
    item_type: str
    color: str
    brand: Optional[str] = None
    season: Optional[str] = None
    style: Optional[str] = None
    material: Optional[str] = None
    occasion: Optional[str] = None

class UserLocation(BaseModel):
    user_id: str
    location: str

# S3 Utility Functions
def save_to_s3(data, key_path, content_type="application/json"):
    """Save data to S3 bucket with specified key path"""
    try:
        if isinstance(data, (dict, list)):
            data = json.dumps(data)
            
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=key_path,
            Body=data,
            ContentType=content_type
        )
        return f"s3://{S3_BUCKET_NAME}/{key_path}"
    except ClientError as e:
        print(f"Error saving to S3: {e}")
        return None

def get_from_s3(key_path):
    """Get data from S3 bucket with specified key path"""
    try:
        response = s3_client.get_object(
            Bucket=S3_BUCKET_NAME,
            Key=key_path
        )
        data = response['Body'].read()
        return data
    except ClientError as e:
        print(f"Error getting from S3: {e}")
        return None

def list_s3_objects(prefix):
    """List objects in S3 bucket with specified prefix"""
    try:
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET_NAME,
            Prefix=prefix
        )
        return [item['Key'] for item in response.get('Contents', [])]
    except ClientError as e:
        print(f"Error listing S3 objects: {e}")
        return []

def save_chat_history(user_id, thread_id, message, role="user"):
    """Save chat message to S3"""
    timestamp = datetime.now().isoformat()
    chat_data = {
        "user_id": user_id,
        "thread_id": thread_id,
        "message": message,
        "role": role,
        "timestamp": timestamp
    }
    
    key_path = f"chats/{user_id}/{thread_id}/{timestamp}.json"
    return save_to_s3(chat_data, key_path)

def save_image_to_s3(user_id, image_data, image_type="product"):
    """Save image to S3 bucket"""
    timestamp = datetime.now().isoformat()
    image_id = str(uuid.uuid4())
    key_path = f"images/{user_id}/{image_type}/{image_id}.jpg"
    
    try:
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=key_path,
            Body=image_data,
            ContentType="image/jpeg"
        )
        return key_path
    except ClientError as e:
        print(f"Error saving image to S3: {e}")
        return None

def save_wardrobe_item(user_id, item_data, image=None):
    """Save wardrobe item data and image to S3"""
    item_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    # Add metadata
    item_data["item_id"] = item_id
    item_data["timestamp"] = timestamp
    
    # Save item data
    data_key_path = f"wardrobe/{user_id}/items/{item_id}.json"
    data_url = save_to_s3(item_data, data_key_path)
    
    # Save image if provided
    image_key_path = None
    if image:
        image_key_path = f"wardrobe/{user_id}/images/{item_id}.jpg"
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=image_key_path,
            Body=image,
            ContentType="image/jpeg"
        )
        
    return {
        "item_id": item_id,
        "data_path": data_key_path,
        "image_path": image_key_path
    }

def get_wardrobe_items(user_id, category="all"):
    """Get wardrobe items for a user, optionally filtered by category"""
    prefix = f"wardrobe/{user_id}/items/"
    item_keys = list_s3_objects(prefix)
    
    items = []
    for key in item_keys:
        try:
            item_data = get_from_s3(key)
            if item_data:
                item_json = json.loads(item_data)
                if category == "all" or item_json.get("item_type") == category:
                    # Add image URL if available
                    item_id = item_json.get("item_id")
                    image_key = f"wardrobe/{user_id}/images/{item_id}.jpg"
                    item_json["image_url"] = f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{image_key}"
                    items.append(item_json)
        except Exception as e:
            print(f"Error processing item {key}: {e}")
    
    return items

# Weather API Functions
def get_weather(location):
    """Get current weather data for a location"""
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": location,
        "appid": WEATHER_API_KEY,
        "units": "metric"  # Use metric units
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Weather API error: {e}")
        return {"error": "Failed to retrieve weather data"}

def get_weather_forecast(location):
    """Get 5-day weather forecast for a location"""
    base_url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {
        "q": location,
        "appid": WEATHER_API_KEY,
        "units": "metric"  # Use metric units
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Process the forecast data to make it more readable
        simplified_forecast = []
        for forecast in data.get("list", [])[:8]:  # Get next 24 hours (3-hour intervals)
            simplified_forecast.append({
                "time": forecast["dt_txt"],
                "temperature": forecast["main"]["temp"],
                "weather": forecast["weather"][0]["main"],
                "description": forecast["weather"][0]["description"],
                "wind_speed": forecast["wind"]["speed"]
            })
        
        result = {
            "location": f"{data['city']['name']}, {data['city']['country']}",
            "forecast": simplified_forecast
        }
        
        return json.dumps(result)
    except requests.RequestException as e:
        print(f"Weather forecast API error: {e}")
        return json.dumps({"error": "Failed to retrieve weather forecast"})

# AI Analysis Functions
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

def analyze_wardrobe(user_id):
    """Analyze a user's wardrobe for insights and recommendations"""
    wardrobe_items = get_wardrobe_items(user_id)
    
    if not wardrobe_items:
        return {"message": "No wardrobe items found for analysis."}
    
    # Convert to dataframe for analysis
    df = pd.DataFrame(wardrobe_items)
    
    # Basic analytics
    item_count_by_type = df['item_type'].value_counts().to_dict()
    color_distribution = df['color'].value_counts().to_dict()
    
    # Find missing essential items
    essential_items = {
        "tops": ["white t-shirt", "button-up shirt", "sweater"],
        "bottoms": ["blue jeans", "black pants", "shorts"],
        "outerwear": ["jacket", "coat"],
        "footwear": ["sneakers", "formal shoes"]
    }
    
    missing_essentials = {}
    for category, items in essential_items.items():
        category_items = df[df['item_type'] == category]['description'].tolist() if 'description' in df.columns else []
        missing = [item for item in items if not any(item.lower() in desc.lower() for desc in category_items)]
        if missing:
            missing_essentials[category] = missing
    
    # Get AI recommendations
    wardrobe_summary = df.to_dict(orient='records')
    ai_recommendations = get_ai_wardrobe_recommendations(wardrobe_summary)
    
    analysis_result = {
        "item_count": len(wardrobe_items),
        "item_distribution": item_count_by_type,
        "color_distribution": color_distribution,
        "missing_essentials": missing_essentials,
        "ai_recommendations": ai_recommendations
    }
    
    # Save analysis to S3
    timestamp = datetime.now().isoformat()
    analysis_key = f"wardrobe/{user_id}/analytics/{timestamp}_analysis.json"
    save_to_s3(analysis_result, analysis_key)
    
    return analysis_result

def get_ai_wardrobe_recommendations(wardrobe_items):
    """Get AI recommendations based on wardrobe items"""
    try:
        response = shopper_assistant_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": """You are a fashion stylist AI that provides wardrobe recommendations. 
                    Analyze the user's wardrobe items and provide the following insights:
                    1. Gaps in their wardrobe and what essential items they should consider adding
                    2. Potential outfit combinations from existing items
                    3. Style recommendations based on their current preferences
                    4. Suggestions for versatile items that would enhance their wardrobe
                    Be specific and practical in your recommendations."""
                },
                {
                    "role": "user",
                    "content": f"Here is my wardrobe data: {json.dumps(wardrobe_items)}. Please provide your analysis and recommendations."
                }
            ],
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error getting AI recommendations: {e}")
        return "Unable to generate recommendations at this time."

def get_weather_based_recommendations(user_id, location):
    """Get outfit recommendations based on weather and user's wardrobe"""
    # Get weather data
    weather_data = get_weather(location)
    
    if "error" in weather_data:
        return {"error": "Failed to retrieve weather data for recommendations"}
    
    # Get user's wardrobe
    wardrobe_items = get_wardrobe_items(user_id)
    
    if not wardrobe_items:
        return {"message": "No wardrobe items found. Please add items to get personalized recommendations."}
    
    # Extract weather conditions
    temperature = weather_data.get("main", {}).get("temp")
    weather_condition = weather_data.get("weather", [{}])[0].get("main")
    
    # Generate AI recommendations
    try:
        response = shopper_assistant_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": """You are a fashion and weather expert. Based on the current weather conditions 
                    and the user's wardrobe, suggest appropriate outfits for the day. Consider temperature, 
                    precipitation, and overall weather conditions when making your recommendations."""
                },
                {
                    "role": "user",
                    "content": f"""Current weather in {location}: 
                    Temperature: {temperature}°C
                    Condition: {weather_condition}
                    
                    My wardrobe items: {json.dumps(wardrobe_items)}
                    
                    Please suggest 2-3 appropriate outfits for today based on the weather and my available clothes.
                    If I'm missing essential items for this weather, please suggest what I should consider purchasing."""
                }
            ],
            max_tokens=500
        )
        
        recommendations = response.choices[0].message.content
        
        result = {
            "weather": {
                "location": location,
                "temperature": temperature,
                "condition": weather_condition
            },
            "recommendations": recommendations
        }
        
        # Save recommendations to S3
        timestamp = datetime.now().isoformat()
        rec_key = f"recommendations/{user_id}/weather/{timestamp}.json"
        save_to_s3(result, rec_key)
        
        return result
        
    except Exception as e:
        print(f"Error getting weather-based recommendations: {e}")
        return {"error": "Failed to generate weather-based recommendations"}

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

def call_functions(required_actions, thread_id, run_id, user_id=None):
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
        elif func_name == "get_weather_forecast":
            output = get_weather_forecast(arguments['location'])
            tool_outputs.append({
                "tool_call_id": action['id'],
                "output": output
            })
        elif func_name == "get_user_wardrobe_items":
            if user_id:
                category = arguments.get('category', 'all')
                items = get_wardrobe_items(user_id, category)
                tool_outputs.append({
                    "tool_call_id": action['id'],
                    "output": json.dumps(items)
                })
            else:
                tool_outputs.append({
                    "tool_call_id": action['id'],
                    "output": json.dumps({"error": "User ID not provided"})
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
    - **user_id**: User identifier for storing chat history
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
    
    # Save user message to S3
    save_chat_history(chat_request.user_id, thread_id, chat_request.message, "user")
    
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
            call_functions(run_status.required_action.submit_tool_outputs.model_dump(), thread_id, run.id, chat_request.user_id)
        
        elif run_status.status == 'completed':
            # Get the latest message
            messages = shopper_assistant_client.beta.threads.messages.list(
                thread_id=thread_id
            )
            
            latest_message = messages.data[0]
            content = latest_message.content[0].text.value
            
            # Save assistant response to S3
            save_chat_history(chat_request.user_id, thread_id, content, "assistant")
            
            return {
                "response": content,
                "thread_id": thread_id
            }
        
        elif run_status.status == 'failed':
            error_message = "Assistant failed to process the request"
            # Save error message to S3
            save_chat_history(chat_request.user_id, thread_id, error_message, "system")
            raise HTTPException(status_code=500, detail=error_message)
        
        time.sleep(1)

@app.post("/analyze_product", tags=["Product Analysis"])
async def analyze_product_endpoint(file: UploadFile = File(...), user_id: str = Form(...)):
    """
    Analyze a product image
    
    - **file**: Product image to analyze
    - **user_id**: User identifier for storing image
    """
    try:
        contents = await file.read()
        
        # Save image to S3
        image_path = save_image_to_s3(user_id, contents, "product_analysis")
        
        # Analyze image
        encoded_image = base64.b64encode(contents).decode('utf-8')
        analysis = analyse_image(encoded_image)
        
        # Save analysis result to S3
        analysis_data = {
            "image_path": image_path,
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }
        
        analysis_path = f"product_analysis/{user_id}/{str(uuid.uuid4())}.json"
        save_to_s3(analysis_data, analysis_path)
        
        return {
            "analysis": analysis,
            "image_path": image_path
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
    - **params**: Contains garment_type, sleeve_length, garment_length, user_id
    """
    try:
        # Read the uploaded images
        garment_contents = await garment_img.read()
        person_contents = await person_img.read()
        
        # Save images to S3
        garment_path = save_image_to_s3(params.user_id, garment_contents, "try_on_garment")
        person_path = save_image_to_s3(params.user_id, person_contents, "try_on_person")
        
        # In a real implementation, you would call the try-on service here
        # For this example, we'll simulate a response
        
        # Simulate processing delay
        time.sleep(2)
        
        result = {
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
            "result_url": "https://example.com/result_image.jpg",
            "garment_path": garment_path,
            "person_path": person_path
        }
        
        # Save result to S3
        result_path = f"try_on_results/{params.user_id}/{str(uuid.uuid4())}.json"
        save_to_s3(result, result_path)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing try-on: {str(e)}")

@app.post("/wardrobe/add_item", tags=["Wardrobe Management"])
async def add_wardrobe_item_endpoint(
    item: WardrobeItem = Body(...),
    image: Optional[UploadFile] = File(None),
    user_id: str = Form(...)
):
    """
    Add an item to the user's digital wardrobe
    
    - **item**: Wardrobe item details
    - **image**: Optional image of the item
    - **user_id**: User identifier
    """
    try:
        # Process image if provided
        image_data = None
        if image:
            image_data = await image.read()
        
        # Convert item to dict and add user_id
        item_dict = item.dict()
        item_dict["user_id"] = user_id
        
        # Save to S3
        result = save_wardrobe_item(user_id, item_dict, image_data)
        
        return {
            "message": "Item added to wardrobe successfully",
            "item_id": result["item_id"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding wardrobe item: {str(e)}")

@app.get("/wardrobe/items/{user_id}", tags=["Wardrobe Management"])
async def get_wardrobe_items_endpoint(user_id: str, category: str = "all"):
    """
    Get wardrobe items for a user
    
    - **user_id**: User identifier
    - **category**: Optional category filter
    """
    try:
        items = get_wardrobe_items(user_id, category)
        
        return {
            "user_id": user_id,
            "category": category,
            "items_count": len(items),
            "items": items
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving wardrobe items: {str(e)}")

@app.get("/wardrobe/analytics/{user_id}", tags=["Wardrobe Analytics"])
async def wardrobe_analytics_endpoint(user_id: str):
    """
    Get analytics for a user's wardrobe
    
    - **user_id**: User identifier
    """
    try:
        analysis = analyze_wardrobe(user_id)
        
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing wardrobe: {str(e)}")

@app.post("/set_user_location", tags=["Weather Integration"])
async def set_user_location_endpoint(location_data: UserLocation):
    """
    Set a user's location for weather-based recommendations
    
    - **user_id**: User identifier
    - **location**: City name or location
    """
    try:
        # Save user location to S3
        location_key = f"user_data/{location_data.user_id}/location.json"
        save_to_s3(location_data.dict(), location_key)
        
        # Get current weather as confirmation
        weather = get_weather(location_data.location)
        
        return {
            "message": f"Location set to {location_data.location}",
            "current_weather": weather
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error setting user location: {str(e)}")

@app.get("/weather_recommendations/{user_id}", tags=["Weather Integration"])
async def weather_recommendations_endpoint(user_id: str, location: Optional[str] = None):
    """
    Get outfit recommendations based on weather and user's wardrobe
    
    - **user_id**: User identifier
    - **location**: Optional location override
    """
    try:
        # Get user's saved location if not provided
        if not location:
            location_key = f"user_data/{user_id}/location.json"
            location_data = get_from_s3(location_key)
            
            if location_data:
                location_json = json.loads(location_data)
                location = location_json.get("location")
            else:
                raise HTTPException(status_code=400, detail="No location provided or saved for user")
        
        recommendations = get_weather_based_recommendations(user_id, location)
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting weather recommendations: {str(e)}")

@app.get("/chat_history/{user_id}", tags=["Chat History"])
async def get_chat_history_endpoint(
    user_id: str, 
    thread_id: Optional[str] = None,
    limit: int = 50,
    skip: int = 0
):
    """
    Get chat history for a user
    
    - **user_id**: User identifier
    - **thread_id**: Optional thread ID to filter by
    - **limit**: Maximum number of messages to return
    - **skip**: Number of messages to skip
    """
    try:
        # Define the prefix based on whether thread_id is provided
        prefix = f"chats/{user_id}/"
        if thread_id:
            prefix = f"chats/{user_id}/{thread_id}/"
        
        # List all chat objects
        chat_keys = list_s3_objects(prefix)
        
        # Sort by timestamp (assuming timestamp is part of the key)
        chat_keys.sort(reverse=True)
        
        # Apply pagination
        paginated_keys = chat_keys[skip:skip+limit]
        
        # Fetch the chat messages
        messages = []
        for key in paginated_keys:
            chat_data = get_from_s3(key)
            if chat_data:
                messages.append(json.loads(chat_data))
        
        return {
            "user_id": user_id,
            "thread_id": thread_id,
            "total_messages": len(chat_keys),
            "messages": messages
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving chat history: {str(e)}")

@app.delete("/wardrobe/items/{user_id}/{item_id}", tags=["Wardrobe Management"])
async def delete_wardrobe_item_endpoint(user_id: str, item_id: str):
    """
    Delete an item from the user's wardrobe
    
    - **user_id**: User identifier
    - **item_id**: Item identifier
    """
    try:
        # Delete item data
        data_key = f"wardrobe/{user_id}/items/{item_id}.json"
        s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=data_key)
        
        # Delete item image if it exists
        image_key = f"wardrobe/{user_id}/images/{item_id}.jpg"
        s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=image_key)
        
        return {
            "message": f"Item {item_id} deleted successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting wardrobe item: {str(e)}")

@app.get("/export_wardrobe/{user_id}", tags=["Wardrobe Management"])
async def export_wardrobe_endpoint(user_id: str, format: str = "json"):
    """
    Export a user's wardrobe data
    
    - **user_id**: User identifier
    - **format**: Export format (json or csv)
    """
    try:
        items = get_wardrobe_items(user_id)
        
        if format.lower() == "csv":
            # Convert to pandas DataFrame and then to CSV
            df = pd.DataFrame(items)
            csv_data = df.to_csv(index=False)
            
            # Save to S3
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_key = f"exports/{user_id}/wardrobe_{timestamp}.csv"
            save_to_s3(csv_data, export_key, "text/csv")
            
            return {
                "message": "Wardrobe exported as CSV",
                "export_path": export_key,
                "download_url": f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{export_key}"
            }
        else:
            # Save as JSON
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_key = f"exports/{user_id}/wardrobe_{timestamp}.json"
            save_to_s3(items, export_key)
            
            return {
                "message": "Wardrobe exported as JSON",
                "export_path": export_key,
                "download_url": f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{export_key}"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting wardrobe: {str(e)}")

@app.get("/health", tags=["Health Check"])
async def health_check():
    """
    Health check endpoint
    """
    # Check if we can connect to S3
    try:
        s3_client.list_buckets()
        s3_status = "healthy"
    except Exception:
        s3_status = "unhealthy"
    
    # Check if we can connect to OpenAI API
    try:
        shopper_assistant_client.models.list()
        openai_status = "healthy"
    except Exception:
        openai_status = "unhealthy"
    
    return {
        "status": "ok",
        "services": {
            "s3": s3_status,
            "openai": openai_status
        },
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    # Check if required environment variables are set
    missing_vars = []
    
    if not OPENAI_API_KEY:
        missing_vars.append("OPENAI_API_KEY")
    if not SERPAPI_API_KEY:
        missing_vars.append("SERPAPI_API_KEY")
    if not AWS_ACCESS_KEY_ID:
        missing_vars.append("AWS_ACCESS_KEY_ID")
    if not AWS_SECRET_ACCESS_KEY:
        missing_vars.append("AWS_SECRET_ACCESS_KEY")
    if not S3_BUCKET_NAME:
        missing_vars.append("S3_BUCKET_NAME")
    if not WEATHER_API_KEY:
        missing_vars.append("WEATHER_API_KEY")
    
    if missing_vars:
        print(f"Error: The following environment variables are not set: {', '.join(missing_vars)}")
        print("Please set these variables in your .env file or environment.")
        exit(1)
        
    print(f"Starting Enhanced Shopping Assistant API on port 8000...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
