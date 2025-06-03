import os
import time
import base64
import json
import uuid
from datetime import datetime, timedelta
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
import firebase_admin
from firebase_admin import credentials, firestore, storage
from gradio_client import Client, file
import tempfile
import pandas as pd
from PIL import Image
import numpy as np

# Load environment variables
load_dotenv()

app = FastAPI(title="Enhanced Shopping Assistant API with Firebase", 
              description="Complete shopping assistant with Firebase integration, virtual try-on, and AI recommendations")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment Variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
FIREBASE_SERVICE_ACCOUNT_PATH = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH", "serviceAccountKey.json")
FIREBASE_STORAGE_BUCKET = os.getenv("FIREBASE_STORAGE_BUCKET")

# Initialize Firebase
try:
    cred = credentials.Certificate(FIREBASE_SERVICE_ACCOUNT_PATH)
    firebase_admin.initialize_app(cred, {
        'storageBucket': FIREBASE_STORAGE_BUCKET
    })
    db = firestore.client()
    bucket = storage.bucket()
    print("Firebase initialized successfully")
except Exception as e:
    print(f"Firebase initialization error: {e}")
    # For development, create mock objects
    db = None
    bucket = None

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Gradio clients for various AI services
try:
    tryon_client = Client("yisol/IDM-VTON")
    print("Virtual try-on client initialized")
except Exception as e:
    print(f"Try-on client initialization error: {e}")
    tryon_client = None

# Data Models
class ChatMessage(BaseModel):
    message: str
    user_id: str
    thread_id: Optional[str] = None

class WardrobeItem(BaseModel):
    item_type: str  # tops, bottoms, dresses, shoes, accessories
    color: str
    brand: Optional[str] = None
    season: Optional[str] = None  # spring, summer, fall, winter
    style: Optional[str] = None
    material: Optional[str] = None
    occasion: Optional[str] = None  # casual, formal, business, party
    description: Optional[str] = None
    price: Optional[float] = None
    purchase_date: Optional[str] = None

class OutfitPlan(BaseModel):
    name: str
    occasion: str
    weather_condition: Optional[str] = None
    items: List[str]  # List of item IDs
    date_planned: Optional[str] = None

class UserPreferences(BaseModel):
    preferred_colors: List[str] = []
    preferred_brands: List[str] = []
    preferred_styles: List[str] = []
    size_info: Dict[str, str] = {}
    budget_range: Dict[str, float] = {}

class TryOnRequest(BaseModel):
    person_image_url: str
    garment_image_url: str
    garment_description: str

# Firebase Helper Functions
def save_to_firestore(collection: str, document_id: str, data: dict):
    """Save data to Firestore"""
    try:
        if db:
            db.collection(collection).document(document_id).set(data, merge=True)
            return True
        return False
    except Exception as e:
        print(f"Error saving to Firestore: {e}")
        return False

def get_from_firestore(collection: str, document_id: str):
    """Get data from Firestore"""
    try:
        if db:
            doc = db.collection(collection).document(document_id).get()
            return doc.to_dict() if doc.exists else None
        return None
    except Exception as e:
        print(f"Error getting from Firestore: {e}")
        return None

def query_firestore(collection: str, filters: List = None):
    """Query Firestore collection with optional filters"""
    try:
        if db:
            query = db.collection(collection)
            if filters:
                for field, operator, value in filters:
                    query = query.where(field, operator, value)
            return [doc.to_dict() for doc in query.stream()]
        return []
    except Exception as e:
        print(f"Error querying Firestore: {e}")
        return []

def upload_to_firebase_storage(file_data, file_path: str):
    """Upload file to Firebase Storage"""
    try:
        if bucket:
            blob = bucket.blob(file_path)
            blob.upload_from_string(file_data, content_type='image/jpeg')
            blob.make_public()
            return blob.public_url
        return None
    except Exception as e:
        print(f"Error uploading to Firebase Storage: {e}")
        return None

def download_from_firebase_storage(file_path: str):
    """Download file from Firebase Storage"""
    try:
        if bucket:
            blob = bucket.blob(file_path)
            return blob.download_as_bytes()
        return None
    except Exception as e:
        print(f"Error downloading from Firebase Storage: {e}")
        return None

# AI Analysis Functions
def analyze_clothing_image(image_data):
    """Analyze clothing image using OpenAI Vision"""
    try:
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        response = client.chat.completions.create(
            model="gpt-4o",  # Updated to latest model
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Analyze this clothing item and provide detailed information in JSON format:
                            {
                                "item_type": "type of clothing (tops, bottoms, dresses, shoes, accessories)",
                                "color": "primary color",
                                "style": "style description",
                                "material": "apparent material",
                                "occasion": "suitable occasion",
                                "season": "suitable season",
                                "description": "detailed description",
                                "tags": ["relevant", "tags", "for", "categorization"]
                            }"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        
        content = response.choices[0].message.content
        # Try to extract JSON from the response
        try:
            # Find JSON in the response
            start = content.find('{')
            end = content.rfind('}') + 1
            json_str = content[start:end]
            return json.loads(json_str)
        except:
            # If JSON parsing fails, return a basic structure
            return {
                "item_type": "unknown",
                "color": "unknown",
                "style": "unknown",
                "material": "unknown",
                "occasion": "unknown",
                "season": "unknown",
                "description": content,
                "tags": []
            }
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return None

def get_ai_recommendations(user_id: str, occasion: str = "casual", weather: str = None):
    """Get AI-powered outfit recommendations based on user's wardrobe"""
    try:
        # Get user's wardrobe items
        wardrobe_items = query_firestore("wardrobe", [("user_id", "==", user_id)])
        
        if not wardrobe_items:
            return {"message": "No wardrobe items found for recommendations"}
        
        # Get user preferences
        preferences = get_from_firestore("user_preferences", user_id) or {}
        
        # Get recent outfit history
        recent_outfits = query_firestore("outfit_history", [
            ("user_id", "==", user_id),
            ("date_worn", ">=", (datetime.now() - timedelta(days=7)).isoformat())
        ])
        
        # Prepare context for AI
        context = {
            "wardrobe_items": wardrobe_items,
            "preferences": preferences,
            "recent_outfits": recent_outfits,
            "occasion": occasion,
            "weather": weather
        }
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": """You are a professional fashion stylist AI. Based on the user's wardrobe, preferences, and context, provide personalized outfit recommendations. 
                    
                    Consider:
                    1. Occasion appropriateness
                    2. Weather conditions
                    3. Color coordination
                    4. Style consistency
                    5. Recently worn items (avoid repetition)
                    6. User preferences
                    
                    Provide 3-5 complete outfit suggestions with explanations."""
                },
                {
                    "role": "user",
                    "content": f"Please recommend outfits based on this data: {json.dumps(context)}"
                }
            ],
            max_tokens=800
        )
        
        recommendations = response.choices[0].message.content
        
        # Save recommendations to Firestore
        recommendation_data = {
            "user_id": user_id,
            "recommendations": recommendations,
            "occasion": occasion,
            "weather": weather,
            "timestamp": datetime.now().isoformat(),
            "context": context
        }
        
        rec_id = str(uuid.uuid4())
        save_to_firestore("recommendations", rec_id, recommendation_data)
        
        return {
            "recommendations": recommendations,
            "recommendation_id": rec_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Error getting AI recommendations: {e}")
        return {"error": "Failed to generate recommendations"}

def analyze_wardrobe_patterns(user_id: str):
    """Analyze user's wardrobe usage patterns and provide insights"""
    try:
        # Get all wardrobe items
        wardrobe_items = query_firestore("wardrobe", [("user_id", "==", user_id)])
        
        # Get outfit history
        outfit_history = query_firestore("outfit_history", [("user_id", "==", user_id)])
        
        if not wardrobe_items:
            return {"message": "No wardrobe data available for analysis"}
        
        # Analyze patterns
        df = pd.DataFrame(wardrobe_items)
        
        analysis = {
            "total_items": len(wardrobe_items),
            "items_by_type": df['item_type'].value_counts().to_dict() if 'item_type' in df.columns else {},
            "items_by_color": df['color'].value_counts().to_dict() if 'color' in df.columns else {},
            "items_by_season": df.get('season', pd.Series()).value_counts().to_dict(),
            "items_by_occasion": df.get('occasion', pd.Series()).value_counts().to_dict(),
        }
        
        # Calculate wear frequency
        item_wear_count = {}
        for outfit in outfit_history:
            for item_id in outfit.get('items', []):
                item_wear_count[item_id] = item_wear_count.get(item_id, 0) + 1
        
        # Find most and least worn items
        if item_wear_count:
            sorted_items = sorted(item_wear_count.items(), key=lambda x: x[1], reverse=True)
            analysis["most_worn_items"] = sorted_items[:5]
            analysis["least_worn_items"] = sorted_items[-5:]
        
        # Get AI insights
        ai_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a wardrobe analytics expert. Analyze the provided wardrobe data and provide insights about usage patterns, gaps, and recommendations for optimization."
                },
                {
                    "role": "user",
                    "content": f"Analyze this wardrobe data: {json.dumps(analysis)}"
                }
            ],
            max_tokens=500
        )
        
        analysis["ai_insights"] = ai_response.choices[0].message.content
        analysis["timestamp"] = datetime.now().isoformat()
        
        # Save analysis
        save_to_firestore("wardrobe_analytics", f"{user_id}_{int(time.time())}", analysis)
        
        return analysis
        
    except Exception as e:
        print(f"Error analyzing wardrobe patterns: {e}")
        return {"error": "Failed to analyze wardrobe patterns"}

def get_weather_forecast(location: str):
    """Get weather forecast for outfit recommendations"""
    try:
        base_url = "https://api.openweathermap.org/data/2.5/forecast"
        params = {
            "q": location,
            "appid": WEATHER_API_KEY,
            "units": "metric"
        }
        
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Simplify forecast data
        forecast = []
        for item in data.get("list", [])[:8]:  # Next 24 hours
            forecast.append({
                "time": item["dt_txt"],
                "temperature": item["main"]["temp"],
                "weather": item["weather"][0]["main"],
                "description": item["weather"][0]["description"]
            })
        
        return {
            "location": f"{data['city']['name']}, {data['city']['country']}",
            "forecast": forecast
        }
    except Exception as e:
        print(f"Weather API error: {e}")
        return {"error": "Failed to get weather data"}

def search_products(query: str):
    """Search for products using SerpAPI"""
    try:
        base_url = "https://serpapi.com/search.json"
        params = {
            "engine": "google_shopping",
            "q": query,
            "api_key": SERPAPI_API_KEY,
            "gl": "US"
        }
        
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        products = []
        for result in data.get("shopping_results", [])[:5]:
            products.append({
                "title": result.get("title", ""),
                "price": result.get("price", ""),
                "link": result.get("link", ""),
                "image": result.get("thumbnail", ""),
                "source": result.get("source", "")
            })
        
        return products
    except Exception as e:
        print(f"Product search error: {e}")
        return []
    
import base64
import tempfile
import os
import uuid
from openai import OpenAI

client = OpenAI()

def virtual_try_on_gpt(person_image_data, garment_image_data, garment_description):
    """Perform virtual try-on using OpenAI's image editing model"""
    try:
        # Create temporary files for OpenAI API
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as person_temp:
            person_temp.write(person_image_data)
            person_path = person_temp.name
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as garment_temp:
            garment_temp.write(garment_image_data)
            garment_path = garment_temp.name
        
        try:
            # Use your original prompt logic
            prompt = """
Generate a photorealistic image of a person using that dress on image 
containing all the items in the reference pictures.
"""
            
            # Use your original OpenAI API call logic
            result = client.images.edit(
                model="gpt-image-1",
                image=[
                    open(person_path, "rb"),
                    open(garment_path, "rb"),
                ],
                prompt=prompt
            )
            
            # Use your original image processing logic
            image_base64 = result.data[0].b64_json
            image_bytes = base64.b64decode(image_base64)
            
            # Save the image to a file (keeping your original save logic)
            output_filename = "gift-basket.png"
            with open(output_filename, "wb") as f:
                f.write(image_bytes)
            
            # Clean up temporary files
            os.unlink(person_path)
            os.unlink(garment_path)
            
            return {
                "result_file": output_filename,
                "result_base64": image_base64,
                "status": "success"
            }
            
        except Exception as api_error:
            # Clean up temporary files on error
            try:
                os.unlink(person_path)
                os.unlink(garment_path)
            except:
                pass
            
            print(f"OpenAI API error: {api_error}")
            return {
                "error": "Virtual try-on service failed",
                "details": str(api_error),
                "status": "api_error"
            }
        
    except Exception as e:
        print(f"Virtual try-on error: {e}")
        return {"error": "Virtual try-on failed", "details": str(e)}


# API Endpoints

@app.post("/chat", tags=["AI Assistant"])
async def chat_with_assistant(chat_request: ChatMessage):
    """Chat with the AI shopping assistant"""
    try:
        # Get user context
        user_data = get_from_firestore("users", chat_request.user_id) or {}
        wardrobe_items = query_firestore("wardrobe", [("user_id", "==", chat_request.user_id)])
        preferences = get_from_firestore("user_preferences", chat_request.user_id) or {}
        
        # Create context-aware system message
        system_message = f"""You are a personal fashion and shopping assistant. You have access to the user's wardrobe data and preferences.
        
        User's wardrobe: {json.dumps(wardrobe_items[:10])}  # Limit to avoid token limits
        User's preferences: {json.dumps(preferences)}
        
        Provide helpful fashion advice, outfit suggestions, shopping recommendations, and style guidance based on their existing wardrobe and preferences."""
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": chat_request.message}
            ],
            max_tokens=500
        )
        
        assistant_response = response.choices[0].message.content
        
        # Save chat history
        chat_data = {
            "user_id": chat_request.user_id,
            "thread_id": chat_request.thread_id or str(uuid.uuid4()),
            "message": chat_request.message,
            "response": assistant_response,
            "timestamp": datetime.now().isoformat()
        }
        
        chat_id = str(uuid.uuid4())
        save_to_firestore("chat_history", chat_id, chat_data)
        
        return {
            "response": assistant_response,
            "thread_id": chat_data["thread_id"],
            "timestamp": chat_data["timestamp"]
        }
        
    except Exception as e:
        print(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Chat service failed")

@app.post("/wardrobe/upload", tags=["Wardrobe Management"])
async def upload_wardrobe_item(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    manual_data: Optional[str] = Form(None)
):
    """Upload and categorize a wardrobe item with automatic image recognition"""
    try:
        # Read image data
        image_data = await file.read()
        
        # Analyze image with AI
        analysis = analyze_clothing_image(image_data)
        
        if not analysis:
            raise HTTPException(status_code=500, detail="Image analysis failed")
        
        # Merge with manual data if provided
        if manual_data:
            try:
                manual_info = json.loads(manual_data)
                analysis.update(manual_info)
            except json.JSONDecodeError:
                pass
        
        # Generate unique item ID
        item_id = str(uuid.uuid4())
        
        # Upload image to Firebase Storage
        image_path = f"wardrobe/{user_id}/{item_id}.jpg"
        image_url = upload_to_firebase_storage(image_data, image_path)
        
        # Prepare wardrobe item data
        wardrobe_item = {
            "item_id": item_id,
            "user_id": user_id,
            "image_url": image_url,
            "upload_date": datetime.now().isoformat(),
            **analysis
        }
        
        # Save to Firestore
        save_to_firestore("wardrobe", item_id, wardrobe_item)
        
        return {
            "message": "Wardrobe item uploaded successfully",
            "item_id": item_id,
            "analysis": analysis,
            "image_url": image_url
        }
        
    except Exception as e:
        print(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload wardrobe item")

@app.get("/wardrobe/{user_id}", tags=["Wardrobe Management"])
async def get_user_wardrobe(user_id: str, category: Optional[str] = None):
    """Get user's wardrobe items, optionally filtered by category"""
    try:
        filters = [("user_id", "==", user_id)]
        if category:
            filters.append(("item_type", "==", category))
        
        items = query_firestore("wardrobe", filters)
        
        return {
            "user_id": user_id,
            "category": category,
            "items_count": len(items),
            "items": items
        }
        
    except Exception as e:
        print(f"Get wardrobe error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve wardrobe")

@app.post("/recommendations", tags=["AI Recommendations"])
async def get_outfit_recommendations(
    user_id: str = Body(...),
    occasion: str = Body("casual"),
    location: Optional[str] = Body(None)
):
    """Get AI-powered outfit recommendations"""
    try:
        weather_data = None
        if location:
            weather_data = get_weather_forecast(location)
        
        recommendations = get_ai_recommendations(
            user_id=user_id,
            occasion=occasion,
            weather=weather_data
        )
        
        return recommendations
        
    except Exception as e:
        print(f"Recommendations error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate recommendations")

@app.post("/outfit/plan", tags=["Outfit Planning"])
async def create_outfit_plan(
    user_id: str = Body(...),
    outfit_name: str = Body(...),
    occasion: str = Body(...),
    item_ids: List[str] = Body(...),
    date_planned: Optional[str] = Body(None)
):
    """Create and save an outfit plan"""
    try:
        outfit_plan = {
            "plan_id": str(uuid.uuid4()),
            "user_id": user_id,
            "name": outfit_name,
            "occasion": occasion,
            "items": item_ids,
            "date_planned": date_planned,
            "created_date": datetime.now().isoformat()
        }
        
        save_to_firestore("outfit_plans", outfit_plan["plan_id"], outfit_plan)
        
        return {
            "message": "Outfit plan created successfully",
            "plan_id": outfit_plan["plan_id"],
            "outfit_plan": outfit_plan
        }
        
    except Exception as e:
        print(f"Outfit planning error: {e}")
        raise HTTPException(status_code=500, detail="Failed to create outfit plan")

@app.get("/outfit/plans/{user_id}", tags=["Outfit Planning"])
async def get_outfit_plans(user_id: str):
    """Get user's saved outfit plans"""
    try:
        plans = query_firestore("outfit_plans", [("user_id", "==", user_id)])
        
        return {
            "user_id": user_id,
            "plans_count": len(plans),
            "plans": plans
        }
        
    except Exception as e:
        print(f"Get outfit plans error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve outfit plans")


@app.post("/try-on", tags=["Virtual Try-On"])
async def virtual_try_on_endpoint(
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    user_id: str = Form(...),
    garment_description: str = Form(...)
):
    """Perform virtual try-on using OpenAI's image editing model"""
    try:
        person_data = await person_image.read()
        garment_data = await garment_image.read()
        
        # Perform virtual try-on with GPT
        result = virtual_try_on_gpt(person_data, garment_data, garment_description)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        # Save try-on result to Firestore
        tryon_data = {
            "tryon_id": str(uuid.uuid4()),
            "user_id": user_id,
            "garment_description": garment_description,
            "timestamp": datetime.now().isoformat(),
            "result_url": result["result_url"],
            "status": result["status"]
        }
        
        save_to_firestore("tryon_results", tryon_data["tryon_id"], tryon_data)
        
        return {
            "message": "Virtual try-on completed successfully",
            "tryon_id": tryon_data["tryon_id"],
            "result_url": result["result_url"],
            "result_base64": result.get("result_base64")
        }
        
    except Exception as e:
        print(f"Virtual try-on endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Virtual try-on failed")

@app.get("/analytics/{user_id}", tags=["Wardrobe Analytics"])
async def get_wardrobe_analytics(user_id: str):
    """Get comprehensive wardrobe analytics and insights"""
    try:
        analytics = analyze_wardrobe_patterns(user_id)
        return analytics
        
    except Exception as e:
        print(f"Analytics error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate analytics")

@app.post("/shopping/search", tags=["E-commerce Integration"])
async def search_products_endpoint(
    query: str = Body(...),
    user_id: str = Body(...)
):
    """Search for products and get shopping suggestions"""
    try:
        products = search_products(query)
        
        # Save search history
        search_data = {
            "search_id": str(uuid.uuid4()),
            "user_id": user_id,
            "query": query,
            "results": products,
            "timestamp": datetime.now().isoformat()
        }
        
        save_to_firestore("search_history", search_data["search_id"], search_data)
        
        return {
            "query": query,
            "results_count": len(products),
            "products": products
        }
        
    except Exception as e:
        print(f"Product search error: {e}")
        raise HTTPException(status_code=500, detail="Product search failed")

@app.post("/preferences", tags=["User Management"])
async def update_user_preferences(
    user_id: str = Body(...),
    preferences: UserPreferences = Body(...)
):
    """Update user's style preferences"""
    try:
        pref_data = preferences.dict()
        pref_data["user_id"] = user_id
        pref_data["updated_date"] = datetime.now().isoformat()
        
        save_to_firestore("user_preferences", user_id, pref_data)
        
        return {
            "message": "Preferences updated successfully",
            "preferences": pref_data
        }
        
    except Exception as e:
        print(f"Preferences update error: {e}")
        raise HTTPException(status_code=500, detail="Failed to update preferences")

@app.get("/preferences/{user_id}", tags=["User Management"])
async def get_user_preferences(user_id: str):
    """Get user's style preferences"""
    try:
        preferences = get_from_firestore("user_preferences", user_id)
        
        if not preferences:
            return {"message": "No preferences found", "preferences": {}}
        
        return {"preferences": preferences}
        
    except Exception as e:
        print(f"Get preferences error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve preferences")

@app.post("/outfit/worn", tags=["Usage Tracking"])
async def record_outfit_worn(
    user_id: str = Body(...),
    item_ids: List[str] = Body(...),
    occasion: str = Body(...),
    date_worn: Optional[str] = Body(None)
):
    """Record when an outfit was worn for analytics"""
    try:
        if not date_worn:
            date_worn = datetime.now().isoformat()
        
        outfit_record = {
            "record_id": str(uuid.uuid4()),
            "user_id": user_id,
            "items": item_ids,
            "occasion": occasion,
            "date_worn": date_worn,
            "recorded_date": datetime.now().isoformat()
        }
        
        save_to_firestore("outfit_history", outfit_record["record_id"], outfit_record)
        
        return {
            "message": "Outfit wear recorded successfully",
            "record_id": outfit_record["record_id"]
        }
        
    except Exception as e:
        print(f"Record outfit error: {e}")
        raise HTTPException(status_code=500, detail="Failed to record outfit")

@app.delete("/wardrobe/{item_id}", tags=["Wardrobe Management"])
async def delete_wardrobe_item(item_id: str, user_id: str = Query(...)):
    """Delete a wardrobe item"""
    try:
        # Verify item belongs to user
        item = get_from_firestore("wardrobe", item_id)
        
        if not item or item.get("user_id") != user_id:
            raise HTTPException(status_code=404, detail="Item not found or access denied")
        
        # Delete from Firestore
        if db:
            db.collection("wardrobe").document(item_id).delete()
        
        # Delete image from storage if exists
        if item.get("image_url"):
            try:
                image_path = f"wardrobe/{user_id}/{item_id}.jpg"
                blob = bucket.blob(image_path)
                blob.delete()
            except:
                pass  # Image deletion is not critical
        
        return {"message": "Wardrobe item deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Delete item error: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete item")

@app.get("/weather/{location}", tags=["Weather Integration"])
async def get_weather_for_outfit(location: str):
    """Get weather information for outfit planning"""
    try:
        weather_data = get_weather_forecast(location)
        
        if "error" in weather_data:
            raise HTTPException(status_code=500, detail=weather_data["error"])
        
        return weather_data
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Weather endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get weather data")

@app.get("/dashboard/{user_id}", tags=["Dashboard"])
async def get_user_dashboard(user_id: str):
    """Get comprehensive dashboard data for user"""
    try:
        # Get all user data
        wardrobe_items = query_firestore("wardrobe", [("user_id", "==", user_id)])
        recent_outfits = query_firestore("outfit_history", [
            ("user_id", "==", user_id),
            ("date_worn", ">=", (datetime.now() - timedelta(days=30)).isoformat())
        ])
        outfit_plans = query_firestore("outfit_plans", [("user_id", "==", user_id)])
        preferences = get_from_firestore("user_preferences", user_id) or {}
        
        # Calculate basic stats
        stats = {
            "total_items": len(wardrobe_items),
            "recent_wears": len(recent_outfits),
            "saved_plans": len(outfit_plans),
            "favorite_colors": {},
            "most_worn_type": {},
            "wardrobe_value": 0
        }
        
        # Calculate wardrobe insights
        if wardrobe_items:
            colors = [item.get('color', 'unknown') for item in wardrobe_items]
            types = [item.get('item_type', 'unknown') for item in wardrobe_items]
            
            stats["favorite_colors"] = {color: colors.count(color) for color in set(colors)}
            stats["most_worn_type"] = {type_: types.count(type_) for type_ in set(types)}
            stats["wardrobe_value"] = sum(item.get('price', 0) for item in wardrobe_items if item.get('price'))
        
        return {
            "user_id": user_id,
            "stats": stats,
            "recent_activity": recent_outfits[-5:] if recent_outfits else [],
            "wardrobe_preview": wardrobe_items[:10],
            "preferences": preferences,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Dashboard error: {e}")
        raise HTTPException(status_code=500, detail="Failed to load dashboard data")

@app.get("/items/{item_id}", tags=["Item Details"])
async def get_item_details(item_id: str, user_id: str = Query(...)):
    """Get detailed information about a specific wardrobe item"""
    try:
        item = get_from_firestore("wardrobe", item_id)
        
        if not item or item.get("user_id") != user_id:
            raise HTTPException(status_code=404, detail="Item not found or access denied")
        
        # Get wear history for this item
        wear_history = query_firestore("outfit_history", [
            ("user_id", "==", user_id),
            ("items", "array_contains", item_id)
        ])
        
        # Calculate usage stats
        item["wear_count"] = len(wear_history)
        item["last_worn"] = max([outfit["date_worn"] for outfit in wear_history]) if wear_history else None
        item["wear_history"] = wear_history
        
        return item
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Item details error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get item details")

@app.put("/wardrobe/{item_id}", tags=["Wardrobe Management"])
async def update_wardrobe_item(
    item_id: str,
    user_id: str = Body(...),
    updates: Dict[str, Any] = Body(...)
):
    """Update a wardrobe item's information"""
    try:
        # Verify item belongs to user
        item = get_from_firestore("wardrobe", item_id)
        
        if not item or item.get("user_id") != user_id:
            raise HTTPException(status_code=404, detail="Item not found or access denied")
        
        # Update the item
        updates["updated_date"] = datetime.now().isoformat()
        save_to_firestore("wardrobe", item_id, updates)
        
        # Get updated item
        updated_item = get_from_firestore("wardrobe", item_id)
        
        return {
            "message": "Item updated successfully",
            "item": updated_item
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Update item error: {e}")
        raise HTTPException(status_code=500, detail="Failed to update item")

@app.get("/export/{user_id}", tags=["Data Export"])
async def export_user_data(user_id: str, format: str = Query("json")):
    """Export user's wardrobe and outfit data"""
    try:
        # Get all user data
        wardrobe_items = query_firestore("wardrobe", [("user_id", "==", user_id)])
        outfit_history = query_firestore("outfit_history", [("user_id", "==", user_id)])
        outfit_plans = query_firestore("outfit_plans", [("user_id", "==", user_id)])
        preferences = get_from_firestore("user_preferences", user_id) or {}
        
        export_data = {
            "user_id": user_id,
            "export_date": datetime.now().isoformat(),
            "wardrobe_items": wardrobe_items,
            "outfit_history": outfit_history,
            "outfit_plans": outfit_plans,
            "preferences": preferences,
            "summary": {
                "total_items": len(wardrobe_items),
                "total_outfits_worn": len(outfit_history),
                "total_plans": len(outfit_plans)
            }
        }
        
        return export_data
        
    except Exception as e:
        print(f"Export error: {e}")
        raise HTTPException(status_code=500, detail="Failed to export data")

@app.post("/import/{user_id}", tags=["Data Import"])
async def import_user_data(
    user_id: str,
    data: Dict[str, Any] = Body(...)
):
    """Import wardrobe and outfit data for a user"""
    try:
        imported_count = {
            "wardrobe_items": 0,
            "outfit_plans": 0,
            "preferences": 0
        }
        
        # Import wardrobe items
        if "wardrobe_items" in data:
            for item in data["wardrobe_items"]:
                item["user_id"] = user_id
                item["import_date"] = datetime.now().isoformat()
                item_id = item.get("item_id", str(uuid.uuid4()))
                save_to_firestore("wardrobe", item_id, item)
                imported_count["wardrobe_items"] += 1
        
        # Import outfit plans
        if "outfit_plans" in data:
            for plan in data["outfit_plans"]:
                plan["user_id"] = user_id
                plan["import_date"] = datetime.now().isoformat()
                plan_id = plan.get("plan_id", str(uuid.uuid4()))
                save_to_firestore("outfit_plans", plan_id, plan)
                imported_count["outfit_plans"] += 1
        
        # Import preferences
        if "preferences" in data:
            preferences = data["preferences"]
            preferences["user_id"] = user_id
            preferences["import_date"] = datetime.now().isoformat()
            save_to_firestore("user_preferences", user_id, preferences)
            imported_count["preferences"] = 1
        
        return {
            "message": "Data imported successfully",
            "imported_count": imported_count,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Import error: {e}")
        raise HTTPException(status_code=500, detail="Failed to import data")

@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint"""
    try:
        # Check Firebase connection
        firebase_status = "connected" if db else "disconnected"
        
        # Check OpenAI API
        openai_status = "connected" if OPENAI_API_KEY else "not configured"
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "firebase": firebase_status,
                "openai": openai_status,
                "weather_api": "configured" if WEATHER_API_KEY else "not configured",
                "serp_api": "configured" if SERPAPI_API_KEY else "not configured",
                "virtual_tryon": "available" if tryon_client else "unavailable"
            }
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/", tags=["System"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Enhanced Shopping Assistant API",
        "version": "1.0.0",
        "description": "Complete shopping assistant with Firebase integration, virtual try-on, and AI recommendations",
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health"
        },
        "features": [
            "AI-powered wardrobe analysis",
            "Personalized outfit recommendations",
            "Virtual try-on capabilities",
            "Weather-based suggestions",
            "Shopping integration",
            "Usage analytics",
            "Outfit planning"
        ]
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"message": "Endpoint not found", "detail": str(exc)}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error", "detail": str(exc)}
    )

# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
