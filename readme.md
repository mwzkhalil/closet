## Getting Started

### Prerequisites

- Python 3.8 or higher
- An OpenAI API key
- A SerpAPI key

### Step 1: Clone the Repository

```bash
git clone https://github.com/mwzkhalil/closet.git
cd closet
```

### Step 2: Set Up a Virtual Environment

#### For Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### For macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables

1. Copy the `.env.example` file to a new file named `.env`:
   ```bash
   cp .env.example .env
   ```

2. Open the `.env` file and replace the placeholder values with your actual API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   SERPAPI_API_KEY=your_serpapi_key_here
   GOOGLE_SHOPPING_REGION=CH
   ```

### Step 5: Run the API

```bash
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`.

## API Endpoints

### 1. Chat 

```
POST /chat
```

**Request Body:**
```json
{
  "message": "I'm looking for running shoes under $100",
  "thread_id": "123"
}
```

**Response:**
```json
{
  "response": "I can help you find running shoes under $100. What type of running do you do?",
  "thread_id": "thread_abc123"
}
```

### 2. Analyze Product Image

```
POST /analyze_product
```

**Request:**
- Form data with a file named "file" containing the product image

**Response:**
```json
{
  "analysis": "This image shows a pair of Nike Air Zoom Pegasus 38 running shoes in blue/white colorway. The shoes feature a mesh upper material with Flywire technology and a Zoom Air cushioning system..."
}
```

### 3. Virtual Try-On

```
POST /try_on
```

**Request:**
- Form data with:
  - "garment_img": Image file of the garment
  - "person_img": Image file of the person
  - JSON body with:
    ```json
    {
      "garment_type": "upper",
      "sleeve_length": "long",
      "garment_length": "regular"
    }
    ```

**Response:**
```json
{
  "success": true,
  "message": "Virtual try-on processed successfully",
  "details": {
    "garment_type": "upper",
    "sleeve_length": "long",
    "garment_length": "regular",
    "garment_file": "sweater.jpg",
    "person_file": "model.jpg"
  },
  "result_url": "https://example.com/result_image.jpg"
}
```


### Project Structure

```
shopping-assistant-api/
├── app.py              # Main application file
├── .env                # Environment variables (not in git)
├── .env.example        # Example environment variables
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

- [FastAPI](https://fastapi.tiangolo.com/)
- [OpenAI](https://openai.com/)
- [SerpAPI](https://serpapi.com/)
