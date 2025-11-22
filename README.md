# Real-time Prompt Categorization API

## Overview
This is a prototype service that classifies user prompts into relevant categories in real time and provides recommended LLM settings for generating responses.  
It allows AI-driven applications to adapt dynamically based on the type of user input.

## Features
- Classify prompts into one or more categories, such as:
  - Coding
  - Factual QA
  - Creative Writing
  - Summarization
  - Translation
  - Planning / Itinerary
  - Sensitive Medical / Legal
  - ChitChat
- Recommend LLM settings per prompt:
  - Temperature
  - Reasoning effort
  - Web-use toggle

## API Usage

**Endpoint:**  
`POST /classify`

**Request Body:**  
```json
{
  "prompt": "Explain the difference between classical and quantum physics."
}
```
**Response:**
```json
{
  "categories": [
    { "name": "Coding", "confidence": 0.91 },
    { "name": "Debugging", "confidence": 0.45 }
  ],
  "settings": {
    "temperature": 0.3,
    "reasoning_effort": "medium",
    "verbosity": "balanced",
    "web": "optional"
  }
}
```
**Endpoint:**
`GET /healthcheck`

**Response:**
```json
{
  "status": "ok"
}
```
**Endpoint:**
`GET /latency`

**Response:**
```json
{
  "average_latency_seconds": 1.19
}
```

**Classification Approach:**  
- I chose to use the `facebook/bart-large-mnli` model for zero-shot classification because it provides high accuracy across a wide range of categories.  
- This model is slower compared to smaller alternatives, and it currently cannot be exported to ONNX for faster inference.  
- If ONNX deployment or lower latency is required, a smaller model (e.g., `distilbart-mnli`) could be used instead.  

**Heuristics:**  
- I did not implement any hand-crafted heuristics because the input prompts can be in any format.  
- Implementing rules for such free-form text would be brittle and likely reduce the system's flexibility.

**Future Improvements / Fine-tuning:**  
- If we decide to fine-tune a model instead of using zero-shot classification, we would need at least 100 examples per category to get reasonable performance.  
- For prompts that could belong to multiple categories, we would need around 75–100 examples per combination of categories to properly capture co-occurrence patterns.  
- Synthetic data can help increase dataset size and cover edge cases, but **human validation is important** to ensure quality.  
- I can also provide a script as an example for generating synthetic prompts to augment the dataset.  -> **syntetic_data_generation.py**


## How Classification Works

1. **Candidate Labels**  
   - These are natural language descriptions used by the zero-shot classifier to define possible categories.  
   - Example: `"The user wants creative writing or storytelling."`

2. **LABEL_MAP**  
   - Maps classifier output to clean category names for API responses.  
   - Example: `"The user wants creative writing or storytelling." → "Creative_Writing"`

3. **Thresholds and Delta**  
   - `delta = top_score - second_score`  
   - If `delta > high_gap`, only the top category is selected (dominant).  
   - Otherwise, categories above `max(min_threshold, top_score * ratio)` are included.  
   - Ensures at least one category is always returned.

4. **Settings Merge**  
   - Each category has predefined AI settings (`temperature`, `reasoning_effort`, `web`, `verbosity`).  
   - Merged settings:  
     - `temperature` → minimum  
     - `reasoning_effort` → maximum  
     - `web` → maximum  
     - `verbosity` → maximum  

5. **Example**  
   - Prompt: `"Help me write a blog post about summer"`  
   - Selected categories: `Factual_QA`, `Creative_Writing`  
   - Merged settings returned for AI configuration.


## Getting Started

Follow these steps to set up and run the project locally:

1. **Clone the repository**  
```bash
git clone <repository_url>
cd <repository_folder>
```
2. **Create a virtual environment**
```bash
# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```
3. **Install dependencies**
```bash
pip install -r requirements.txt
```
4. **Run the API**
```bash
uvicorn app.main:app --reload
```

5. **Run the frontend**
#in another terminal
```bash
streamlit run app/frontend/streamlit.py
```