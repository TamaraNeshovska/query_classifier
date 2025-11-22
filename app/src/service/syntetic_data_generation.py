import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class SyntheticUserDataGenerator:
    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4.1-mini"
        self.temperature = 0.8  
     
    def create_prompt(self, category, num_examples=5):
        prompt = f"""
            You are a data generation assistant. Your task is to generate {num_examples} **human-like user queries** for the category "Coding". 

            These queries should **look exactly like a real user typed them into a search bar or chatbot**, meaning: 
            - A mix of short keyword-style fragments (e.g., 'reverse list python') 
            - And normal question-style queries (e.g., 'how to implement reverse function in python') 
            - Typos or missing punctuation are okay
            - Only the essential words a user would type, casual phrasing

            Focus on **coding, programming tasks, writing code, or implementing algorithms**. 
            Do NOT include queries about errors or debugging. 

            Provide the output as a JSON array of objects with the following format:

            [
                {{"example": "user query here", "category": "Coding"}}
            ]

            **Important guidelines:**
            1. Each example should be completely unique.
            2. Make it look like real user input in a search bar or chatbot.
            3. Include **some queries with code snippets typed naturally**, at least 40% of all queries.
            4. Generate a mix of short fragments and full question-style queries.
            5. Do NOT include any queries about errors, debugging, or troubleshooting.
            6. Ensure the category field is exactly "Coding".

            **Here are 8 sample examples for guidance:**
            [
                {{"example": "reverse list python", "category": "Coding"}},
                {{"example": "sort array fastest js", "category": "Coding"}},
                {{"example": "python count vowels string function", "category": "Coding"}},
                {{"example": "how to implement reverse function in python", "category": "Coding"}},
                {{"example": "javascript function to flatten nested arrays", "category": "Coding"}},
                {{"example": "python: def reverse_list(lst): return lst[::-1] ?", "category": "Coding"}},
                {{"example": "js: arr.map(x => x*2) example", "category": "Coding"}},
                {{"example": "for i in range(10) print(i)", "category": "Coding"}}
            ]
        """
        return prompt

      
    def create_medical_prompt(self, category, num_examples=5):
        prompt = f"""
            You are a data generation assistant. Your task is to generate {num_examples} **human-like user queries** for the category "Sensitive_Medical_Legal". 

            These queries should **look exactly like a real user typed them into a search bar or chatbot**, meaning: 
            - A mix of short keyword-style fragments (e.g., 'high ALP level causes') 
            - And normal question-style queries (e.g., 'what does a high ALP level mean for my liver health?') 
            - Typos or missing punctuation are okay
            - Only the essential words a user would type, casual phrasing

            Focus on **medical symptoms, lab results, health conditions, medications, treatments, legal issues, patient rights, or healthcare regulations**.  
            Do NOT include generic chit-chat or unrelated topics. 

            Provide the output as a JSON array of objects with the following format:

            [
                {{"example": "user query here", "category": "Sensitive_Medical_Legal"}}
            ]

            **Important guidelines:**
            1. Each example should be completely unique.
            2. Make it look like real user input in a search bar or chatbot.
            3. Include **both medical and legal related queries**, roughly 60% medical, 40% legal.
            4. Include **some queries with specific numbers, lab results, or medical terminology typed naturally**.
            5. Generate a mix of short fragments and full question-style queries.
            6. Do NOT include queries about coding, debugging, or general knowledge.
            7. Ensure the category field is exactly "Sensitive_Medical_Legal".

            **Here are 8 sample examples for guidance:**
            [
                {{"example": "high ALP level causes", "category": "Sensitive_Medical_Legal"}},
                {{"example": "what does a red bump on my arm mean", "category": "Sensitive_Medical_Legal"}},
                {{"example": "can my doctor share my medical records legally", "category": "Sensitive_Medical_Legal"}},
                {{"example": "ibuprofen dosage for 35kg child", "category": "Sensitive_Medical_Legal"}},
                {{"example": "patient rights in hospital admission", "category": "Sensitive_Medical_Legal"}},
                {{"example": "symptoms of low hemoglobin", "category": "Sensitive_Medical_Legal"}},
                {{"example": "how to write medical consent form legally", "category": "Sensitive_Medical_Legal"}},
                {{"example": "bp 150/90 is this dangerous", "category": "Sensitive_Medical_Legal"}}
            ]
            """
        return prompt   

    def create_travel_prompt(self, category, num_examples=5):
        prompt = f"""
            You are a data generation assistant. Your task is to generate {num_examples} **human-like user queries** for the category "Planning_Itinerary". 

            These queries should **look exactly like a real user typed them into a search bar or chatbot**, meaning: 
            - A mix of short keyword-style fragments (e.g., 'study schedule next week') 
            - And normal question-style queries (e.g., 'how should I plan my study schedule for finals next week?') 
            - Typos or missing punctuation are okay
            - Only the essential words a user would type, casual phrasing

            Focus on **structured planning or scheduling requests**, including:  
            - Travel itineraries, sightseeing, hotels, restaurants  
            - Daily or weekly study plans  
            - Work schedules, meetings, or event agendas  
            - Step-by-step task organization  

            Do NOT include casual chit-chat, general questions, or unrelated topics. 

            Provide the output as a JSON array of objects with the following format:

            [
                {{"example": "user query here", "category": "Planning_Itinerary"}}
            ]

            **Important guidelines:**
            1. Each example should be completely unique.
            2. Make it look like real user input in a search bar or chatbot.
            3. Include a mix of **short fragments and full question-style queries**.
            4. Include **some queries with dates, locations, or specific activities typed naturally**.
            5. Generate a variety of planning types: travel, study, work, meetings, or multi-day schedules.
            6. Do NOT include queries about coding, debugging, medical, or legal topics.
            7. Ensure the category field is exactly "Planning_Itinerary".

            **Here are 10 sample examples for guidance:**
            [
                {{"example": "5 day norway hiking trip", "category": "Planning_Itinerary"}},
                {{"example": "how to plan 3 days in paris with museums and cafes", "category": "Planning_Itinerary"}},
                {{"example": "study schedule next week for finals", "category": "Planning_Itinerary"}},
                {{"example": "organize monday work tasks by priority", "category": "Planning_Itinerary"}},
                {{"example": "weeklong hiking trip colorado national parks", "category": "Planning_Itinerary"}},
                {{"example": "daily routine for productivity and study", "category": "Planning_Itinerary"}},
                {{"example": "plan team meeting agenda 3 hours", "category": "Planning_Itinerary"}},
                {{"example": "things to do in london 5 days sightseeing", "category": "Planning_Itinerary"}},
                {{"example": "schedule gym and study sessions for week", "category": "Planning_Itinerary"}},
                {{"example": "plan weekend getaway amsterdam flights hotels", "category": "Planning_Itinerary"}}
            ]
            """
        return prompt
    
    def create_data_analysis_prompt(self, category, num_examples=5):
        prompt = f"""
            You are a data generation assistant. Your task is to generate {num_examples} human-like user queries for the category "Data_Analysis".

            These queries must look like real inputs typed into a search bar or chatbot, including:
            - Short keyword-style fragments (e.g., "correlation matrix pandas")
            - Natural question-style queries (e.g., "How do I analyze this CSV file?")
            - Occasional typos or missing punctuation
            - Casual, human phrasing rather than perfect textbook language

            The "Data_Analysis" category includes any user query where a person:
            - Provides data and asks for analysis (e.g., "here is the data analyze it")
            - Works with datasets in Python, SQL, Excel, R, or raw text
            - Requests statistical calculations or aggregations
            - **Asks for analysis of macro-level data (e.g., market trends, economic indicators, industry reports)**
            - Asks to clean, filter, transform, or preprocess data
            - Wants insights, trends, summaries, or interpretations
            - Asks about plots or data visualizations (charts, graphs)
            - Asks to interpret metrics, distributions, or results

            Allowed data formats:
            - numbers, text, logs, tables, JSON, CSV, Excel, lists, mixed content, or referencing a market/economy (like 'Macedonian market')

            Do NOT generate:
            - Debugging requests or error messages
            - Pure coding questions unrelated to analyzing data
            - Medical, legal, or off-topic queries

            Output format (strict):
            [
                {{"example": "user query here", "category": "Data_Analysis"}}
            ]

            Requirements:
            1. Every example must be unique.
            2. Mix short fragment-style queries and full sentences.
            3. Include queries that explicitly mention data being provided (e.g., "here is the dataset, analyze it") OR a large market/topic (like 'Macedonian market').
            4. Include queries referencing common data terms: dataset names, columns, metrics, charts, etc.
            5. Keep the category field exactly equal to "Data_Analysis".

            Here are sample examples for style guidance (do NOT repeat them):
            [
                {{"example": "calculate correlation pandas two cols", "category": "Data_Analysis"}},
                {{"example": "how to find outliers in dataset", "category": "Data_Analysis"}},
                {{"example": "sql group by month sum revenue", "category": "Data_Analysis"}},
                {{"example": "summarize sales.csv find trends", "category": "Data_Analysis"}},
                {{"example": "scatter plot age vs income python", "category": "Data_Analysis"}},
                {{"example": "interpret confusion matrix precision recall", "category": "Data_Analysis"}},
                {{"example": "find anomalies in time series data", "category": "Data_Analysis"}},
                {{"example": "analyze the macedonian market tech sector growth data", "category": "Data_Analysis"}}
            ]
        """
        return prompt
    
    def create_factual_qa_prompt(self, category="Factual_QA", num_examples=5):
        prompt = f"""
            You are a data generation assistant. Your task is to generate {num_examples} human-like user queries for the category "Factual_QA".

            These queries must look like real inputs typed into a search bar or chatbot, including:
            - Short keyword-style fragments (e.g., "capital of france")
            - Natural question-style queries (e.g., "When did World War 2 end?")
            - Occasional typos or missing punctuation
            - Casual, human phrasing rather than perfect textbook language

            The "Factual_QA" category includes any user query where a person asks for a **specific, verifiable, objective piece of information or fact** that can be found in a general knowledge base.

            The answer should ideally be short (a name, date, number, definition, or short statement).

            "Factual_QA" includes questions about:
            - **People:** Names, dates of birth/death, historical roles (e.g., "who invented the telephone")
            - **Places:** Capitals, geographical features, famous landmarks (e.g., "tallest mountain")
            - **Definitions/Concepts:** Simple explanations of scientific, historical, or common terms (e.g., "what is photosynthesis")
            - **Events/Dates:** Specific dates or timelines for historical or well-known events (e.g., "when was the internet invented")
            - **Specific Numerical Values:** Population, measurements, statistics (e.g., "population of canada")

            Do NOT generate:
            - **Subjective/Opinion-based questions** (e.g., "what is the best movie")
            - **Hypothetical questions** (e.g., "what if gravity stopped working")
            - **Analysis requests** (e.g., "analyze this dataset" - this is Data_Analysis)
            - **Coding or Debugging requests**
            - Medical, legal, or off-topic queries

            Output format (strict):
            [
                {{"example": "user query here", "category": "Factual_QA"}}
            ]

            Requirements:
            1. Every example must be unique.
            2. Mix short fragment-style queries and full sentences.
            3. Include a variety of topics (history, science, geography, etc.).
            4. Include a query that might have a slight typo or casual phrasing.
            5. Keep the category field exactly equal to "Factual_QA".

            Here are sample examples for style guidance (do NOT repeat them):
            [
                {{"example": "who was the first president of the usa", "category": "Factual_QA"}},
                {{"example": "what is the largest ocean in the world", "category": "Factual_QA"}},
                {{"example": "define the term neural network", "category": "Factual_QA"}},
                {{"example": "speed of light in a vacum", "category": "Factual_QA"}},
                {{"example": "what year did the cold war end", "category": "Factual_QA"}},
                {{"example": "current CEO of google", "category": "Factual_QA"}},
                {{"example": "where is the great barrier reef located", "category": "Factual_QA"}},
                {{"example": "mass of the sun in kilograms", "category": "Factual_QA"}}
            ]
        """
        return prompt
    

    def create_creative_writing_prompt(self, category="Creative_Writing", num_examples=5):
        prompt = f"""
            You are a data generation assistant. Your task is to generate {num_examples} human-like user queries for the category "{category}".

            These queries must look like real inputs typed into a search bar or chatbot, including:
            - Requests that ask for generation of original, imaginative, or expressive text.
            - Casual, human phrasing, often phrased as a "write me a..." or "help me draft..."
            - Instructions that include a specific tone, style, character, or plot element.
            - Queries that focus on **style, emotion, narrative, or literary technique.**

            The "{category}" category includes any user query where a person asks for the **creation of original, expressive, and non-factual text** that requires imagination, emotional depth, or a specific literary style.

            Creative_Writing tasks include:
            - **Fiction/Narrative:** Generating plot ideas, character descriptions, short scenes, or stories.
            - **Poetry/Verse:** Writing poems, lyrics, rhymes, or verses on a given theme or emotion.
            - **Scripts/Dialogue:** Creating short dialogue exchanges or basic scene descriptions for a play/film.
            - **Conceptual/Stylistic Writing:** Drafting something in a specific fictional voice, or using heavy figurative language (e.g., a metaphor for aging).
            - **Idea Generation/Prompts:** Asking for suggestions, concepts, plot twists, or starting points for a creative project.

            Do NOT generate (and why it belongs to another category):
            - **Summarization/Extraction** (e.g., "summarize this article"): This is **Summarization**.
            - **Factual Q&A** (e.g., "what is the capital of spain"): This is **Factual_QA**.
            - **Analysis/Critique** (e.g., "analyze the theme of Hamlet"): This is **Text_Analysis**.
            - **Code/Technical** (e.g., "write a python function"): This is **Code_Generation**.
            - **Formal/Professional** (e.g., "draft a resignation letter"): This is **Professional_Writing**.

            Output format (strict):
            [
                {{"example": "user query here", "category": "{category}"}}
            ]

            Requirements:
            1. Every example must be unique and clearly fall under "{category}".
            2. Mix requests for different creative forms (story, poem, description, dialogue, **ideas**).
            3. Include a variety of themes (sad, action, fantasy, everyday life).
            4. Include a query that might have a slight typo or casual phrasing.
            5. Keep the category field exactly equal to "{category}".

            Here are sample examples for style guidance (do NOT repeat them):
            [
                {{"example": "suggest me some unique plot twists for a murder mystery in space", "category": "Creative_Writing"}},
                {{"example": "Can you draft a poem about the color blue but make it sound sad", "category": "Creative_Writing"}},
                {{"example": "help me write the opening lines of a fantasy novel where the hero is scared", "category": "Creative_Writing"}},
                {{"example": "dialogue between a wise old wizard and a nervous apprentice about a bad omen", "category": "Creative_Writing"}},
                {{"example": "describe a bustling cyberpunk city at midnight using vivid imagery", "category": "Creative_Writing"}}
            ]
        """
        return prompt
    
    def create_debugging_prompt(self, category="Debugging", num_examples=5):
        prompt = f"""
            You are a data generation assistant. Your task is to generate {num_examples} human-like user queries for the category "{category}".

            These queries must look like real inputs typed into a search bar or chatbot, including:
            - Requests that ask for **identifying, analyzing, or fixing a problem** in provided code, system output, or logic.
            - Casual, human phrasing, often phrased as a "why is my..." or "how to fix this..."
            - Instructions that include a specific **error message, expected vs. actual output, or performance issue.**
            - Queries that focus on **locating an error source (bug), fixing a malfunction, or understanding an unexpected failure.**

            The "{category}" category includes any user query where a person asks for assistance with an **existing problem, error, or failure** in code or a technical process. The core intent is to *resolve* an issue.

            Debugging tasks include:
            - **Error Analysis:** Providing an error message (e.g., KeyError, 500 status) and asking for the cause or fix.
            - **Logic Correction:** Explaining the current incorrect output/behavior and asking for the line of code that contains the mistake.
            - **Performance Issue:** Asking why a specific function or script is running too slowly or using too much memory.
            - **Troubleshooting:** Asking for steps to diagnose a known, unexpected behavior (e.g., "why does this loop run forever?").

            Do NOT generate (and why it belongs to another category):
            - **Generation of new, complete code/functions** (e.g., "write a Python function to sort a list"): This is **Code_Generation**.
            - **Factual Q&A about programming** (e.g., "what is a closure in JavaScript"): This is **Factual_QA**.
            - **Code Conversion/Translation** (e.g., "convert this C++ loop to Python"): This is **Code_Conversion**.
            - **Documentation/Explanation of working code** (e.g., "explain what this API route does"): This is **Text_Analysis**.

            Output format (strict):
            [
                {{"example": "user query here", "category": "{category}"}}
            ]

            Requirements:
            1. Every example must be unique and clearly fall under "{category}".
            2. Mix requests for different problem types (errors, logic, performance).
            3. Include specific, realistic error messages or problem descriptions.
            4. Include a query that might have a slight typo or casual phrasing.
            5. Keep the category field exactly equal to "{category}".

            Here are sample examples for style guidance (do NOT repeat them):
            [
                {{"example": "I'm getting 'IndexError: list index out of range'. How do I fix this in my python loop?", "category": "Debugging"}},
                {{"example": "my SQL query keeps returning empty results even though I know the data is there, what did I miss?", "category": "Debugging"}},
                {{"example": "Why is my JavaScript function returning 'undefined' instead of the calculation result? I think it's a scope problem.", "category": "Debugging"}},
                {{"example": "my website loads so slow sometimes. what's the first thing i shud check for performance problems?", "category": "Debugging"}},
                {{"example": "The API call keeps failing with a 404 error but the URL is correct. help me figure out why.", "category": "Debugging"}}
            ]
        """
        return prompt
    
    def generate_examples(self, category, num_examples=10):
        prompt = self.create_debugging_prompt(category, num_examples)

        messages = [
            {"role": "system", "content": "You are an expert synthetic data generator for human-like queries."},
            {"role": "user", "content": prompt},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                top_p=1,
                max_tokens=2000,
            )

            content = response.choices[0].message.content

            # Try to parse JSON
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                result = {"raw_response": content}

            return result

        except Exception as e:
            raise Exception(f"Failed to generate synthetic data: {str(e)}")

def save_to_json_append(data, filename="synthetic_debugging_data.json"):
    """Append new data to existing JSON file (or create if it doesn't exist)."""
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = []
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []

    # Append new batch
    if isinstance(data, list):
        existing_data.extend(data)
    else:
        existing_data.append(data)

    # Save combined data
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(data)} new examples, total {len(existing_data)} examples in {filename}")


if __name__ == "__main__":
    generator = SyntheticUserDataGenerator()
    category = "Debugging"  # Example category
    
    total_examples = 100
    batch_size = 20

    for _ in range(total_examples // batch_size):
        print(f" batch generating...")
        batch = generator.generate_examples(category, num_examples=batch_size)
        save_to_json_append(batch, filename=f"{category}_synthetic_data.json")