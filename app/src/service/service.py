from transformers import pipeline
import json
import time
import os
from .logger import get_logger

logger = get_logger("script:classify")
from pathlib import Path
LATENCY_FILE = "latency_log.json"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # folder of service.py
MAPPING_PATH = os.path.join(BASE_DIR, "mapping.json")

# Load category mapping JSON
with open(MAPPING_PATH, "r") as f:
    CATEGORY_MAPPING = json.load(f)

labels = [
    "Coding", "Debugging", "Creative_Writing", "Factual_QA",
    "Summarization", "Translation", "Data_Analysis", "Planning_Itinerary",
    "Sensitive_Medical_Legal", "ChitChat"
]

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli", 
    device=0
)

candidate_labels = [
    "The user is asking about programming or code.",
    "The user is asking for help debugging an error.",
    "The user wants creative writing or storytelling.",
    "The user wants factual general knowledge.",
    "The user wants a summary of text.",
    "The user wants translation to another language.",
    "The user wants data analysis or statistics.",
    "The user is planning a trip, schedule or time.",
    "The user is asking medical or legal questions.",
    "The user is having casual chitchat or greeting."
]

LABEL_MAP = {
    "The user wants creative writing or storytelling.": "Creative_Writing",
    "The user is having casual chitchat or greeting.": "ChitChat",
    "The user wants translation to another language.": "Translation",
    "The user wants a summary of text.": "Summarization",
    "The user is asking for help debugging an error.": "Debugging",
    "The user wants factual general knowledge.": "Factual_QA",
    "The user is asking about programming or code.": "Coding",
    "The user is planning a trip, schedule or time.": "Planning_Itinerary",
    "The user is asking medical or legal questions.": "Sensitive_Medical_Legal",
    "The user wants data analysis or statistics.": "Data_Analysis"
}


def map_to_settings(filtered_labels, filtered_scores):
    """Merge LLM settings for filtered labels only"""
    RE_ORDER = ["minimal", "medium", "high", "maximal"]
    WEB_ORDER = ["disabled", "optional", "mandatory"]
    VERBOSITY_ORDER = ["concise", "balanced", "verbose"]

    logger.info("=== Step: Merging settings from filtered labels ===")
    logger.info(f"Filtered labels: {filtered_labels}")
    logger.info(f"Scores: {filtered_scores}")

    keys = []
    for lbl in filtered_labels:
        if lbl in LABEL_MAP:
            keys.append(LABEL_MAP[lbl])

    logger.info(f"Category keys: {keys}")

    temps = [CATEGORY_MAPPING[k]["temperature"] for k in keys]
    re_values = [RE_ORDER.index(CATEGORY_MAPPING[k]["reasoning_effort"]) for k in keys]
    web_values = [WEB_ORDER.index(CATEGORY_MAPPING[k]["web"]) for k in keys]
    verb_values = [VERBOSITY_ORDER.index(CATEGORY_MAPPING[k]["verbosity"]) for k in keys]

    merged = {
        "temperature": min(temps),
        "reasoning_effort": RE_ORDER[max(re_values)],
        "web": WEB_ORDER[max(web_values)],
        "verbosity": VERBOSITY_ORDER[max(verb_values)]
    }

    logger.info("=== Final merged settings ===")
    logger.info(merged)
    return merged

def update_latency_log(prompt, latency):
    # Load previous log or create new
    if Path(LATENCY_FILE).exists():
        with open(LATENCY_FILE, "r") as f:
            data = json.load(f)
    else:
        data = {"queries": [], "average_latency": 0}

    # Add new entry
    data["queries"].append({"prompt": prompt, "latency": latency})

    # Recalculate average latency
    latencies = [q["latency"] for q in data["queries"]]
    avg_latency = sum(latencies) / len(latencies)
    data["average_latency"] = avg_latency
    # data["average_latency"] = sum(latencies) / len(latencies)

    # Save back
    with open(LATENCY_FILE, "w") as f:
        json.dump(data, f, indent=2)

    # return data["average_latency"]
    return latency, avg_latency
 

def classify_prompt(prompt: str, high_gap=0.15, ratio=0.8, min_threshold=0.2):
    logger.info(f"\nInput Prompt: {prompt}")
    start_time = time.time()
    result = classifier(prompt, candidate_labels, multi_label=True)
    latency = time.time() - start_time

    labels = result["labels"]
    scores = result["scores"]
    labels = result["labels"]
    scores = result["scores"]
    logger.info("Raw classifier output ---")
    logger.info(f"Labels: {labels}")
    logger.info(f"Scores: {scores}")
    logger.info(f"Latency: {latency:.3f} seconds")

    # Step 1: Sort labels by score descending
    sorted_pairs = sorted(zip(labels, scores), key=lambda x: x[1], reverse=True)
    top_label, top_score = sorted_pairs[0]
    second_score = sorted_pairs[1][1] if len(sorted_pairs) > 1 else 0
    delta = top_score - second_score
    logger.info(f"Top score: {top_score:.3f}, Second score: {second_score:.3f}, Delta: {delta:.3f}")

    filtered_labels = []
    filtered_scores = []

    # Step 2: Apply dual-threshold logic
    if delta > high_gap:
        # Top category dominates → only pick top
        filtered_labels = [top_label]
        filtered_scores = [top_score]
        logger.info(f"Dominant category selected: {top_label} ({top_score:.3f})")

    else:
        # Scores close → pick categories >= max(min_threshold, top_score * ratio)
        dynamic_threshold = max(min_threshold, top_score * ratio)
        logger.info(f"Dynamic threshold for inclusion: {dynamic_threshold:.3f}")
        for label, score in sorted_pairs:
            if score >= dynamic_threshold:
                filtered_labels.append(label)
                filtered_scores.append(score)
                logger.info(f"Selected by relative threshold: {label} ({score:.3f})")

    # Step 3: fallback if nothing selected
    if not filtered_labels:
        filtered_labels = [top_label]
        filtered_scores = [top_score]
        logger.info(f"Fallback to top-1 category: {top_label} ({top_score:.3f})")

    # Step 4: build filtered_categories for output
    filtered_categories = [{"name": LABEL_MAP[label], "confidence": score}
                           for label, score in zip(filtered_labels, filtered_scores)]

    # Step 5: calculate merged settings
    settings = map_to_settings(filtered_labels, filtered_scores)
    avg_latency = update_latency_log(prompt, latency)
    logger.info(f"Latency: {latency:.3f}s, Average latency so far: {avg_latency:.3f}s")
    settings["latency_seconds"] = round(latency, 3)

    return filtered_categories, settings
