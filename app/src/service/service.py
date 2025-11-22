from transformers import pipeline
import json
import time
from pathlib import Path
import torch
from .logger import get_logger

logger = get_logger("script:classify")

# --- Paths ---
BASE_DIR = Path(__file__).parent
MAPPING_PATH = BASE_DIR / "mapping.json"
LATENCY_FILE = BASE_DIR / "latency_log.json"
print(f"latency file {LATENCY_FILE.resolve()}")
LATENCY_FILE2 = Path(__file__).parent / "latency_log.json"
print("Latency file path:", LATENCY_FILE2.resolve())

# --- Load category mapping ---
with MAPPING_PATH.open("r") as f:
    CATEGORY_MAPPING = json.load(f)

# --- Labels and candidate labels ---
labels = [
    "Coding", "Debugging", "Creative_Writing", "Factual_QA",
    "Summarization", "Translation", "Data_Analysis", "Planning_Itinerary",
    "Sensitive_Medical_Legal", "ChitChat"
]

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

# --- Precompute order indices ---
RE_ORDER = ["minimal", "medium", "high", "maximal"]
WEB_ORDER = ["disabled", "optional", "mandatory"]
VERBOSITY_ORDER = ["concise", "balanced", "verbose"]

RE_INDEX = {v: i for i, v in enumerate(RE_ORDER)}
WEB_INDEX = {v: i for i, v in enumerate(WEB_ORDER)}
VERBOSITY_INDEX = {v: i for i, v in enumerate(VERBOSITY_ORDER)}

# --- Detect device ---
device = 0 if torch.cuda.is_available() else -1
logger.info(f"Using device: {'GPU' if device >= 0 else 'CPU'}")

# --- Load classifier once ---
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=device
)


# --- Helper functions ---
def get_default_settings():
    fallback = next(iter(CATEGORY_MAPPING.values()), {})
    return {
        "temperature": fallback.get("temperature", 0.7),
        "reasoning_effort": fallback.get("reasoning_effort", "medium"),
        "web": fallback.get("web", "optional"),
        "verbosity": fallback.get("verbosity", "balanced")
    }


def map_to_settings(filtered_labels, filtered_scores):
    """Merge LLM settings for filtered labels only"""
    logger.info("=== Step: Merging settings from filtered labels ===")
    logger.info(f"Filtered labels: {filtered_labels}")
    logger.info(f"Scores: {filtered_scores}")

    keys = [LABEL_MAP[lbl] for lbl in filtered_labels if lbl in LABEL_MAP]
    logger.info(f"Category keys: {keys}")

    try:
        temps = [CATEGORY_MAPPING[k]["temperature"] for k in keys]
        re_values = [RE_INDEX[CATEGORY_MAPPING[k]["reasoning_effort"]] for k in keys]
        web_values = [WEB_INDEX[CATEGORY_MAPPING[k]["web"]] for k in keys]
        verb_values = [VERBOSITY_INDEX[CATEGORY_MAPPING[k]["verbosity"]] for k in keys]

        merged = {
            "temperature": min(temps),
            "reasoning_effort": RE_ORDER[max(re_values)],
            "web": WEB_ORDER[max(web_values)],
            "verbosity": VERBOSITY_ORDER[max(verb_values)]
        }
    except KeyError as e:
        logger.warning(f"Missing category mapping for key: {e}. Using default settings.")
        merged = get_default_settings()

    logger.info(f"=== Final merged settings ===\n{merged}")
    return merged


def update_latency_log(prompt, latency):
    try:
        data = {"queries": [], "average_latency": 0}
        if LATENCY_FILE.exists():
            with LATENCY_FILE.open("r") as f:
                data = json.load(f)

        data["queries"].append({"prompt": prompt, "latency": latency})
        avg_latency = sum(q["latency"] for q in data["queries"]) / len(data["queries"])
        data["average_latency"] = avg_latency

        with LATENCY_FILE.open("w") as f:
            json.dump(data, f, indent=2)

        return latency, avg_latency
    except Exception as e:
        logger.warning(f"Failed to update latency log: {e}")
        return latency, latency


def log_selected_labels(labels, scores, msg="Selected labels"):
    for lbl, sc in zip(labels, scores):
        logger.info(f"{msg}: {lbl} ({sc:.3f})")


# --- Main classification function ---
def classify_prompt(prompt: str, high_gap=0.15, ratio=0.8, min_threshold=0.2):
    logger.info(f"\nInput Prompt: {prompt}")
    start_time = time.time()

    try:
        result = classifier(prompt, candidate_labels, multi_label=True)
        labels_out, scores_out = result["labels"], result["scores"]
    except Exception as e:
        logger.error(f"Classifier error: {e}")
        labels_out, scores_out = [], []

    latency = round(time.time() - start_time, 3)
    if not labels_out:
        logger.warning("No classifier output. Returning defaults.")
        settings = get_default_settings()
        settings["latency_seconds"] = latency
        return [], settings

    # --- Dual-threshold selection ---
    sorted_pairs = sorted(zip(labels_out, scores_out), key=lambda x: x[1], reverse=True)
    top_label, top_score = sorted_pairs[0]
    second_score = sorted_pairs[1][1] if len(sorted_pairs) > 1 else 0
    delta = top_score - second_score

    if delta > high_gap:
        filtered = [(top_label, top_score)]
    else:
        threshold = max(min_threshold, top_score * ratio)
        filtered = [(lbl, sc) for lbl, sc in sorted_pairs if sc >= threshold]

    if not filtered:
        filtered = [(top_label, top_score)]

    filtered_labels, filtered_scores = zip(*filtered)
    log_selected_labels(filtered_labels, filtered_scores)

    # --- Build output ---
    filtered_categories = [{"name": LABEL_MAP[label], "confidence": score}
                           for label, score in filtered]

    settings = map_to_settings(filtered_labels, filtered_scores)
    _, avg_latency = update_latency_log(prompt, latency)
    settings["latency_seconds"] = latency

    logger.info(f"Latency: {latency}s, Avg latency: {avg_latency}s")
    return filtered_categories, settings

