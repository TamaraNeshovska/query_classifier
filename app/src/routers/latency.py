from pathlib import Path
from fastapi import APIRouter
import json

# Initialize a new router
router = APIRouter()

LATENCY_FILE = Path(__file__).parent.parent / "service" / "latency_log.json"
print("Latency file path:", LATENCY_FILE.resolve())


@router.get("/latency")
def get_average_latency():
    if LATENCY_FILE.exists():
        with open(LATENCY_FILE, "r") as f:
            data = json.load(f)
        return {"average_latency_seconds": data.get("average_latency", 0)}
    return {"average_latency_seconds": 0}
