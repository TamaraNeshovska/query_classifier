from fastapi import APIRouter, HTTPException
from ..schemas.schemas import PromptRequest, ClassificationResponse
from ..service.service import classify_prompt
from ..service.logger import get_logger
router = APIRouter()

logger = get_logger("script:classify")

@router.post("/classify", response_model=ClassificationResponse)
async def classify(req: PromptRequest):
    try:
        categories, settings = classify_prompt(req.prompt)
        return ClassificationResponse(categories=categories, settings=settings)
    except Exception as e:
        logger.error(f"Error in classify endpoint: {e}")
        raise HTTPException(status_code=500, detail="Failed to classify prompt")