from fastapi import APIRouter
from ..schemas.schemas import PromptRequest, ClassificationResponse
from ..service.service import classify_prompt

router = APIRouter()

@router.post("/classify", response_model=ClassificationResponse)
async def classify(req: PromptRequest):
    categories, settings = classify_prompt(req.prompt)
    return ClassificationResponse(categories=categories, settings=settings)
