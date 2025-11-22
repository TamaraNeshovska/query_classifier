from pydantic import BaseModel

class PromptRequest(BaseModel):
    prompt: str

class Category(BaseModel):
    name: str
    confidence: float

class Settings(BaseModel):
    temperature: float
    reasoning_effort: str
    verbosity: str
    web: str

class ClassificationResponse(BaseModel):
    categories: list[Category]
    settings: Settings
