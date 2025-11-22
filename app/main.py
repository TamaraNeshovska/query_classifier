from fastapi import FastAPI
from app.src.routers.classify import router as classify_router
from app.src.routers.healthcheck import router as health_router
from app.src.routers.latency import router as latency_router
app = FastAPI()

# include endpoints
app.include_router(health_router)
app.include_router(classify_router)
app.include_router(latency_router)