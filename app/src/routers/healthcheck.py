from fastapi import APIRouter

# Initialize a new router
router = APIRouter()

@router.get("/healthcheck")
async def healthcheck():
    try:
        return {
            "status": "ok"
        }
    except Exception as e:
        return {
            "message": "Error occurred in healthcheck",
            "error": str(e)
        }
