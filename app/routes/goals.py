from fastapi import APIRouter

router = APIRouter()

@router.get("/example")
def example():
    return {"message": "Hello from goals"}