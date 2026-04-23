# app/routes/checkins.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app import crud, schemas
from app.database import get_db

router = APIRouter(prefix="/checkins", tags=["Check-ins"])

@router.post("/", response_model=schemas.CheckInResponse)
def log_checkin(checkin: schemas.CheckInCreate, db: Session = Depends(get_db)):
    """Log a daily habit check-in for a user/goal."""
    return crud.create_checkin(db=db, checkin=checkin)

@router.get("/{user_id}", response_model=list[schemas.CheckInResponse])
def get_user_checkins(user_id: int, db: Session = Depends(get_db)):
    """Get all check-ins for a specific user."""
    return crud.get_checkins(db=db, user_id=user_id)
