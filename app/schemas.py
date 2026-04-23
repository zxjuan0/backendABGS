# app/schemas.py
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class CheckInCreate(BaseModel):
    goal_id: int
    user_id: int
    notes:   Optional[str] = None

class CheckInResponse(BaseModel):
    id:            int
    goal_id:       int
    user_id:       int
    checked_in_at: datetime
    notes:         Optional[str]

    class Config:
        from_attributes = True   # Pydantic v2 (was orm_mode in v1)

class GoalCreate(BaseModel):
    title:       str
    description: Optional[str] = None
    user_id:     int

class GoalResponse(GoalCreate):
    id:         int
    created_at: datetime
    is_active:  bool
    class Config: from_attributes = True
