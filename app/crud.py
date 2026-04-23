# app/crud.py
from sqlalchemy.orm import Session
from app import models, schemas

def create_checkin(db: Session, checkin: schemas.CheckInCreate):
    db_checkin = models.CheckIn(**checkin.dict())
    db.add(db_checkin)
    db.commit()
    db.refresh(db_checkin)
    return db_checkin

def get_checkins(db: Session, user_id: int, skip: int = 0, limit: int = 50):
    return db.query(models.CheckIn)\
             .filter(models.CheckIn.user_id == user_id)\
             .offset(skip).limit(limit).all()

def create_goal(db: Session, goal: schemas.GoalCreate):
    db_goal = models.Goal(**goal.dict())
    db.add(db_goal)
    db.commit()
    db.refresh(db_goal)
    return db_goal

def get_goals(db: Session, user_id: int):
    return db.query(models.Goal)\
             .filter(models.Goal.user_id == user_id).all()

