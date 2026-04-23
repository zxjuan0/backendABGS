  # app/models.py
from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base

class User(Base):
    __tablename__ = "users"
    id         = Column(Integer, primary_key=True, index=True)
    username   = Column(String, unique=True, index=True)
    email      = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    goals      = relationship("Goal", back_populates="owner")

class Goal(Base):
    __tablename__ = "goals"
    id          = Column(Integer, primary_key=True, index=True)
    title       = Column(String, nullable=False)
    description = Column(String)
    user_id     = Column(Integer, ForeignKey("users.id"))
    created_at  = Column(DateTime, default=datetime.utcnow)
    is_active   = Column(Boolean, default=True)
    owner       = relationship("User", back_populates="goals")
    checkins    = relationship("CheckIn", back_populates="goal")

class CheckIn(Base):
    __tablename__ = "checkins"
    id            = Column(Integer, primary_key=True, index=True)
    goal_id       = Column(Integer, ForeignKey("goals.id"))
    user_id       = Column(Integer, ForeignKey("users.id"))
    checked_in_at = Column(DateTime, default=datetime.utcnow)
    notes         = Column(String, nullable=True)
    goal          = relationship("Goal", back_populates="checkins")