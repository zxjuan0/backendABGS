from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ml.predict import predict_all
from app.ml_predictions import predict_all_models


app = FastAPI(
    title="ABGS | Adaptive Behavior Goal System",
    description="AI-powered habit and goal tracking API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://abgs-wine.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Existing prediction input for your original /predict endpoint
class UserFeatures(BaseModel):
    streak_length: float
    missed_days_last_7: float
    checkin_hour_avg: float
    engagement_freq: float
    days_since_last: float
    goal_age_days: float


# New prediction input for the 4-model Client A comparison
class ClientPredictionInput(BaseModel):
    habit_type: str
    activity_type: str
    difficulty_level: str
    streak_length: int
    missed_days_last_7: int
    checkin_hour_avg: int
    engagement_freq: float
    days_since_last: int
    goal_age_days: int
    pomodoro_sessions: int
    pomodoro_completion_rate: float


@app.get("/")
def root():
    return {"message": "ABGS backend is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


# Existing endpoint | this for my current working prediction flow
@app.post("/predict")
def predict(features: UserFeatures):
    return predict_all(features.dict())


# New endpoint — use this for Client A / 4-model comparison dashboard
@app.post("/predict/client-a")
def predict_client_a(features: ClientPredictionInput):
    return predict_all_models(features.dict())