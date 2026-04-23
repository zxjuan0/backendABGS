from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ml.predict import predict_dropout

app = FastAPI(
    title="ABGS - Adaptive Behavior Goal System",
    description="AI-powered habit and goal tracking API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserFeatures(BaseModel):
    streak_length: float
    missed_days_last_7: float
    checkin_hour_avg: float
    engagement_freq: float
    days_since_last: float
    goal_age_days: float

@app.get("/")
def root():
    return {"message": "ABGS backend is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(features: UserFeatures):
    return predict_dropout(features.dict())