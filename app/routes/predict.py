from fastapi import APIRouter
from ml.predict import predict_dropout
from pydantic import BaseModel

router = APIRouter(prefix="/predict", tags=["ML Predictions"])

class UserFeatures(BaseModel):
    streak_length:     float
    missed_days_last_7: float
    checkin_hour_avg:  float
    engagement_freq:   float
    days_since_last:   float
    goal_age_days:     float

@router.post("/")
def get_prediction(features: UserFeatures):
    return predict_dropout(features.dict())
