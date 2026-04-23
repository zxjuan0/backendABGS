import joblib
import numpy as np

model = joblib.load("ml/models/dropout_model.joblib")
scaler = joblib.load("ml/models/scaler.joblib")

FEATURES = [
    "streak_length",
    "missed_days_last_7",
    "checkin_hour_avg",
    "engagement_freq",
    "days_since_last",
    "goal_age_days"
]

def predict_dropout(user_features: dict) -> dict:
    x = np.array([[user_features[f] for f in FEATURES]])
    x_scaled = scaler.transform(x)
    prob = model.predict_proba(x_scaled)[0][1]
    risk = "high" if prob > 0.7 else ("medium" if prob > 0.4 else "low")

    return {
        "dropout_probability": round(float(prob), 3),
        "risk_level": risk
    }