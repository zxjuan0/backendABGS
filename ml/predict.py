import joblib
import pandas as pd

FEATURES = [
    "streak_length",
    "missed_days_last_7",
    "checkin_hour_avg",
    "engagement_freq",
    "days_since_last",
    "goal_age_days"
]

# Load all models
model_200 = joblib.load("ml/models/model_200.pkl")
model_20k = joblib.load("ml/models/model_20k.pkl")
model_200k = joblib.load("ml/models/model_200k.pkl")


def predict_all(user_features: dict):
    x = pd.DataFrame([[user_features[f] for f in FEATURES]], columns=FEATURES)

    def run_model(model):
        prob = model.predict_proba(x)[0][1]
        risk = "high" if prob > 0.7 else ("medium" if prob > 0.4 else "low")
        return {
            "probability": round(float(prob), 3),
            "risk": risk
        }

    return {
        "model_200": run_model(model_200),
        "model_20k": run_model(model_20k),
        "model_200k": run_model(model_200k),
    }