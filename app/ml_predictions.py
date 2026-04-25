import os
import pandas as pd
import joblib


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "ml", "models")

print("MODEL DIR:", MODEL_DIR)
print("FILES:", os.listdir(MODEL_DIR))

model_200 = joblib.load(os.path.join(MODEL_DIR, "model_200.pkl"))
model_20k = joblib.load(os.path.join(MODEL_DIR, "model_20k.pkl"))
model_200k = joblib.load(os.path.join(MODEL_DIR, "model_200k.pkl"))
model4_overall = joblib.load(os.path.join(MODEL_DIR, "abgs_model4_overall.pkl"))
model4_activity = joblib.load(os.path.join(MODEL_DIR, "abgs_model4_activity.pkl"))


MONOLITHIC_FEATURES = [
    "streak_length",
    "missed_days_last_7",
    "checkin_hour_avg",
    "engagement_freq",
    "days_since_last",
    "goal_age_days"
]


ABGS_FEATURES = [
    "habit_type",
    "activity_type",
    "difficulty_level",
    "streak_length",
    "missed_days_last_7",
    "checkin_hour_avg",
    "engagement_freq",
    "days_since_last",
    "goal_age_days",
    "pomodoro_sessions",
    "pomodoro_completion_rate"
]


def risk_label(prediction):
    return "High Risk" if int(prediction) == 1 else "Low Risk"


def probability_score(model, input_df):
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        return round(float(max(probabilities)) * 100, 2)

    return None


def predict_all_models(payload):
    monolithic_df = pd.DataFrame([{
        feature: payload[feature]
        for feature in MONOLITHIC_FEATURES
    }])

    abgs_df = pd.DataFrame([{
        feature: payload[feature]
        for feature in ABGS_FEATURES
    }])

    pred_200 = model_200.predict(monolithic_df)[0]
    pred_20k = model_20k.predict(monolithic_df)[0]
    pred_200k = model_200k.predict(monolithic_df)[0]

    pred4_overall = model4_overall.predict(abgs_df)[0]
    pred4_activity = model4_activity.predict(abgs_df)[0]

    return {
        "client": "Client A",
        "input_summary": payload,
        "models": [
            {
                "model": "Model 1",
                "dataset": "200 records",
                "type": "Monolithic",
                "prediction": risk_label(pred_200),
                "confidence": probability_score(model_200, monolithic_df),
                "description": "Baseline model trained on the smallest dataset."
            },
            {
                "model": "Model 2",
                "dataset": "20,000 records",
                "type": "Monolithic",
                "prediction": risk_label(pred_20k),
                "confidence": probability_score(model_20k, monolithic_df),
                "description": "Medium synthetic dataset used to test scaling effects."
            },
            {
                "model": "Model 3",
                "dataset": "200,000 records",
                "type": "Monolithic",
                "prediction": risk_label(pred_200k),
                "confidence": probability_score(model_200k, monolithic_df),
                "description": "Large synthetic dataset used to test whether performance plateaus."
            },
            {
                "model": "Model 4",
                "dataset": "ABGS structured dataset",
                "type": "Habit-aware",
                "prediction": risk_label(pred4_overall),
                "activity_prediction": risk_label(pred4_activity),
                "confidence": probability_score(model4_overall, abgs_df),
                "activity_confidence": probability_score(model4_activity, abgs_df),
                "description": "Structured model using habit type, activity type, difficulty, and Pomodoro behavior."
            }
        ]
    }