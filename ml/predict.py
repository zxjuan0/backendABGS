import joblib
import pandas as pd

FEATURES = [
    "streak_length",
    "missed_days_last_7",
    "checkin_hour_avg",
    "engagement_freq",
    "days_since_last",
    "goal_age_days",
]

model_200 = joblib.load("ml/models/model_200.pkl")
model_20k = joblib.load("ml/models/model_20k.pkl")
model_200k = joblib.load("ml/models/model_200k.pkl")


def get_risk(probability):
    if probability > 0.7:
        return "high"
    if probability > 0.4:
        return "medium"
    return "low"


def get_positive_class_probability(model, x):
    probabilities = model.predict_proba(x)[0]
    classes = list(model.classes_)

    # If the model has class 1, return probability for class 1
    if 1 in classes:
        class_index = classes.index(1)
        return probabilities[class_index]

    # If the model was trained on only class 0, probability of dropout/class 1 is 0
    return 0.0


def run_model(model, user_features):
    x = pd.DataFrame([[user_features[f] for f in FEATURES]], columns=FEATURES)

    probability = get_positive_class_probability(model, x)
    risk = get_risk(probability)

    return {
        "probability": round(float(probability), 3),
        "risk": risk,
        "classes_seen": [int(c) for c in model.classes_],
    }


def predict_all(user_features: dict) -> dict:
    return {
        "model_200": run_model(model_200, user_features),
        "model_20k": run_model(model_20k, user_features),
        "model_200k": run_model(model_200k, user_features),
    }