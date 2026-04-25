import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# =========================
# PATH SETUP
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "ml", "data")
MODEL_DIR = os.path.join(BASE_DIR, "ml", "models")

os.makedirs(MODEL_DIR, exist_ok=True)

print("BASE DIR:", BASE_DIR)
print("DATA DIR:", DATA_DIR)
print("FILES IN DATA DIR:", os.listdir(DATA_DIR))


# =========================
# FEATURE SETS
# =========================

MONOLITHIC_FEATURES = [
    "streak_length",
    "missed_days_last_7",
    "checkin_hour_avg",
    "engagement_freq",
    "days_since_last",
    "goal_age_days",
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
    "pomodoro_completion_rate",
]


# =========================
# TRAIN MONOLITHIC MODELS
# =========================

def train_monolithic_model(dataset_filename, model_filename, model_name):
    print(f"\n========== Training {model_name} ==========")

    path = os.path.join(DATA_DIR, dataset_filename)
    print("Loading:", path)

    df = pd.read_csv(path)

    X = df[MONOLITHIC_FEATURES]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)

    print(classification_report(y_test, preds, zero_division=0))

    results = {
        "model": model_name,
        "dataset": dataset_filename,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }

    print(results)

    output_path = os.path.join(MODEL_DIR, model_filename)
    joblib.dump(model, output_path)

    print("Saved model:", output_path)

    return results


# =========================
# TRAIN ABGS MODEL 4
# =========================

def train_abgs_model():
    print("\n========== Training Model 4: ABGS Structured Model ==========")

    path = os.path.join(DATA_DIR, "abgs_model4_client_dataset.csv")
    print("Loading:", path)

    df = pd.read_csv(path)

    X = df[ABGS_FEATURES]

    categorical_features = [
        "habit_type",
        "activity_type",
        "difficulty_level",
    ]

    numeric_features = [
        "streak_length",
        "missed_days_last_7",
        "checkin_hour_avg",
        "engagement_freq",
        "days_since_last",
        "goal_age_days",
        "pomodoro_sessions",
        "pomodoro_completion_rate",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("numeric", "passthrough", numeric_features),
        ]
    )

    def train_target(target_column, model_filename, model_name):
        print(f"\n--- Training {model_name} ---")

        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=150,
                        random_state=42,
                        class_weight="balanced",
                    ),
                ),
            ]
        )

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        accuracy = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, zero_division=0)
        recall = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)

        print(classification_report(y_test, preds, zero_division=0))

        results = {
            "model": model_name,
            "dataset": "abgs_model4_client_dataset.csv",
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

        print(results)

        output_path = os.path.join(MODEL_DIR, model_filename)
        joblib.dump(model, output_path)

        print("Saved model:", output_path)

        return results

    overall_results = train_target(
        "overall_risk",
        "abgs_model4_overall.pkl",
        "Model 4: ABGS Overall Risk",
    )

    activity_results = train_target(
        "activity_specific_risk",
        "abgs_model4_activity.pkl",
        "Model 4: ABGS Activity-Specific Risk",
    )

    return overall_results, activity_results


# =========================
# RUN ALL TRAINING
# =========================

if __name__ == "__main__":
    results = []

    results.append(
        train_monolithic_model(
            "habit_data_200.csv",
            "model_200.pkl",
            "Model 1: 200 Records",
        )
    )

    results.append(
        train_monolithic_model(
            "habit_data_20k.csv",
            "model_20k.pkl",
            "Model 2: 20,000 Records",
        )
    )

    results.append(
        train_monolithic_model(
            "habit_data_200k.csv",
            "model_200k.pkl",
            "Model 3: 200,000 Records",
        )
    )

    abgs_overall, abgs_activity = train_abgs_model()

    results.append(abgs_overall)
    results.append(abgs_activity)

    results_df = pd.DataFrame(results)

    results_path = os.path.join(MODEL_DIR, "model_comparison_results.csv")
    results_df.to_csv(results_path, index=False)

    print("\n========== FINAL MODEL COMPARISON ==========")
    print(results_df)
    print("\nSaved comparison results to:", results_path)