# ml/train.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import joblib, os

df = pd.read_csv("data/habit_data.csv")

FEATURES = ["streak_length", "missed_days_last_7",
            "checkin_hour_avg", "engagement_freq",
            "days_since_last", "goal_age_days"]
TARGET = "label"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_s, y_train)

print(classification_report(y_test, model.predict(X_test_s)))

os.makedirs("ml/models", exist_ok=True)
joblib.dump(model, "ml/models/dropout_model.joblib")
joblib.dump(scaler, "ml/models/scaler.joblib")
print("Models saved to ml/models/")
