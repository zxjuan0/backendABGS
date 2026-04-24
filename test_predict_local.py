from ml.predict import predict_all

sample = {
    "streak_length": 7,
    "missed_days_last_7": 2,
    "checkin_hour_avg": 18,
    "engagement_freq": 0.65,
    "days_since_last": 1,
    "goal_age_days": 30,
}

result = predict_all(sample)
print(result)