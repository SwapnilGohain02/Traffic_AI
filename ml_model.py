# ml_model.py
import joblib
import numpy as np
from datetime import datetime

# Load model & encoder
model = joblib.load("models/xgb_generalized_model.pkl")
label_encoder = joblib.load("models/label_encoder_generalized.pkl")

def compute_vehicle_score(counts):
    return (
        1.0 * counts["car"] +
        0.5 * counts["motorcycle"] +
        3.0 * counts["bus"] +
        3.0 * counts["truck"]
    )

def ml_predict(counts, road_type):
    # --- vehicle score (USED IN TRAINING) ---
    vehicle_score = compute_vehicle_score(counts)

    # --- time features ---
    hour = datetime.now().hour
    is_peak = 1 if (8 <= hour <= 10 or 17 <= hour <= 19) else 0

    # --- feature vector MUST match training order ---
    feature_vector = np.array([[
        counts["car"],
        counts["motorcycle"],
        counts["bus"],
        counts["truck"],
        vehicle_score,
        hour,
        is_peak,
        1 if road_type == "City Street" else 0,
        1 if road_type == "Highway" else 0,
        1 if road_type == "Residential" else 0,
        # remaining features here IF your notebook had them
    ]])

    pred = model.predict(feature_vector)
    return label_encoder.inverse_transform(pred)[0]
