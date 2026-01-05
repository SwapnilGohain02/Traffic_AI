# app.py
from inference import analyze_video
from ml_model import ml_predict
from config import ROAD_TYPES

video_path = input("Enter video path: ")
road_type = input("Enter road type (City Street / Highway / Residential): ")

if road_type not in ROAD_TYPES:
    raise ValueError(f"Invalid road type. Choose from {ROAD_TYPES}")

print("\nAnalyzing video...")

counts = analyze_video(video_path)
ml_suggestion = ml_predict(counts, road_type)

print("\nVehicle Counts:", counts)
print("ML-Based Signal Suggestion:", ml_suggestion)
