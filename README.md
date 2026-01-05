# Traffic Signal AI – Prototype

## Overview
This repository contains a working prototype of an AI-based traffic signal
recommendation system. The prototype analyzes traffic videos and provides
signal timing suggestions using computer vision and machine learning.

The objective of this project is to demonstrate feasibility and system design,
not production-scale deployment.

---

## What the System Does
- Accepts a traffic video as input
- Detects and counts vehicles using a computer vision model
- Extracts traffic composition features
- Uses a supervised machine learning model to generate signal timing suggestions
- Displays results through a simple dashboard interface

---

## Inputs
- Traffic video (roadside camera footage)
- Road type (City Street, Highway, Residential)

---

## Outputs
- Vehicle counts (cars, motorcycles, buses, trucks)
- Traffic signal timing recommendation

---

## System Design
- Vehicle detection is performed using a YOLO-based model
- Signal timing decisions are generated using a trained ML classifier
- Hard-coded rule-based logic is intentionally avoided in the final prototype
- Vehicle speed is excluded, as it cannot be reliably inferred from a single
  uncalibrated camera

---

## Tech Stack
- Python
- YOLOv8 (vehicle detection)
- XGBoost (supervised ML model)
- Streamlit (dashboard UI)
- OpenCV (video processing)

---

## How to Run Locally

python -m streamlit run ui_app.py
pip install -r requirements.txt
