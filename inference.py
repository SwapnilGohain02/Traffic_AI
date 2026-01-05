# inference.py
import cv2
from ultralytics import YOLO

# Load YOLO model
yolo = YOLO("models/yolov8n.pt")

def analyze_video(video_path, max_frames=60, frame_interval=30):
    cap = cv2.VideoCapture(video_path)

    counts = {
        "car": 0,
        "motorcycle": 0,
        "bus": 0,
        "truck": 0
    }

    frame_id = 0
    used_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_interval == 0:
            results = yolo(frame, conf=0.4, verbose=False)
            for r in results:
                for cls in r.boxes.cls:
                    label = yolo.names[int(cls)]
                    if label in counts:
                        counts[label] += 1

            used_frames += 1
            if used_frames >= max_frames:
                break

        frame_id += 1

    cap.release()
    return counts
