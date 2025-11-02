import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
from tracker import Tracker

# Load YOLO model
model = YOLO("yolov8s.pt")

# Load COCO classes
with open("coco.txt", "r") as f:
    class_list = [c.strip() for c in f.read().splitlines() if c.strip()]

# Initialize tracker
tracker = Tracker(max_disappeared=30, max_distance=80)

# Video path
video_path = r"C:\Users\Guru\Music\shoppingmallexitentercounting-main\uhd_30fps.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise SystemExit(f"Cannot open video: {video_path}")

# ✅ One common rectangle zone (Green)
area_zone = [(290,410), (270,418), (443,498), (474,490)]  # Green rectangle

# Counters
counter_in = set()
counter_out = set()
person_status = {}  # Stores each person's state

def point_in_poly(point, poly):
    """Return True if point lies inside polygon."""
    return cv2.pointPolygonTest(np.array(poly, np.int32), point, False) >= 0

frame_count = 0
skip_frames = 2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % (skip_frames + 1) != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))

    # YOLOv8 inference
    results = model(frame)[0]
    detections = []
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if class_list[cls] == "person" and conf > 0.45:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append([x1, y1, x2, y2])

    # Track detections
    tracked = tracker.update(detections)

    # Draw single green rectangle zone
    cv2.polylines(frame, [np.array(area_zone, np.int32)], True, (0, 255, 0), 2)

    # Loop through tracked persons
    for (x1, y1, x2, y2, obj_id) in tracked:
        if obj_id == -1:
            continue

        # Dots
        front_dot = (x2, y2)  # 🔴 Red
        back_dot = (x1, y2)   # 🔵 Blue

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(frame, "Person", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Dots on corners
        cv2.circle(frame, front_dot, 5, (0, 0, 255), -1)
        cv2.circle(frame, back_dot, 5, (255, 0, 0), -1)

        # ✅ Counting logic
        if obj_id not in person_status:
            if point_in_poly(front_dot, area_zone):
                counter_in.add(obj_id)
                person_status[obj_id] = "entered"
            elif point_in_poly(back_dot, area_zone):
                counter_out.add(obj_id)
                person_status[obj_id] = "exited"
        else:
            continue

    # Display counters
    cvzone.putTextRect(frame, f'Entered: {len(counter_in)}', (20, 40),
                       scale=1.2, thickness=2, colorR=(0, 255, 0))
    cvzone.putTextRect(frame, f'Exited: {len(counter_out)}', (20, 90),
                       scale=1.2, thickness=2, colorR=(0, 0, 255))

    cv2.imshow("Output", frame)
    if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
        break

cap.release()
cv2.destroyAllWindows()
