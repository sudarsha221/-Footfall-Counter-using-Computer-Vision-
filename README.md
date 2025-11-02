# -Footfall-Counter-using-Computer-Vision-
# Project Overview
This project detects, tracks, and counts how many people enter or exit through a single green zone in a video.
It uses YOLOv8 trained on the COCO dataset to detect people and applies simple tracking logic to decide whether a person is entering or exiting based on their movement.
# Key Features
Person Detection: Uses YOLOv8 model pretrained on the COCO dataset to detect humans.

Tracking: Keeps track of each person across frames using unique IDs.

Single Green Rectangle: Acts as both entry and exit zone.

Red Dot (Front): When it reaches the green rectangle → Entry Count +1.

Blue Dot (Back): When it reaches the green rectangle → Exit Count +1.

Duplicate Prevention: Each person is counted only once (either entry or exit).

Live Visualization: Shows bounding boxes, direction dots, and counts on the video in real time.

# Install dependencies from your requirements file
pip install -r requirements.txt
# Run your script
python main.py
