"""
Extract MediaPipe hand landmarks from ASL dataset images and save as CSV.

For each image:
  - Run MediaPipe Hands to get 21 3D landmarks
  - Normalize relative to wrist (landmark 0) and scale to [-1, 1]
  - Save as a row: label, x0, y0, z0, ..., x20, y20, z20

Images that MediaPipe can't detect a hand in are skipped.
"""

import csv
import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

DATA_DIR = "./asl_data/asl_alphabet_train/asl_alphabet_train"
OUTPUT_CSV = "./landmarks.csv"
MODEL_PATH = "./hand_landmarker.task"

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3,
)
detector = vision.HandLandmarker.create_from_options(options)

columns = ["label"]
for i in range(21):
    columns += [f"x{i}", f"y{i}", f"z{i}"]

classes = sorted(os.listdir(DATA_DIR))
print(f"Found {len(classes)} classes: {classes}\n")

total_written = 0
total_skipped = 0

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(columns)

    for label in classes:
        class_dir = os.path.join(DATA_DIR, label)
        if not os.path.isdir(class_dir):
            continue

        images = os.listdir(class_dir)
        written = 0
        skipped = 0

        for img_name in images:
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                skipped += 1
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            result = detector.detect(mp_image)

            if not result.hand_landmarks:
                skipped += 1
                continue

            lm = result.hand_landmarks[0]

            # Normalize: subtract wrist position so hand location in frame doesn't matter
            wrist_x, wrist_y, wrist_z = lm[0].x, lm[0].y, lm[0].z
            coords = []
            for point in lm:
                coords += [point.x - wrist_x, point.y - wrist_y, point.z - wrist_z]

            # Scale so max absolute value = 1 (makes hand size invariant)
            max_val = max(abs(v) for v in coords) or 1.0
            coords = [v / max_val for v in coords]

            writer.writerow([label] + coords)
            written += 1

        total_written += written
        total_skipped += skipped
        print(f"  {label}: {written} written, {skipped} skipped")

detector.close()
print(f"\nDone. Total written: {total_written}, skipped: {total_skipped}")
print(f"CSV saved to: {OUTPUT_CSV}")
