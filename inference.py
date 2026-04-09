"""
Real-time ASL fingerspelling inference with caption display.

Controls:
  Backspace  — delete last character
  C          — clear all text
  Q / Esc    — quit

UX logic:
  - Prediction is smoothed over a rolling window of frames
  - A letter is committed only after being the stable prediction for CONFIRM_FRAMES
  - After committing, a cooldown prevents the same letter re-committing immediately
    (you must briefly show "nothing" or a different letter to repeat)
  - Holding the "space" sign for CONFIRM_FRAMES inserts a space
  - "nothing" (no recognizable sign) does nothing
"""

import pickle
import collections
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- Tunable UX parameters ---
CONFIRM_FRAMES = 60   # frames a sign must be held before committing — lower = faster, higher = more deliberate
SMOOTH_WINDOW  = 7    # rolling window size for prediction smoothing

MODEL_PATH          = "./model.pkl"
ENCODER_PATH        = "./label_encoder.pkl"
LANDMARKER_PATH     = "./hand_landmarker.task"
CAPTION_HEIGHT      = 60   # pixels
CAPTION_FONT_SCALE  = 1.2
CAPTION_COLOR       = (255, 255, 255)
CAPTION_BG          = (30, 30, 30)


def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_encoder(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def init_landmarker(path):
    base_options = python.BaseOptions(model_asset_path=path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def extract_landmarks(result):
    """Return normalized 63-length feature vector, or None if no hand detected."""
    if not result.hand_landmarks:
        return None

    lm = result.hand_landmarks[0]
    wrist_x, wrist_y, wrist_z = lm[0].x, lm[0].y, lm[0].z
    coords = []
    for point in lm:
        coords += [point.x - wrist_x, point.y - wrist_y, point.z - wrist_z]

    max_val = max(abs(v) for v in coords) or 1.0
    return [v / max_val for v in coords]


def draw_caption(frame, text):
    """Draw a caption bar at the bottom of the frame."""
    h, w = frame.shape[:2]
    bar_y = h - CAPTION_HEIGHT
    cv2.rectangle(frame, (0, bar_y), (w, h), CAPTION_BG, -1)

    # Truncate text to fit frame width
    display = text if text else ""
    cv2.putText(
        frame, display,
        (10, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        CAPTION_FONT_SCALE,
        CAPTION_COLOR,
        2,
        cv2.LINE_AA,
    )


def draw_prediction_overlay(frame, prediction, stable_count):
    """Show current (uncommitted) prediction and a fill bar indicating commit progress."""
    if prediction is None or prediction == "nothing":
        return

    progress = min(stable_count / CONFIRM_FRAMES, 1.0)
    bar_w = int(200 * progress)

    cv2.putText(frame, f"[ {prediction} ]", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 120), 2, cv2.LINE_AA)
    cv2.rectangle(frame, (10, 55), (210, 70), (60, 60, 60), -1)
    cv2.rectangle(frame, (10, 55), (10 + bar_w, 70), (0, 255, 120), -1)


def main():
    print("Loading model...")
    model = load_model(MODEL_PATH)
    le = load_encoder(ENCODER_PATH)
    print("Loading hand landmarker...")
    detector = init_landmarker(LANDMARKER_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    caption        = ""
    stable_count   = 0
    last_stable    = None   # current stable prediction being tracked
    last_committed = None   # last sign that was committed
    in_cooldown    = False  # True if same-sign cooldown is active
    cooldown_count = 0
    recent_preds   = collections.deque(maxlen=SMOOTH_WINDOW)

    print("Running. Press Q or Esc to quit, Backspace to delete, C to clear.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror so it feels natural
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = detector.detect(mp_image)

        features = extract_landmarks(result)

        if features is None:
            recent_preds.append("nothing")
        else:
            pred = le.inverse_transform(model.predict([features]))[0]
            recent_preds.append(pred)

        # Smooth: take most common prediction in rolling window
        smoothed = collections.Counter(recent_preds).most_common(1)[0][0]

        # --- Cooldown countdown ---
        if in_cooldown:
            cooldown_count -= 1
            if cooldown_count <= 0:
                in_cooldown = False

        # --- "nothing" resets tracking but does nothing else ---
        if smoothed == "nothing":
            stable_count = 0
            last_stable = None

        # --- All actionable signs (letters + space) ---
        else:
            if smoothed == last_stable:
                stable_count += 1
            else:
                last_stable = smoothed
                stable_count = 1

            if stable_count >= CONFIRM_FRAMES:
                if not (in_cooldown and smoothed == last_committed):
                    if smoothed == "space":
                        caption += " "
                    elif smoothed in ("del", "nothing"):
                        pass  # not real signs — ignore
                    else:
                        caption += smoothed
                    last_committed = smoothed
                    in_cooldown = True
                    cooldown_count = CONFIRM_FRAMES
                    stable_count = 0

        # --- Draw ---
        draw_prediction_overlay(frame, smoothed if smoothed not in ("nothing",) else None, stable_count)
        draw_caption(frame, caption)
        cv2.imshow("ASL Fingerspelling", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):   # Q or Esc
            break
        elif key == 8:              # Backspace
            caption = caption[:-1] if caption else ""
        elif key == ord("c"):
            caption = ""

    cap.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
