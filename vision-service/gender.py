import os
import threading

import cv2
from PIL import Image
from transformers import pipeline

MODEL_ID = os.getenv("GENDER_MODEL", "rizvandwiki/gender-classification")
KP_MIN_CONF = 0.5
MIN_CROP = int(os.getenv("GENDER_MIN_CROP", "40"))


def head_crop(frame, box, kps, kp_conf):
    """Crop the head using face keypoints (nose, eyes, ears); fall back to
    the top 40% of the person box. Returns None if the crop is too small."""
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    pts = []
    if kps and kp_conf:
        pts = [kps[i] for i in range(5) if i < len(kp_conf) and kp_conf[i] >= KP_MIN_CONF]

    if len(pts) >= 2:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        size = max(max(xs) - min(xs), max(ys) - min(ys), 1) * 2.2
        size = max(size, (x2 - x1) * 0.45)
        cx1, cy1 = int(cx - size / 2), int(cy - size * 0.6)
        cx2, cy2 = int(cx + size / 2), int(cy + size * 0.6)
    else:
        cx1, cy1, cx2, cy2 = x1, y1, x2, y1 + int((y2 - y1) * 0.4)

    cx1, cy1 = max(cx1, 0), max(cy1, 0)
    cx2, cy2 = min(cx2, w), min(cy2, h)
    if cx2 - cx1 < MIN_CROP or cy2 - cy1 < MIN_CROP:
        return None
    return frame[cy1:cy2, cx1:cx2].copy()


class GenderClassifier:
    """Estimates gender presentation from a head crop. Returns the model's
    probability for 'female' (0..1); the caller decides when it is
    confident enough to show a label."""

    def __init__(self):
        device = os.getenv("GENDER_DEVICE", "cpu")
        self.pipe = pipeline(
            "image-classification", model=MODEL_ID,
            device=-1 if device == "cpu" else device,
        )
        self.lock = threading.Lock()

    def p_female(self, crop_bgr):
        img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        with self.lock:
            out = self.pipe(img, top_k=2)
        scores = {o["label"].lower(): float(o["score"]) for o in out}
        if "female" in scores:
            return scores["female"]
        if "male" in scores:
            return 1.0 - scores["male"]
        return None
