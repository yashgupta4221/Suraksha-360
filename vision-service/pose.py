import math

import cv2

# COCO keypoint indices
NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 0, 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 11, 12, 13, 14, 15, 16

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

KP_MIN_CONF = 0.5


def _visible(kp_conf, i):
    return i < len(kp_conf) and kp_conf[i] >= KP_MIN_CONF


def _mid(kps, kp_conf, a, b):
    if _visible(kp_conf, a) and _visible(kp_conf, b):
        return ((kps[a][0] + kps[b][0]) / 2, (kps[a][1] + kps[b][1]) / 2)
    if _visible(kp_conf, a):
        return tuple(kps[a])
    if _visible(kp_conf, b):
        return tuple(kps[b])
    return None


def classify_pose(box, kps, kp_conf):
    """Rule-based posture from keypoints. Returns one of:
    Lying, Hands Up, Hand Raised, Bending, Sitting, Standing, Unknown."""
    if not kps or not kp_conf:
        return "Unknown"

    x1, y1, x2, y2 = box
    w, h = max(x2 - x1, 1), max(y2 - y1, 1)

    shoulders = _mid(kps, kp_conf, L_SH, R_SH)
    hips = _mid(kps, kp_conf, L_HIP, R_HIP)

    torso_angle = None
    torso_len = None
    if shoulders and hips:
        dx, dy = hips[0] - shoulders[0], hips[1] - shoulders[1]
        torso_len = math.hypot(dx, dy)
        torso_angle = math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))

    if (torso_angle is not None and torso_angle > 60) or (w > 1.4 * h and torso_angle is None):
        return "Lying"

    head_y = None
    if _visible(kp_conf, NOSE):
        head_y = kps[NOSE][1]
    elif shoulders:
        head_y = shoulders[1] - 0.25 * h
    if head_y is not None:
        l_up = _visible(kp_conf, L_WR) and kps[L_WR][1] < head_y
        r_up = _visible(kp_conf, R_WR) and kps[R_WR][1] < head_y
        if l_up and r_up:
            return "Hands Up"
        if l_up or r_up:
            return "Hand Raised"

    if torso_angle is not None and torso_angle > 30:
        return "Bending"

    knees = _mid(kps, kp_conf, L_KNEE, R_KNEE)
    if hips and knees and torso_len:
        if abs(knees[1] - hips[1]) < 0.45 * torso_len:
            return "Sitting"

    if shoulders and hips:
        return "Standing"
    return "Unknown"


def draw_skeleton(frame, kps, kp_conf, color):
    if not kps or not kp_conf:
        return
    for a, b in SKELETON:
        if _visible(kp_conf, a) and _visible(kp_conf, b):
            pa = (int(kps[a][0]), int(kps[a][1]))
            pb = (int(kps[b][0]), int(kps[b][1]))
            cv2.line(frame, pa, pb, color, 2)
    for i, (x, y) in enumerate(kps):
        if _visible(kp_conf, i):
            cv2.circle(frame, (int(x), int(y)), 3, (255, 255, 255), -1)
