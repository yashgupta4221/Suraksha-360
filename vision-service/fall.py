import os
from collections import deque

FALL_WINDOW = float(os.getenv("FALL_WINDOW", "1.5"))
FALL_CONFIRM = float(os.getenv("FALL_CONFIRM", "2.0"))
FALL_RECOVER = float(os.getenv("FALL_RECOVER", "3.0"))
FALL_DROP = float(os.getenv("FALL_DROP", "0.25"))


def _is_down(box, raw_pose):
    x1, y1, x2, y2 = box
    w, h = max(x2 - x1, 1), max(y2 - y1, 1)
    return raw_pose == "Lying" or w > 1.2 * h


class FallDetector:
    """normal -> falling -> fallen -> normal, per tracked person."""

    def __init__(self):
        self.state = {}

    def forget(self, tid):
        self.state.pop(tid, None)

    def update(self, tid, now, box, raw_pose):
        s = self.state.setdefault(tid, {
            "status": "normal", "upright": deque(maxlen=60),
            "down_since": None, "up_since": None,
        })
        x1, y1, x2, y2 = box
        cy, h = (y1 + y2) / 2, max(y2 - y1, 1)
        down = _is_down(box, raw_pose)
        new_fall = False

        if not down:
            s["upright"].append((now, cy, h))
            if s["status"] == "falling":
                s["status"], s["down_since"] = "normal", None
            elif s["status"] == "fallen":
                s["up_since"] = s["up_since"] or now
                if now - s["up_since"] >= FALL_RECOVER:
                    s["status"], s["down_since"], s["up_since"] = "normal", None, None
            return s["status"], new_fall

        s["up_since"] = None
        if s["status"] == "normal":
            recent = [u for u in s["upright"] if now - u[0] <= FALL_WINDOW]
            if recent:
                _, cy_up, h_up = recent[0]
                if cy - cy_up > FALL_DROP * h_up or h < 0.65 * h_up:
                    s["status"], s["down_since"] = "falling", now
        elif s["status"] == "falling" and now - s["down_since"] >= FALL_CONFIRM:
            s["status"] = "fallen"
            new_fall = True
        return s["status"], new_fall
