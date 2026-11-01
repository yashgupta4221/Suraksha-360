import os
from collections import deque

FIGHT_DIST = float(os.getenv("FIGHT_DIST", "1.0"))
FIGHT_SPEED = float(os.getenv("FIGHT_SPEED", "0.8"))
FIGHT_WINDOW = float(os.getenv("FIGHT_WINDOW", "2.0"))
FIGHT_MIN_HITS = int(os.getenv("FIGHT_MIN_HITS", "3"))
FIGHT_CLEAR = float(os.getenv("FIGHT_CLEAR", "3.0"))
KP_MIN_CONF = 0.5
WRISTS = (9, 10)


def _wrists(d):
    kps, conf = d.get("keypoints") or [], d.get("kp_conf") or []
    return [tuple(kps[i]) for i in WRISTS if i < len(conf) and conf[i] >= KP_MIN_CONF]


def _inside(pt, box, pad):
    x1, y1, x2, y2 = box
    return x1 - pad <= pt[0] <= x2 + pad and y1 - pad <= pt[1] <= y2 + pad


class FightDetector:
    """Two tracked people are flagged as fighting when, repeatedly within
    FIGHT_WINDOW seconds, they are close, one moves their hands fast, and a
    hand reaches into the other person's box."""

    def __init__(self):
        self.last = {}
        self.speed = {}
        self.pairs = {}

    def forget(self, tid):
        self.last.pop(tid, None)
        self.speed.pop(tid, None)
        for key in [k for k in self.pairs if tid in k]:
            del self.pairs[key]

    def _update_speed(self, tid, now, d):
        x1, y1, x2, y2 = d["box"]
        h = max(y2 - y1, 1)
        wr = _wrists(d)
        prev = self.last.get(tid)
        self.last[tid] = (now, wr)
        if not prev or not wr or not prev[1] or now - prev[0] <= 0:
            self.speed[tid] = self.speed.get(tid, 0.0) * 0.7
            return
        dt = now - prev[0]
        moves = [((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 for a in wr for b in prev[1]]
        inst = min(moves) / dt / h if moves else 0.0
        self.speed[tid] = 0.5 * self.speed.get(tid, 0.0) + 0.5 * inst

    def update(self, now, detections):
        tracked = [d for d in detections if d.get("id") is not None]
        for d in tracked:
            self._update_speed(d["id"], now, d)

        for i, a in enumerate(tracked):
            for b in tracked[i + 1:]:
                ax1, ay1, ax2, ay2 = a["box"]
                bx1, by1, bx2, by2 = b["box"]
                ha, hb = max(ay2 - ay1, 1), max(by2 - by1, 1)
                hm = (ha + hb) / 2
                dist = (((ax1 + ax2) - (bx1 + bx2)) ** 2 + ((ay1 + ay2) - (by1 + by2)) ** 2) ** 0.5 / 2
                if dist > FIGHT_DIST * hm:
                    continue
                fast = max(self.speed.get(a["id"], 0), self.speed.get(b["id"], 0)) >= FIGHT_SPEED
                reach = any(_inside(w, b["box"], 0.05 * hb) for w in _wrists(a)) or \
                    any(_inside(w, a["box"], 0.05 * ha) for w in _wrists(b))
                if fast and reach:
                    key = tuple(sorted((a["id"], b["id"])))
                    p = self.pairs.setdefault(key, {"hits": deque(), "active": False, "last_hit": now})
                    p["hits"].append(now)
                    p["last_hit"] = now

        fighting, new_events = set(), []
        for key, p in list(self.pairs.items()):
            while p["hits"] and now - p["hits"][0] > FIGHT_WINDOW:
                p["hits"].popleft()
            if not p["active"] and len(p["hits"]) >= FIGHT_MIN_HITS:
                p["active"] = True
                new_events.append(key)
            if p["active"] and now - p["last_hit"] > FIGHT_CLEAR:
                p["active"] = False
            if p["active"]:
                fighting.update(key)
            elif not p["hits"]:
                del self.pairs[key]
        return fighting, new_events
