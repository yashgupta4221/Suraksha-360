import os
import queue
import threading
import time
from collections import Counter, deque

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from pose import classify_pose, draw_skeleton
from fall import FallDetector
from fight import FightDetector
from video_stream import VideoStream

ROTATION = int(os.getenv("ROTATION", "0"))
ROTATE_CODES = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "640"))
DETECT = os.getenv("DETECT", "1") == "1"
TRACK = os.getenv("TRACK", "1") == "1"
POSE = os.getenv("POSE", "1") == "1"
GENDER = os.getenv("GENDER", "1") == "1"
TRAIL_LEN = int(os.getenv("TRAIL_LEN", "30"))
TRACK_TTL = float(os.getenv("TRACK_TTL", "3"))
MIN_SEEN_SECONDS = float(os.getenv("MIN_SEEN_SECONDS", "1.5"))
POSE_SMOOTHING = int(os.getenv("POSE_SMOOTHING", "7"))
FALL = os.getenv("FALL", "1") == "1"
FIGHT = os.getenv("FIGHT", "1") == "1"
GENDER_EVERY = int(os.getenv("GENDER_EVERY", "10"))
GENDER_MAX_VOTES = int(os.getenv("GENDER_MAX_VOTES", "15"))
GENDER_MIN_VOTES = int(os.getenv("GENDER_MIN_VOTES", "3"))
GENDER_MIN_CONF = float(os.getenv("GENDER_MIN_CONF", "0.75"))

if DETECT:
    from detector import PersonDetector

gender_model = None
head_crop = None
if DETECT and GENDER:
    from gender import GenderClassifier, head_crop
    gender_model = GenderClassifier()

app = FastAPI(title="Suraksha360 Vision Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["GET"],
    allow_headers=["*"],
)

PALETTE = [
    (80, 220, 100), (255, 170, 60), (60, 170, 255), (220, 90, 220),
    (90, 230, 230), (240, 240, 90), (150, 120, 255), (255, 120, 120),
]
ALERT_POSES = {"Hands Up", "Lying"}


def color_for(track_id):
    return PALETTE[track_id % len(PALETTE)] if track_id is not None else PALETTE[0]


def gender_summary(probs):
    if len(probs) < GENDER_MIN_VOTES:
        return "uncertain", None
    p = sum(probs) / len(probs)
    if p >= GENDER_MIN_CONF:
        return "female", p
    if 1 - p >= GENDER_MIN_CONF:
        return "male", 1 - p
    return "uncertain", max(p, 1 - p)


def gender_tag(d):
    g, conf = d.get("gender"), d.get("gender_conf")
    if g == "female":
        return f"F {int(conf * 100)}%"
    if g == "male":
        return f"M {int(conf * 100)}%"
    return "?"


def prepare(frame):
    code = ROTATE_CODES.get(ROTATION)
    if code is not None:
        frame = cv2.rotate(frame, code)
    if frame.shape[1] > FRAME_WIDTH:
        h = int(frame.shape[0] * FRAME_WIDTH / frame.shape[1])
        frame = cv2.resize(frame, (FRAME_WIDTH, h))
    return frame


def draw(frame, detections, trails, total_seen):
    for i, d in enumerate(detections, start=1):
        x1, y1, x2, y2 = d["box"]
        tid = d.get("id")
        color = color_for(tid)
        points = trails.get(tid, [])
        for a, b in zip(points, points[1:]):
            cv2.line(frame, a, b, color, 2)
        if POSE:
            draw_skeleton(frame, d.get("keypoints"), d.get("kp_conf"), color)
        fallen = d.get("fall") == "fallen"
        fighting = bool(d.get("fight"))
        box_color = (60, 60, 255) if fallen or fighting or d.get("pose") in ALERT_POSES else color
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        label = f"ID {tid}" if tid is not None else f"Person {i}"
        if d.get("pose"):
            label += f" | {d['pose']}"
        if gender_model is not None and tid is not None:
            label += f" | {gender_tag(d)}"
        if fallen:
            label = "FALL! " + label
        if fighting:
            label = "FIGHT! " + label
        cv2.putText(frame, label, (x1, max(y1 - 8, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
    if any(d.get("fall") == "fallen" for d in detections):
        cv2.rectangle(frame, (0, 32), (frame.shape[1], 70), (0, 0, 200), -1)
        cv2.putText(frame, "FALL DETECTED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    if any(d.get("fight") for d in detections):
        cv2.rectangle(frame, (0, 72), (frame.shape[1], 110), (0, 0, 200), -1)
        cv2.putText(frame, "FIGHT DETECTED", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    header = f"Persons: {len(detections)}   Seen: {total_seen}"
    cv2.rectangle(frame, (0, 0), (260, 30), (0, 0, 0), -1)
    cv2.putText(frame, header, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


class CameraWorker:
    def __init__(self, cam_id, source):
        self.id = cam_id
        self.stream = VideoStream(source)
        self.detector = PersonDetector() if DETECT else None
        self.lock = threading.Lock()
        self.jpeg = None
        self.detections = []
        self.fps = 0.0
        self.updated = 0.0
        self.tracks = {}
        self.seen_ids = set()
        self.frame_no = 0
        self.gender_queue = queue.Queue(maxsize=8)
        self.gender_probs = {}
        self.gender_last = {}
        self.glock = threading.Lock()
        self.fall = FallDetector() if FALL else None
        self.fight = FightDetector() if FIGHT else None
        self.events = deque(maxlen=50)
        threading.Thread(target=self._loop, daemon=True).start()
        if gender_model is not None:
            threading.Thread(target=self._gender_loop, daemon=True).start()

    def _gender_loop(self):
        while True:
            tid, crop = self.gender_queue.get()
            try:
                p = gender_model.p_female(crop)
            except Exception as exc:
                print(f"[gender] {exc}")
                continue
            if p is None:
                continue
            with self.glock:
                self.gender_probs.setdefault(tid, deque(maxlen=GENDER_MAX_VOTES)).append(p)

    def _update_gender(self, frame, d, tid):
        with self.glock:
            probs = list(self.gender_probs.get(tid, []))
        due = self.frame_no - self.gender_last.get(tid, -10 ** 9) >= GENDER_EVERY
        if len(probs) < GENDER_MAX_VOTES and due:
            crop = head_crop(frame, d["box"], d.get("keypoints"), d.get("kp_conf"))
            if crop is not None:
                try:
                    self.gender_queue.put_nowait((tid, crop))
                    self.gender_last[tid] = self.frame_no
                except queue.Full:
                    pass
        g, conf = gender_summary(probs)
        d["gender"] = g
        d["gender_conf"] = round(conf, 2) if conf is not None else None
        d["gender_looks"] = len(probs)

    def _update_tracks(self, frame, detections, now):
        for d in detections:
            raw_pose = classify_pose(d["box"], d.get("keypoints"), d.get("kp_conf")) if POSE else None
            tid = d.get("id")
            if tid is None:
                d["pose"] = raw_pose
                continue
            x1, y1, x2, y2 = d["box"]
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            track = self.tracks.get(tid)
            if track is None:
                track = {
                    "first_seen": now,
                    "trail": deque(maxlen=TRAIL_LEN),
                    "poses": deque(maxlen=POSE_SMOOTHING),
                }
                self.tracks[tid] = track
            track["last_seen"] = now
            track["trail"].append(center)
            if raw_pose and raw_pose != "Unknown":
                track["poses"].append(raw_pose)
            d["pose"] = Counter(track["poses"]).most_common(1)[0][0] if track["poses"] else raw_pose
            d["duration"] = round(now - track["first_seen"], 1)
            if d["duration"] >= MIN_SEEN_SECONDS:
                self.seen_ids.add(tid)
            if gender_model is not None:
                self._update_gender(frame, d, tid)
            if self.fall is not None:
                fall_state, new_fall = self.fall.update(tid, now, d["box"], raw_pose)
                d["fall"] = fall_state
                if new_fall:
                    self.events.appendleft({"type": "fall", "id": tid, "time": now})
                    print(f"[fall] FALL DETECTED for ID {tid}")
        if self.fight is not None:
            fighting, new_fights = self.fight.update(now, detections)
            for d in detections:
                d["fight"] = d.get("id") in fighting
            for pair in new_fights:
                self.events.appendleft({"type": "fight", "ids": list(pair), "time": now})
                print(f"[fight] FIGHT DETECTED between IDs {pair[0]} and {pair[1]}")
        for tid in [k for k, t in self.tracks.items() if now - t["last_seen"] > TRACK_TTL]:
            del self.tracks[tid]
            if self.fall is not None:
                self.fall.forget(tid)
            if self.fight is not None:
                self.fight.forget(tid)
            self.gender_last.pop(tid, None)
            with self.glock:
                self.gender_probs.pop(tid, None)
        return {tid: list(t["trail"]) for tid, t in self.tracks.items()}

    def _loop(self):
        last_frame_time = 0.0
        while True:
            if self.stream.last_seen == last_frame_time:
                time.sleep(0.01)
                continue
            last_frame_time = self.stream.last_seen
            ok, frame = self.stream.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            start = time.time()
            self.frame_no += 1
            frame = prepare(frame)
            detections = self.detector.detect(frame, track=TRACK) if self.detector else []
            trails = self._update_tracks(frame, detections, time.time())
            frame = draw(frame, detections, trails, len(self.seen_ids))
            _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            elapsed = time.time() - start
            with self.lock:
                self.jpeg = jpg.tobytes()
                self.detections = detections
                self.updated = time.time()
                if elapsed > 0:
                    self.fps = 1 / elapsed if self.fps == 0 else 0.9 * self.fps + 0.1 / elapsed

    def online(self):
        return self.stream.status and (time.time() - self.stream.last_seen) < 5

    def snapshot(self):
        with self.lock:
            return self.jpeg, list(self.detections), self.fps, self.updated, len(self.seen_ids)


workers = {
    name.strip(): CameraWorker(name.strip(), src.strip())
    for name, src in (
        c.split("=", 1) for c in os.getenv("CAMERAS", "").split(",") if "=" in c
    )
}


def get_worker(cam_id):
    if cam_id not in workers:
        raise HTTPException(status_code=404, detail="Unknown camera")
    return workers[cam_id]


@app.get("/health")
def health():
    return {
        "ok": True, "cameras": len(workers), "detection": DETECT,
        "tracking": DETECT and TRACK, "pose": DETECT and POSE,
        "gender": gender_model is not None,
    }


@app.get("/cameras")
def list_cameras():
    result = []
    for cid, w in workers.items():
        _, dets, fps, _, seen = w.snapshot()
        poses = Counter(d.get("pose") for d in dets if d.get("pose"))
        genders = Counter(d.get("gender") for d in dets if d.get("gender"))
        result.append({
            "id": cid, "online": w.online(), "persons": len(dets), "seen": seen,
            "fps": round(fps, 1), "poses": dict(poses),
            "genders": {k: genders.get(k, 0) for k in ("female", "male", "uncertain")}
            if gender_model is not None else None,
            "alerts": sorted(
                {p for p in poses if p in ALERT_POSES}
                | ({"Fall Detected"} if any(d.get("fall") == "fallen" for d in dets) else set())
                | ({"Fight Detected"} if any(d.get("fight") for d in dets) else set())
            ),
        })
    return result


@app.get("/detections/{cam_id}")
def detections(cam_id: str):
    w = get_worker(cam_id)
    _, dets, fps, updated, seen = w.snapshot()
    return {
        "id": cam_id, "persons": len(dets), "seen": seen, "fps": round(fps, 1),
        "updated": updated, "detections": dets,
    }


def mjpeg(worker, max_fps=15):
    last = 0.0
    while True:
        jpg, _, _, updated, _ = worker.snapshot()
        if jpg is not None and updated != last:
            last = updated
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        time.sleep(1 / max_fps)


@app.get("/stream/{cam_id}")
def stream(cam_id: str):
    return StreamingResponse(
        mjpeg(get_worker(cam_id)),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/events")
def events():
    out = []
    for cid, w in workers.items():
        out.extend({"camera": cid, **e} for e in list(w.events))
    return sorted(out, key=lambda e: e["time"], reverse=True)[:50]
