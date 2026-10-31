import os
import threading
import time

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from video_stream import VideoStream

ROTATION = int(os.getenv("ROTATION", "0"))
ROTATE_CODES = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "640"))
DETECT = os.getenv("DETECT", "1") == "1"

detector = None
if DETECT:
    from detector import PersonDetector
    detector = PersonDetector()

app = FastAPI(title="Suraksha360 Vision Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["GET"],
    allow_headers=["*"],
)


def prepare(frame):
    code = ROTATE_CODES.get(ROTATION)
    if code is not None:
        frame = cv2.rotate(frame, code)
    if frame.shape[1] > FRAME_WIDTH:
        h = int(frame.shape[0] * FRAME_WIDTH / frame.shape[1])
        frame = cv2.resize(frame, (FRAME_WIDTH, h))
    return frame


def draw(frame, detections):
    green = (80, 220, 100)
    for i, d in enumerate(detections, start=1):
        x1, y1, x2, y2 = d["box"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), green, 2)
        label = f"Person {i}  {d['conf']:.2f}"
        cv2.putText(frame, label, (x1, max(y1 - 8, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, green, 2)
    cv2.rectangle(frame, (0, 0), (150, 30), (0, 0, 0), -1)
    cv2.putText(frame, f"Persons: {len(detections)}", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


class CameraWorker:
    """Reads the newest camera frame, runs detection once, and keeps the
    annotated JPEG + detections ready for every viewer."""

    def __init__(self, cam_id, source):
        self.id = cam_id
        self.stream = VideoStream(source)
        self.lock = threading.Lock()
        self.jpeg = None
        self.detections = []
        self.fps = 0.0
        self.updated = 0.0
        threading.Thread(target=self._loop, daemon=True).start()

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
            frame = prepare(frame)
            detections = detector.detect(frame) if detector else []
            frame = draw(frame, detections)
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
            return self.jpeg, list(self.detections), self.fps, self.updated


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
    return {"ok": True, "cameras": len(workers), "detection": detector is not None}


@app.get("/cameras")
def list_cameras():
    result = []
    for cid, w in workers.items():
        _, dets, fps, _ = w.snapshot()
        result.append({"id": cid, "online": w.online(), "persons": len(dets), "fps": round(fps, 1)})
    return result


@app.get("/detections/{cam_id}")
def detections(cam_id: str):
    w = get_worker(cam_id)
    _, dets, fps, updated = w.snapshot()
    return {"id": cam_id, "persons": len(dets), "fps": round(fps, 1), "updated": updated, "detections": dets}


def mjpeg(worker, max_fps=15):
    last = 0.0
    while True:
        jpg, _, _, updated = worker.snapshot()
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
