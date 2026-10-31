import os
import time

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from video_stream import VideoStream

app = FastAPI(title="Suraksha360 Vision Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["GET"],
    allow_headers=["*"],
)

cameras = {
    name.strip(): VideoStream(src.strip())
    for name, src in (
        c.split("=", 1) for c in os.getenv("CAMERAS", "").split(",") if "=" in c
    )
}


@app.get("/health")
def health():
    return {"ok": True, "cameras": len(cameras)}


@app.get("/cameras")
def list_cameras():
    now = time.time()
    return [
        {"id": cid, "online": cam.status and (now - cam.last_seen) < 5}
        for cid, cam in cameras.items()
    ]


ROTATION = int(os.getenv("ROTATION", "0"))
ROTATE_CODES = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def rotate(frame):
    code = ROTATE_CODES.get(ROTATION)
    return cv2.rotate(frame, code) if code is not None else frame


def mjpeg(cam, fps=12, width=640):
    while True:
        ok, frame = cam.read()
        if ok and frame is not None:
            frame = rotate(frame)
            if frame.shape[1] > width:
                h = int(frame.shape[0] * width / frame.shape[1])
                frame = cv2.resize(frame, (width, h))
            _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + jpg.tobytes() + b"\r\n")
        time.sleep(1 / fps)


@app.get("/stream/{cam_id}")
def stream(cam_id: str):
    if cam_id not in cameras:
        raise HTTPException(status_code=404, detail="Unknown camera")
    return StreamingResponse(
        mjpeg(cameras[cam_id]),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
