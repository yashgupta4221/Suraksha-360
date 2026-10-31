import os
import threading

from ultralytics import YOLO


class PersonDetector:
    """YOLOv8 pose model + ByteTrack. Returns, for every person: a stable
    track ID, box, confidence and 17 COCO keypoints."""

    def __init__(self):
        self.model = YOLO(os.getenv("YOLO_MODEL", "yolov8n-pose.pt"))
        self.conf = float(os.getenv("PERSON_CONF", "0.5"))
        self.device = os.getenv("YOLO_DEVICE", "cpu")
        self.tracker = os.getenv("TRACKER", os.path.join(os.path.dirname(__file__), "tracker.yaml"))
        self.lock = threading.Lock()

    def detect(self, frame, track=True):
        with self.lock:
            if track:
                result = self.model.track(
                    frame, conf=self.conf, classes=[0], device=self.device,
                    persist=True, tracker=self.tracker, verbose=False,
                )[0]
            else:
                result = self.model.predict(
                    frame, conf=self.conf, classes=[0], device=self.device, verbose=False
                )[0]

        detections = []
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else None
        kps = result.keypoints
        kp_xy = kps.xy.cpu().numpy() if kps is not None else None
        kp_conf = kps.conf.cpu().numpy() if kps is not None and kps.conf is not None else None

        for i, (box, conf) in enumerate(zip(boxes, confs)):
            detections.append({
                "id": int(ids[i]) if ids is not None else None,
                "box": [int(v) for v in box],
                "conf": round(float(conf), 3),
                "keypoints": kp_xy[i].round(1).tolist() if kp_xy is not None else [],
                "kp_conf": kp_conf[i].round(3).tolist() if kp_conf is not None else [],
            })
        return detections
