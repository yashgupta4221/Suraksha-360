import cv2
import threading
import time


class VideoStream:
    def __init__(self, source):
        self.source = int(source) if str(source).isdigit() else source
        self.lock = threading.Lock()
        self.frame = None
        self.status = False
        self.last_seen = 0.0
        self.stopped = False
        threading.Thread(target=self._update, daemon=True).start()

    def _open(self):
        cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _update(self):
        cap = self._open()
        while not self.stopped:
            ok, frame = cap.read()
            if not ok:
                self.status = False
                cap.release()
                time.sleep(2)
                cap = self._open()
                continue
            with self.lock:
                self.frame = frame
                self.status = True
                self.last_seen = time.time()
        cap.release()

    def read(self):
        with self.lock:
            return self.status, (None if self.frame is None else self.frame.copy())

    def stop(self):
        self.stopped = True
