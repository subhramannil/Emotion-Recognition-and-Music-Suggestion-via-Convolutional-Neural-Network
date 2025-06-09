import cv2
from threading import Thread
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebcamVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not self.stream.isOpened():
            raise IOError("Cannot open webcam")
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False


    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if not grabbed:
                logger.warning("Frame not grabbed, retrying...")
                time.sleep(0.1)
                continue
            self.frame = frame

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        if self.stream.isOpened():
            self.stream.release()
        logger.info("Webcam released")