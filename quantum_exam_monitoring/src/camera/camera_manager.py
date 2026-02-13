import cv2
from core.config import config
from core.logger import logger
from core.exceptions import CameraInitializationError, FrameCaptureError


class CameraManager:
    def __init__(self):
        self.cap = None

    def initialize(self):
        logger.info("Initializing camera...")

        self.cap = cv2.VideoCapture(config.camera_index)

        if not self.cap.isOpened():
            logger.error("Camera failed to initialize.")
            raise CameraInitializationError("Unable to open camera.")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, config.fps)

        logger.info("Camera initialized successfully.")

    def read_frame(self):
        if self.cap is None:
            raise CameraInitializationError("Camera not initialized.")

        ret, frame = self.cap.read()

        if not ret:
            logger.error("Failed to capture frame.")
            raise FrameCaptureError("Frame capture failed.")

        return frame

    def release(self):
        if self.cap:
            self.cap.release()
            logger.info("Camera released.")
