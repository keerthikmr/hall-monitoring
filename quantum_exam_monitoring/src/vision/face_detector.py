from typing import List, Dict, Any
import torch
import numpy as np
from facenet_pytorch import MTCNN
from core.config import config
from core.logger import logger


class FaceDetector:
    def __init__(self):
        logger.info("Initializing FaceDetector...")

        self.device = torch.device(config.device)

        self.detector = MTCNN(
            keep_all=True,
            device=self.device,
            thresholds=[0.6, 0.7, config.face_detection_confidence]
        )

        logger.info(f"FaceDetector loaded on device: {self.device}")

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in frame.

        Returns:
            List of dict:
            [
                {
                    "box": [x1, y1, x2, y2],
                    "confidence": float
                }
            ]
        """
        boxes, probs = self.detector.detect(frame)

        results = []

        if boxes is None:
            return results

        for box, prob in zip(boxes, probs):
            if prob is None:
                continue

            x1, y1, x2, y2 = map(int, box)

            results.append({
                "box": [x1, y1, x2, y2],
                "confidence": float(prob)
            })

        return results
