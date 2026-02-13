import os
import cv2
import numpy as np
from typing import Dict, Tuple, Optional
from vision.face_embedder import FaceEmbedder
from utils.similarity import cosine_similarity
from core.config import config
from core.logger import logger


class AttendanceService:
    def __init__(self, data_path: str = "data/registered_faces"):
        self.embedder = FaceEmbedder()
        self.data_path = data_path
        self.known_embeddings: Dict[str, np.ndarray] = {}

        self._load_registered_faces()

    def _load_registered_faces(self):
        logger.info("Loading registered faces...")

        for candidate_id in os.listdir(self.data_path):
            candidate_folder = os.path.join(self.data_path, candidate_id)

            if not os.path.isdir(candidate_folder):
                continue

            embeddings = []

            for img_name in os.listdir(candidate_folder):
                img_path = os.path.join(candidate_folder, img_name)
                image = cv2.imread(img_path)

                if image is None:
                    continue

                embedding = self.embedder.get_embedding(image)

                if embedding is not None:
                    embeddings.append(embedding)

            if embeddings:
                mean_embedding = np.mean(embeddings, axis=0)
                self.known_embeddings[candidate_id] = mean_embedding

        logger.info(f"Loaded {len(self.known_embeddings)} registered candidates.")

    def match(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        best_match = None
        highest_score = 0.0

        for candidate_id, known_embedding in self.known_embeddings.items():
            score = cosine_similarity(embedding, known_embedding)

            if score > highest_score:
                highest_score = score
                best_match = candidate_id

        if highest_score >= config.identity_threshold:
            return best_match, highest_score

        return None, highest_score
