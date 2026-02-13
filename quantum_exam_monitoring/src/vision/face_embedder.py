from typing import Optional
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from core.config import config
from core.logger import logger


class FaceEmbedder:
    def __init__(self):
        logger.info("Initializing FaceEmbedder...")

        self.device = torch.device(config.device)

        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        logger.info(f"FaceEmbedder loaded on device: {self.device}")

    def get_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate embedding vector for a cropped face.
        """
        try:
            tensor = self.preprocess(face_image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                embedding = self.model(tensor)

            return embedding.cpu().numpy()[0]

        except Exception as e:
            logger.exception("Failed to generate embedding.")
            return None
