import os
import torch
from dataclasses import dataclass


@dataclass
class SystemConfig:
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    fps: int = 30

    use_gpu: bool = torch.cuda.is_available()
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    face_detection_confidence: float = 0.90
    identity_threshold: float = 0.75
    temporal_window_seconds: int = 3

    log_level: str = "INFO"
    log_file: str = "logs/system.log"


config = SystemConfig()
