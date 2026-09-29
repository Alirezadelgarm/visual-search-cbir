from __future__ import annotations

import cv2
import numpy as np


def preprocess_image(image_bgr: np.ndarray, max_side: int) -> tuple[np.ndarray, np.ndarray]:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Image is empty.")

    height, width = image_bgr.shape[:2]
    largest_side = max(height, width)
    if largest_side > max_side:
        scale = max_side / float(largest_side)
        new_size = (int(round(width * scale)), int(round(height * scale)))
        image_bgr = cv2.resize(image_bgr, new_size, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return image_bgr, gray


def l2_normalize(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / (norm + eps)
