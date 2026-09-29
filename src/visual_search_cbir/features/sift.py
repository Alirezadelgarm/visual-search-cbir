from __future__ import annotations

import cv2
import numpy as np


def extract_sift(gray: np.ndarray) -> np.ndarray:
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("This OpenCV build does not provide cv2.SIFT_create().")

    detector = cv2.SIFT_create()
    _, descriptors = detector.detectAndCompute(gray, None)
    if descriptors is None:
        return np.empty((0, 128), dtype=np.float32)
    return np.asarray(descriptors, dtype=np.float32)
