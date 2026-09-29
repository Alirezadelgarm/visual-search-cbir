from __future__ import annotations

import cv2
import numpy as np


def extract_harris_patches(gray: np.ndarray, max_points: int, patch_size: int) -> np.ndarray:
    """Extract normalized grayscale patches around strong Harris corners."""
    if patch_size <= 0 or patch_size % 2 != 0:
        raise ValueError("patch_size must be a positive even integer.")

    gray_float = np.asarray(gray, dtype=np.float32) / 255.0
    response = cv2.cornerHarris(gray_float, blockSize=2, ksize=3, k=0.04)
    response = cv2.dilate(response, None)

    if response.size == 0 or float(response.max()) <= 0.0:
        return np.empty((0, patch_size * patch_size), dtype=np.float32)

    threshold = 0.01 * float(response.max())
    ys, xs = np.where(response > threshold)
    if len(xs) == 0:
        return np.empty((0, patch_size * patch_size), dtype=np.float32)

    scores = response[ys, xs]
    order = np.argsort(scores)[::-1][:max_points]
    xs = xs[order]
    ys = ys[order]

    half = patch_size // 2
    height, width = gray.shape[:2]
    patches: list[np.ndarray] = []

    for x, y in zip(xs, ys, strict=True):
        x0, x1 = int(x - half), int(x + half)
        y0, y1 = int(y - half), int(y + half)
        if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
            continue

        patch = gray_float[y0:y1, x0:x1]
        if patch.shape != (patch_size, patch_size):
            continue

        vector = patch.reshape(-1).astype(np.float32)
        vector -= float(vector.mean())
        vector /= float(vector.std()) + 1e-6
        patches.append(vector)

    if not patches:
        return np.empty((0, patch_size * patch_size), dtype=np.float32)
    return np.stack(patches, axis=0).astype(np.float32)
