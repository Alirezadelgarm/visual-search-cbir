from __future__ import annotations

import numpy as np

from visual_search_cbir.features import l2_normalize


def rocchio_refine(
    query: np.ndarray,
    relevant_vectors: np.ndarray,
    feedback_lambda: float = 0.60,
) -> np.ndarray:
    """Move a query vector toward the centroid of user-selected relevant results."""
    if relevant_vectors.size == 0:
        return np.asarray(query, dtype=np.float32)

    lam = min(1.0, max(0.0, float(feedback_lambda)))
    centroid = relevant_vectors.mean(axis=0)
    centroid = l2_normalize(centroid.reshape(1, -1)).reshape(-1)
    refined = (1.0 - lam) * query + lam * centroid
    return l2_normalize(refined.reshape(1, -1)).reshape(-1)
