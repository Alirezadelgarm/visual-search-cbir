from __future__ import annotations

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from visual_search_cbir.features import l2_normalize


def sample_descriptors(descriptor_sets: list[np.ndarray], cap: int, seed: int) -> np.ndarray:
    usable = [descriptors for descriptors in descriptor_sets if descriptors.size > 0]
    if not usable:
        return np.empty((0, 1), dtype=np.float32)

    stacked = np.vstack(usable).astype(np.float32)
    if stacked.shape[0] <= cap:
        return stacked

    rng = np.random.default_rng(seed)
    selection = rng.choice(stacked.shape[0], size=cap, replace=False)
    return stacked[selection]


def fit_vocabulary(descriptors: np.ndarray, cluster_count: int, seed: int) -> MiniBatchKMeans:
    if descriptors.ndim != 2 or descriptors.shape[0] < 2:
        raise ValueError("At least two descriptors are required to fit a visual vocabulary.")

    clusters = min(max(2, int(cluster_count)), descriptors.shape[0])
    model = MiniBatchKMeans(
        n_clusters=clusters,
        random_state=seed,
        batch_size=2048,
        n_init="auto",
        max_iter=200,
        reassignment_ratio=0.01,
    )
    model.fit(descriptors)
    return model


def visual_word_histogram(words: np.ndarray, cluster_count: int) -> np.ndarray:
    histogram = np.bincount(words, minlength=cluster_count).astype(np.float32)
    return l2_normalize(histogram.reshape(1, -1)).reshape(-1)
