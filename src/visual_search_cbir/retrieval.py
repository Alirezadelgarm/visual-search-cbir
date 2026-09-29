from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from visual_search_cbir.config import FeatureConfig, SearchConfig
from visual_search_cbir.features import (
    extract_harris_patches,
    extract_sift,
    l2_normalize,
    preprocess_image,
)
from visual_search_cbir.index import CBIRIndex
from visual_search_cbir.vocabulary import visual_word_histogram


@dataclass(frozen=True, slots=True)
class SearchHit:
    index: int
    score: float
    label: str
    path: Path


def compute_query_vectors(
    index: CBIRIndex,
    query_path: Path,
    config: FeatureConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    config = config or FeatureConfig(**index.feature_config)
    image = cv2.imread(str(query_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read query image: {query_path}")

    _, gray = preprocess_image(image, config.max_image_side)
    sift_descriptors = extract_sift(gray)
    harris_descriptors = extract_harris_patches(
        gray,
        max_points=config.harris_max_points,
        patch_size=config.harris_patch_size,
    )

    if sift_descriptors.shape[0] > 0:
        sift_words = index.sift_kmeans.predict(sift_descriptors)
        sift_query = visual_word_histogram(sift_words, index.sift_clusters)
    else:
        sift_query = np.zeros(index.sift_clusters, dtype=np.float32)

    if harris_descriptors.shape[0] > 0:
        harris_words = index.harris_kmeans.predict(harris_descriptors)
        harris_query = visual_word_histogram(harris_words, index.harris_clusters)
    else:
        harris_query = np.zeros(index.harris_clusters, dtype=np.float32)

    return (
        l2_normalize(sift_query.reshape(1, -1)).reshape(-1),
        l2_normalize(harris_query.reshape(1, -1)).reshape(-1),
    )


def search_top_k(
    index: CBIRIndex,
    sift_query: np.ndarray,
    harris_query: np.ndarray,
    *,
    dataset_root: Path | None = None,
    search_config: SearchConfig | None = None,
    exclude_indices: set[int] | None = None,
) -> list[SearchHit]:
    search_config = search_config or SearchConfig()
    sift_weight, harris_weight = search_config.normalized_weights()

    sift_scores = index.sift_histograms @ sift_query
    harris_scores = index.harris_histograms @ harris_query
    scores = sift_weight * sift_scores + harris_weight * harris_scores

    excluded = exclude_indices or set()
    order = np.argsort(scores)[::-1]
    hits: list[SearchHit] = []
    for raw_index in order:
        image_index = int(raw_index)
        if image_index in excluded:
            continue
        hits.append(
            SearchHit(
                index=image_index,
                score=float(scores[image_index]),
                label=index.labels[image_index],
                path=index.resolve_path(image_index, dataset_root),
            )
        )
        if len(hits) >= search_config.top_k:
            break
    return hits
