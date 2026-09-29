from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from visual_search_cbir.config import SearchConfig
from visual_search_cbir.index import CBIRIndex
from visual_search_cbir.retrieval import compute_query_vectors, search_top_k


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    query_count: int
    precision_at_5: float
    precision_at_10: float
    map_at_10: float


def precision_at_k(retrieved_labels: list[str], expected_label: str, k: int) -> float:
    if k <= 0:
        raise ValueError("k must be greater than zero.")
    considered = retrieved_labels[:k]
    if not considered:
        return 0.0
    relevant = sum(label == expected_label for label in considered)
    return relevant / len(considered)


def average_precision_at_k(
    retrieved_labels: list[str],
    expected_label: str,
    k: int,
    *,
    total_relevant: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be greater than zero.")
    if total_relevant <= 0:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for rank, label in enumerate(retrieved_labels[:k], start=1):
        if label == expected_label:
            hits += 1
            precision_sum += hits / rank

    denominator = min(total_relevant, k)
    return precision_sum / denominator


def evaluate_index(
    index: CBIRIndex,
    *,
    dataset_root: Path | None = None,
    max_queries: int | None = None,
    seed: int = 123,
) -> EvaluationSummary:
    indices = np.arange(index.image_count)
    if max_queries is not None and 0 < max_queries < len(indices):
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(indices, size=max_queries, replace=False))

    class_counts: dict[str, int] = {}
    for label in index.labels:
        class_counts[label] = class_counts.get(label, 0) + 1

    p5_values: list[float] = []
    p10_values: list[float] = []
    ap_values: list[float] = []

    for raw_index in indices:
        query_index = int(raw_index)
        query_path = index.resolve_path(query_index, dataset_root)
        sift_query, harris_query = compute_query_vectors(index, query_path)
        hits = search_top_k(
            index,
            sift_query,
            harris_query,
            dataset_root=dataset_root,
            search_config=SearchConfig(top_k=10),
            exclude_indices={query_index},
        )
        labels = [hit.label for hit in hits]
        expected = index.labels[query_index]
        total_relevant = max(0, class_counts[expected] - 1)
        p5_values.append(precision_at_k(labels, expected, 5))
        p10_values.append(precision_at_k(labels, expected, 10))
        ap_values.append(
            average_precision_at_k(
                labels,
                expected,
                10,
                total_relevant=total_relevant,
            )
        )

    return EvaluationSummary(
        query_count=len(p5_values),
        precision_at_5=float(np.mean(p5_values)) if p5_values else 0.0,
        precision_at_10=float(np.mean(p10_values)) if p10_values else 0.0,
        map_at_10=float(np.mean(ap_values)) if ap_values else 0.0,
    )
