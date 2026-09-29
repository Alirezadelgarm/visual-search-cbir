from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from visual_search_cbir.config import FeatureConfig
from visual_search_cbir.dataset import DatasetScan, scan_dataset
from visual_search_cbir.exceptions import ArtifactError, DatasetError
from visual_search_cbir.features import (
    extract_harris_patches,
    extract_sift,
    l2_normalize,
    preprocess_image,
)
from visual_search_cbir.vocabulary import fit_vocabulary, sample_descriptors, visual_word_histogram


ARTIFACT_VERSION = 1
StatusCallback = Callable[[str], None] | None


@dataclass(slots=True)
class CBIRIndex:
    dataset_root_hint: str
    relative_paths: tuple[str, ...]
    labels: tuple[str, ...]
    sift_kmeans: MiniBatchKMeans
    harris_kmeans: MiniBatchKMeans
    sift_histograms: np.ndarray
    harris_histograms: np.ndarray
    feature_config: dict[str, int]

    @property
    def image_count(self) -> int:
        return len(self.relative_paths)

    @property
    def sift_clusters(self) -> int:
        return int(self.sift_kmeans.n_clusters)

    @property
    def harris_clusters(self) -> int:
        return int(self.harris_kmeans.n_clusters)

    def resolve_path(self, index: int, dataset_root: Path | None = None) -> Path:
        root = Path(dataset_root) if dataset_root is not None else Path(self.dataset_root_hint)
        return root / self.relative_paths[index]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"artifact_version": ARTIFACT_VERSION, "index": self}
        try:
            joblib.dump(payload, path)
        except Exception as exc:
            raise ArtifactError(f"Could not save index artifact: {path}") from exc

    @classmethod
    def load(cls, path: Path) -> CBIRIndex:
        if not path.exists():
            raise ArtifactError(f"Index artifact does not exist: {path}")
        try:
            payload = joblib.load(path)
        except Exception as exc:
            raise ArtifactError(f"Could not load index artifact: {path}") from exc

        if payload.get("artifact_version") != ARTIFACT_VERSION:
            raise ArtifactError("Unsupported index artifact version.")
        index = payload.get("index")
        if not isinstance(index, cls):
            raise ArtifactError("Artifact does not contain a valid CBIRIndex.")
        return index


def _status(callback: StatusCallback, message: str) -> None:
    if callback is not None:
        callback(message)


def _subsample_rows(array: np.ndarray, maximum: int, rng: np.random.Generator) -> np.ndarray:
    if array.shape[0] <= maximum:
        return array
    selection = rng.choice(array.shape[0], size=maximum, replace=False)
    return array[selection]


def _extract_dataset_descriptors(
    scan: DatasetScan,
    config: FeatureConfig,
    callback: StatusCallback,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rng = np.random.default_rng(config.random_state)
    sift_sets: list[np.ndarray] = []
    harris_sets: list[np.ndarray] = []

    for position, image_path in enumerate(scan.image_paths, start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            sift_sets.append(np.empty((0, 128), dtype=np.float32))
            harris_sets.append(
                np.empty((0, config.harris_patch_size**2), dtype=np.float32)
            )
            continue

        _, gray = preprocess_image(image, config.max_image_side)
        sift = _subsample_rows(
            extract_sift(gray),
            config.sift_max_descriptors_per_image,
            rng,
        )
        harris = _subsample_rows(
            extract_harris_patches(
                gray,
                max_points=config.harris_max_points,
                patch_size=config.harris_patch_size,
            ),
            config.harris_max_descriptors_per_image,
            rng,
        )
        sift_sets.append(sift)
        harris_sets.append(harris)

        if position % 50 == 0 or position == scan.total_images:
            _status(callback, f"Extracted descriptors from {position}/{scan.total_images} images")

    return sift_sets, harris_sets


def build_index(
    dataset_root: Path,
    artifact_path: Path,
    config: FeatureConfig | None = None,
    status_callback: StatusCallback = None,
) -> CBIRIndex:
    config = config or FeatureConfig()
    scan = scan_dataset(dataset_root)
    _status(status_callback, f"Dataset root: {scan.root}")
    _status(status_callback, f"Images found: {scan.total_images}")
    _status(status_callback, f"Missing classes: {list(scan.missing_classes)}")

    sift_sets, harris_sets = _extract_dataset_descriptors(scan, config, status_callback)
    sift_total = sum(descriptors.shape[0] for descriptors in sift_sets)
    harris_total = sum(descriptors.shape[0] for descriptors in harris_sets)
    _status(status_callback, f"SIFT descriptors: {sift_total}")
    _status(status_callback, f"Harris descriptors: {harris_total}")

    sift_samples = sample_descriptors(
        sift_sets,
        cap=config.sift_sample_cap,
        seed=config.random_state,
    )
    harris_samples = sample_descriptors(
        harris_sets,
        cap=config.harris_sample_cap,
        seed=config.random_state + 1,
    )
    if sift_samples.shape[0] < 10:
        raise DatasetError("Not enough SIFT descriptors to build a visual vocabulary.")
    if harris_samples.shape[0] < 10:
        raise DatasetError("Not enough Harris descriptors to build a visual vocabulary.")

    _status(status_callback, "Fitting SIFT visual vocabulary")
    sift_kmeans = fit_vocabulary(
        sift_samples,
        cluster_count=config.sift_clusters,
        seed=config.random_state,
    )
    _status(status_callback, "Fitting Harris visual vocabulary")
    harris_kmeans = fit_vocabulary(
        harris_samples,
        cluster_count=config.harris_clusters,
        seed=config.random_state + 1,
    )

    sift_histograms = np.zeros(
        (scan.total_images, int(sift_kmeans.n_clusters)),
        dtype=np.float32,
    )
    harris_histograms = np.zeros(
        (scan.total_images, int(harris_kmeans.n_clusters)),
        dtype=np.float32,
    )

    for index, (sift, harris) in enumerate(zip(sift_sets, harris_sets, strict=True)):
        if sift.shape[0] > 0:
            words = sift_kmeans.predict(sift)
            sift_histograms[index] = visual_word_histogram(words, sift_kmeans.n_clusters)
        if harris.shape[0] > 0:
            words = harris_kmeans.predict(harris)
            harris_histograms[index] = visual_word_histogram(words, harris_kmeans.n_clusters)

    sift_histograms = l2_normalize(sift_histograms)
    harris_histograms = l2_normalize(harris_histograms)

    relative_paths = tuple(str(path.relative_to(scan.root)) for path in scan.image_paths)
    cbir_index = CBIRIndex(
        dataset_root_hint=str(scan.root),
        relative_paths=relative_paths,
        labels=scan.labels,
        sift_kmeans=sift_kmeans,
        harris_kmeans=harris_kmeans,
        sift_histograms=sift_histograms,
        harris_histograms=harris_histograms,
        feature_config=config.to_dict(),
    )
    cbir_index.save(artifact_path)
    _status(status_callback, f"Saved index: {artifact_path}")
    return cbir_index
