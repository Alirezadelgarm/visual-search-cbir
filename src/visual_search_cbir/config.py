from __future__ import annotations

from dataclasses import asdict, dataclass


REQUIRED_CLASSES: tuple[str, ...] = (
    "aeroplanes",
    "benches_and_chairs",
    "buildings",
    "cars",
    "doors",
    "signs",
    "windows",
)

SUPPORTED_EXTENSIONS: tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
)


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    max_image_side: int = 512
    sift_clusters: int = 64
    harris_clusters: int = 64
    sift_max_descriptors_per_image: int = 250
    harris_max_descriptors_per_image: int = 300
    harris_max_points: int = 300
    harris_patch_size: int = 16
    sift_sample_cap: int = 60_000
    harris_sample_cap: int = 60_000
    random_state: int = 123

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SearchConfig:
    top_k: int = 10
    sift_weight: float = 0.60
    harris_weight: float = 0.40
    feedback_lambda: float = 0.60

    def normalized_weights(self) -> tuple[float, float]:
        sift = max(0.0, float(self.sift_weight))
        harris = max(0.0, float(self.harris_weight))
        total = sift + harris
        if total <= 1e-12:
            return 0.5, 0.5
        return sift / total, harris / total
