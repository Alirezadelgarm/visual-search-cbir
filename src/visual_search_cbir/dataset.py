from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from visual_search_cbir.config import REQUIRED_CLASSES, SUPPORTED_EXTENSIONS
from visual_search_cbir.exceptions import DatasetError


@dataclass(frozen=True, slots=True)
class DatasetScan:
    root: Path
    image_paths: tuple[Path, ...]
    labels: tuple[str, ...]
    counts: dict[str, int]
    missing_classes: tuple[str, ...]

    @property
    def total_images(self) -> int:
        return len(self.image_paths)


def detect_dataset_root(selected_path: Path) -> Path:
    """Find the directory that directly contains the assignment class folders."""
    selected = selected_path.expanduser().resolve()
    single = selected / "Single"
    if single.is_dir():
        return single

    msrc = selected / "msrcorid"
    if msrc.is_dir():
        nested_single = msrc / "Single"
        if nested_single.is_dir():
            return nested_single
        return msrc

    return selected


def _iter_supported_images(class_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def scan_dataset(dataset_root: Path) -> DatasetScan:
    root = detect_dataset_root(dataset_root)
    paths: list[Path] = []
    labels: list[str] = []
    counts: dict[str, int] = {}
    missing: list[str] = []

    for class_name in REQUIRED_CLASSES:
        class_dir = root / class_name
        if not class_dir.is_dir():
            counts[class_name] = 0
            missing.append(class_name)
            continue

        class_paths = _iter_supported_images(class_dir)
        counts[class_name] = len(class_paths)
        paths.extend(class_paths)
        labels.extend([class_name] * len(class_paths))

    if not paths:
        raise DatasetError(
            "No supported images were found under the required class folders. "
            f"Checked dataset root: {root}"
        )

    return DatasetScan(
        root=root,
        image_paths=tuple(paths),
        labels=tuple(labels),
        counts=counts,
        missing_classes=tuple(missing),
    )
