from pathlib import Path

from visual_search_cbir.dataset import detect_dataset_root, scan_dataset


def test_detect_single_dataset_and_scan(tmp_path: Path) -> None:
    root = tmp_path / "msrcorid" / "Single"
    for class_name in ("aeroplanes", "cars"):
        class_dir = root / class_name
        class_dir.mkdir(parents=True)
        (class_dir / f"{class_name}.jpg").write_bytes(b"not-decoded-in-this-test")

    assert detect_dataset_root(tmp_path) == root.resolve()
    scan = scan_dataset(tmp_path)
    assert scan.total_images == 2
    assert scan.counts["aeroplanes"] == 1
    assert scan.counts["cars"] == 1
    assert "doors" in scan.missing_classes
