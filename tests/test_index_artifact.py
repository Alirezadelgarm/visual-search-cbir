from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from visual_search_cbir.index import CBIRIndex


def test_index_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    sift_data = rng.normal(size=(20, 4)).astype(np.float32)
    harris_data = rng.normal(size=(20, 6)).astype(np.float32)
    sift_kmeans = MiniBatchKMeans(n_clusters=2, random_state=1, n_init="auto").fit(sift_data)
    harris_kmeans = MiniBatchKMeans(n_clusters=2, random_state=2, n_init="auto").fit(harris_data)

    index = CBIRIndex(
        dataset_root_hint="dataset",
        relative_paths=("cars/a.jpg", "doors/b.jpg"),
        labels=("cars", "doors"),
        sift_kmeans=sift_kmeans,
        harris_kmeans=harris_kmeans,
        sift_histograms=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        harris_histograms=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        feature_config={
            "max_image_side": 512,
            "sift_clusters": 64,
            "harris_clusters": 64,
            "sift_max_descriptors_per_image": 250,
            "harris_max_descriptors_per_image": 300,
            "harris_max_points": 300,
            "harris_patch_size": 16,
            "sift_sample_cap": 60000,
            "harris_sample_cap": 60000,
            "random_state": 123,
        },
    )
    artifact = tmp_path / "index.joblib"
    index.save(artifact)

    loaded = CBIRIndex.load(artifact)
    assert loaded.image_count == 2
    assert loaded.labels == ("cars", "doors")
    assert loaded.relative_paths[0] == "cars/a.jpg"
