import numpy as np

from visual_search_cbir.vocabulary import visual_word_histogram


def test_visual_word_histogram_is_l2_normalized() -> None:
    histogram = visual_word_histogram(np.array([0, 0, 1, 2, 2, 2]), cluster_count=4)

    assert histogram.shape == (4,)
    assert np.isclose(np.linalg.norm(histogram), 1.0)
