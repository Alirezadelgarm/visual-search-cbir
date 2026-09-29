import numpy as np

from visual_search_cbir.feedback import rocchio_refine


def test_rocchio_moves_query_toward_relevant_centroid() -> None:
    query = np.array([1.0, 0.0], dtype=np.float32)
    relevant = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)

    refined = rocchio_refine(query, relevant, feedback_lambda=0.75)

    assert refined[1] > refined[0]
    assert np.isclose(np.linalg.norm(refined), 1.0)
