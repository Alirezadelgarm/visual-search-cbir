from visual_search_cbir.evaluation import average_precision_at_k, precision_at_k


def test_retrieval_metrics() -> None:
    labels = ["cars", "cars", "doors", "cars", "windows"]

    assert precision_at_k(labels, "cars", 5) == 0.6
    assert average_precision_at_k(
        labels,
        "cars",
        5,
        total_relevant=3,
    ) > 0.7
