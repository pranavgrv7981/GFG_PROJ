import pytest
from src.clustering.clusterer import ComplaintClusterer


def test_clusterer_loaded():
    clusterer = ComplaintClusterer()
    assert clusterer.is_loaded() is True
    assert len(clusterer.get_cluster_labels()) > 0
    assert len(clusterer.get_cluster_themes()) > 0


def test_clusterer_predict_single():
    clusterer = ComplaintClusterer()
    res = clusterer.predict("Still waiting for my refund after 2 weeks, where is my money?")
    assert "cluster_id" in res
    assert isinstance(res["cluster_id"], int)
    assert "cluster_label" in res
    assert "complaint_category" in res
    assert len(res["complaint_category"]) > 0


def test_clusterer_predict_batch():
    clusterer = ComplaintClusterer()
    texts = [
        "Delivery was 5 days late!",
        "Charged twice on my credit card",
        "App keeps crashing on checkout page"
    ]
    results = clusterer.predict_batch(texts)
    assert len(results) == 3
    for r in results:
        assert "cluster_id" in r
        assert "complaint_category" in r
        assert isinstance(r["cluster_id"], int)


def test_clusterer_edge_cases():
    clusterer = ComplaintClusterer()
    # Empty string or single word
    res_empty = clusterer.predict("")
    assert "cluster_id" in res_empty

    res_short = clusterer.predict("broken")
    assert "cluster_id" in res_short

    # Unknown tokens
    res_unknown = clusterer.predict("xyzabc random words 98765")
    assert "cluster_id" in res_unknown


def test_clusterer_unloaded_error():
    clusterer = ComplaintClusterer(model_dir="non_existent_clustering_dir_123")
    assert clusterer.is_loaded() is False
    with pytest.raises(RuntimeError):
        clusterer.predict("Some complaint")
