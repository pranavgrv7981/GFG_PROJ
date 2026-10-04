import pytest
from fastapi.testclient import TestClient
from src.api.app import app, session_data

client = TestClient(app)


def setup_function():
    session_data.clear()


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_model_info_endpoint():
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment_model_loaded"] is True
    assert data["clustering_model_loaded"] is True


def test_clusters_endpoint():
    response = client.get("/clusters")
    assert response.status_code == 200
    data = response.json()
    assert "clusters" in data
    assert isinstance(data["clusters"], list)
    assert len(data["clusters"]) > 0


def test_predict_single_positive():
    payload = {
        "text": "Amazing experience with @Swiggy, super fast delivery!",
        "platform": "twitter",
        "brand": "Swiggy"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "positive"
    assert data["is_negative"] is False
    assert data["complaint_cluster"] is None
    assert data["platform"] == "twitter"
    assert data["brand"] == "Swiggy"


def test_predict_single_negative():
    payload = {
        "text": "The app keeps crashing when I try to pay! Fix this @PhonePe",
        "platform": "reddit",
        "brand": "PhonePe"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "negative"
    assert data["is_negative"] is True
    assert data["complaint_cluster"] is not None
    assert data["complaint_category"] is not None


def test_predict_single_empty_text():
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 400


def test_predict_single_missing_text():
    response = client.post("/predict", json={"platform": "twitter"})
    assert response.status_code == 422


def test_predict_batch():
    payload = {
        "comments": [
            {"text": "Loving this brand so far!", "platform": "twitter", "brand": "Amazon"},
            {"text": "Order was 4 days late and damaged.", "platform": "instagram", "brand": "Flipkart"},
            {"text": "Is the sale live today?", "platform": "facebook", "brand": "Amazon"}
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 3
    assert "summary" in data
    assert data["summary"]["total"] == 3


def test_predict_batch_empty():
    response = client.post("/predict/batch", json={"comments": []})
    assert response.status_code == 400


def test_insights_endpoint():
    # Make a prediction to populate session data
    client.post("/predict", json={"text": "Awesome product!", "platform": "twitter", "brand": "Amazon"})
    client.post("/predict", json={"text": "Overcharged twice on invoice!", "platform": "reddit", "brand": "Swiggy"})

    response = client.get("/insights")
    assert response.status_code == 200
    data = response.json()
    assert data["total_comments"] == 2
    assert data["positive_count"] >= 1
    assert data["negative_count"] >= 1
    assert "complaint_categories" in data
    assert "platform_distribution" in data
    assert "brand_distribution" in data
