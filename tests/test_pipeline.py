import pytest
from src.pipeline import Pipeline


def test_pipeline_process_positive_comment():
    pipe = Pipeline()
    res = pipe.process_comment(
        text="Great delivery by @Amazon, loved it! #happy",
        platform="twitter",
        brand="Amazon"
    )
    assert res["sentiment"] == "positive"
    assert res["is_negative"] is False
    assert res["complaint_cluster"] is None
    assert res["complaint_category"] is None
    assert res["brand"] == "Amazon"
    assert res["platform"] == "twitter"


def test_pipeline_process_negative_comment():
    pipe = Pipeline()
    res = pipe.process_comment(
        text="Customer support never replied and refund is missing! @Swiggy",
        platform="instagram",
        brand="Swiggy"
    )
    assert res["sentiment"] == "negative"
    assert res["is_negative"] is True
    assert res["complaint_cluster"] is not None
    assert res["complaint_category"] is not None


def test_pipeline_invalid_comment():
    pipe = Pipeline()
    res = pipe.process_comment("")
    assert "error" in res

    res_whitespace = pipe.process_comment("   ")
    assert "error" in res_whitespace


def test_pipeline_process_batch():
    pipe = Pipeline()
    batch = [
        {"text": "Fantastic product and rapid delivery!", "platform": "twitter", "brand": "Amazon"},
        {"text": "App crashed and money was deducted twice!", "platform": "reddit", "brand": "PhonePe"},
        {"text": "What is the update on order?", "platform": "facebook", "brand": "Flipkart"},
        {"text": ""}  # empty comment should be safely skipped
    ]
    results = pipe.process_batch(batch)
    assert len(results) == 3
    assert results[0]["sentiment"] == "positive"
    assert results[1]["sentiment"] == "negative"
    assert results[1]["complaint_cluster"] is not None
