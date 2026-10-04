import pytest
from src.sentiment.analyzer import SentimentAnalyzer


def test_sentiment_analyzer_loaded():
    analyzer = SentimentAnalyzer()
    assert analyzer.is_loaded() is True


def test_sentiment_analyzer_predict_positive():
    analyzer = SentimentAnalyzer()
    res = analyzer.predict("Absolutely love using this service! It is amazing.")
    assert "sentiment" in res
    assert "confidence" in res
    assert res["sentiment"] == "positive"
    assert 0.0 <= res["confidence"] <= 1.0


def test_sentiment_analyzer_predict_negative():
    analyzer = SentimentAnalyzer()
    res = analyzer.predict("Terrible experience. Worst customer support and late delivery.")
    assert res["sentiment"] == "negative"
    assert 0.0 <= res["confidence"] <= 1.0


def test_sentiment_analyzer_predict_batch():
    analyzer = SentimentAnalyzer()
    texts = [
        "Highly recommended! Five stars.",
        "Worst app ever, completely broken.",
        "Just downloaded the app today."
    ]
    results = analyzer.predict_batch(texts)
    assert len(results) == 3
    assert results[0]["sentiment"] == "positive"
    assert results[1]["sentiment"] == "negative"


def test_sentiment_analyzer_edge_cases():
    analyzer = SentimentAnalyzer()
    # Very short comments
    res_short = analyzer.predict("good")
    assert "sentiment" in res_short

    # URLs and hashtags
    res_url = analyzer.predict("https://example.com #awesome @brand Great!")
    assert res_url["sentiment"] == "positive"


def test_sentiment_analyzer_untrained_error():
    analyzer = SentimentAnalyzer(model_dir="non_existent_dir_123")
    assert analyzer.is_loaded() is False
    with pytest.raises(RuntimeError):
        analyzer.predict("Test")
