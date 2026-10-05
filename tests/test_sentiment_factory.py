import pytest
from src.sentiment.factory import get_sentiment_analyzer
from src.sentiment.analyzer import SentimentAnalyzer
from src.sentiment.distilbert_analyzer import DistilBertSentimentAnalyzer


def test_factory_returns_distilbert_by_default():
    analyzer = get_sentiment_analyzer("distilbert")
    assert isinstance(analyzer, DistilBertSentimentAnalyzer)
    assert analyzer.is_loaded() is True


def test_factory_returns_tfidf_explicit():
    analyzer = get_sentiment_analyzer("tfidf")
    assert isinstance(analyzer, SentimentAnalyzer)
    assert analyzer.is_loaded() is True


def test_factory_fallback_on_invalid_distilbert_path():
    analyzer = get_sentiment_analyzer("distilbert", model_dir="non_existent_folder_abc123")
    assert isinstance(analyzer, SentimentAnalyzer)
    assert analyzer.is_loaded() is True


def test_factory_analyzers_interface_uniformity():
    distilbert = get_sentiment_analyzer("distilbert")
    tfidf = get_sentiment_analyzer("tfidf")

    text = "The product arrived broken and customer support was rude."
    res_d = distilbert.predict(text)
    res_t = tfidf.predict(text)

    for res in [res_d, res_t]:
        assert "sentiment" in res
        assert "confidence" in res
        assert res["sentiment"] in ["positive", "neutral", "negative"]
        assert 0.0 <= res["confidence"] <= 1.0

    batch_texts = ["Great experience!", "Awful delay and bad packaging."]
    b_res_d = distilbert.predict_batch(batch_texts)
    b_res_t = tfidf.predict_batch(batch_texts)

    assert len(b_res_d) == 2
    assert len(b_res_t) == 2
