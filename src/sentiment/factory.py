import logging
from typing import Optional, Union
from src.config import settings
from src.sentiment.analyzer import SentimentAnalyzer
from src.sentiment.distilbert_analyzer import DistilBertSentimentAnalyzer

logger = logging.getLogger(__name__)


def get_sentiment_analyzer(
    model_name: Optional[str] = None,
    model_dir: Optional[str] = None,
    device: Optional[str] = None
) -> Union[DistilBertSentimentAnalyzer, SentimentAnalyzer]:
    selected = (model_name or getattr(settings, "SENTIMENT_MODEL", "distilbert")).lower()

    if selected in ["distilbert", "transformer", "bert"]:
        try:
            target_dir = model_dir or "models/distilbert"
            analyzer = DistilBertSentimentAnalyzer(model_dir=target_dir, device=device)
            if analyzer.is_loaded():
                return analyzer
            logger.warning("DistilBERT model artifacts not found at %s. Falling back to TF-IDF.", target_dir)
        except Exception as e:
            logger.warning("Failed to initialize DistilBERT (%s). Falling back to TF-IDF.", e)

    target_dir = model_dir if (model_dir and selected in ["tfidf", "logistic_regression"]) else "models/sentiment"
    return SentimentAnalyzer(model_dir=target_dir)
