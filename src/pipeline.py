from src.preprocessing.cleaner import clean_text, preprocess_for_ml
from src.preprocessing.validator import validate_comment
from src.sentiment.analyzer import SentimentAnalyzer
from src.clustering.clusterer import ComplaintClusterer


class Pipeline:
    def __init__(self):
        self._sentiment_analyzer = None
        self._clusterer = None

    @property
    def sentiment_analyzer(self):
        if self._sentiment_analyzer is None:
            self._sentiment_analyzer = SentimentAnalyzer()
        return self._sentiment_analyzer

    @property
    def clusterer(self):
        if self._clusterer is None:
            self._clusterer = ComplaintClusterer()
        return self._clusterer

    def process_comment(self, text, platform=None, brand=None):
        is_valid, msg = validate_comment({"text": text})
        if not is_valid:
            return {"error": msg}

        cleaned = clean_text(text)

        sentiment_result = self.sentiment_analyzer.predict(cleaned)
        sentiment_label = sentiment_result.get("sentiment", "neutral")
        confidence = sentiment_result.get("confidence", 0.0)
        is_negative = sentiment_label == "negative"

        result = {
            "original_text": text,
            "cleaned_text": cleaned,
            "sentiment": sentiment_label,
            "sentiment_confidence": confidence,
            "is_negative": is_negative,
            "platform": platform,
            "brand": brand,
            "complaint_cluster": None,
            "complaint_category": None,
        }

        if is_negative and self.clusterer.is_loaded():
            cluster_result = self.clusterer.predict(cleaned)
            result["complaint_cluster"] = cluster_result.get("cluster_id")
            result["complaint_category"] = cluster_result.get("complaint_category") or cluster_result.get("cluster_label")

        return result

    def process_batch(self, comments):
        results = []
        for comment in comments:
            text = comment.get("text", "")
            platform = comment.get("platform")
            brand = comment.get("brand")
            if text and text.strip():
                results.append(self.process_comment(text, platform, brand))
        return results