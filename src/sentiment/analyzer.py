import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from typing import List, Dict

try:
    from src.preprocessing.cleaner import preprocess_for_ml
except ImportError:
    # fallback
    def preprocess_for_ml(text):
        return str(text).lower()

class SentimentAnalyzer:
    def __init__(self, model_dir='models/sentiment'):
        self.model_dir = model_dir
        self.pipeline = None
        self.load(model_dir)

    def train(self, texts: List[str], labels: List[str]):
        if not texts or not labels or len(texts) != len(labels):
            raise ValueError("Texts and labels must be non-empty and of the same length.")
        
        vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2), sublinear_tf=True)
        classifier = LogisticRegression(max_iter=1000)
        
        self.pipeline = Pipeline([
            ('tfidf', vectorizer),
            ('clf', classifier)
        ])
        
        processed_texts = [preprocess_for_ml(text) for text in texts]
        self.pipeline.fit(processed_texts, labels)

    def predict(self, text: str) -> Dict:
        if not self.is_loaded():
            raise RuntimeError("Model is not loaded or trained.")
        
        processed_text = preprocess_for_ml(text)
        probs = self.pipeline.predict_proba([processed_text])[0]
        classes = self.pipeline.classes_
        
        best_idx = np.argmax(probs)
        return {
            'sentiment': str(classes[best_idx]),
            'confidence': float(probs[best_idx])
        }

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        if not self.is_loaded():
            raise RuntimeError("Model is not loaded or trained.")
        
        processed_texts = [preprocess_for_ml(text) for text in texts]
        probs = self.pipeline.predict_proba(processed_texts)
        classes = self.pipeline.classes_
        
        results = []
        for p in probs:
            best_idx = np.argmax(p)
            results.append({
                'sentiment': str(classes[best_idx]),
                'confidence': float(p[best_idx])
            })
        return results

    def save(self, model_dir=None):
        if not self.is_loaded():
            raise RuntimeError("Cannot save an untrained model.")
        
        save_dir = model_dir or self.model_dir
        os.makedirs(save_dir, exist_ok=True)
        
        model_path = os.path.join(save_dir, 'pipeline.joblib')
        joblib.dump(self.pipeline, model_path)

    def load(self, model_dir=None):
        load_dir = model_dir or self.model_dir
        model_path = os.path.join(load_dir, 'pipeline.joblib')
        
        if os.path.exists(model_path):
            self.pipeline = joblib.load(model_path)
            self.model_dir = load_dir

    def is_loaded(self) -> bool:
        return self.pipeline is not None
