import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.sentiment.analyzer import SentimentAnalyzer

def main():
    data_path = 'data/synthetic/sentiment_train.csv'
    model_dir = 'models/sentiment'
    metrics_path = os.path.join(model_dir, 'metrics.json')

    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found.")
        return

    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"Error reading data: {e}")
        return

    if 'text' not in df.columns or 'sentiment' not in df.columns:
        print("Error: Dataset must contain 'text' and 'sentiment' columns.")
        return

    df.dropna(subset=['text', 'sentiment'], inplace=True)

    X = df['text'].astype(str).tolist()
    y = df['sentiment'].astype(str).tolist()

    if len(X) == 0:
        print("Error: Dataset is empty after dropping nulls.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    analyzer = SentimentAnalyzer()
    
    print("Training sentiment analyzer...")
    analyzer.train(X_train, y_train)

    print("Evaluating model...")
    preds_dicts = analyzer.predict_batch(X_test)
    y_pred = [d['sentiment'] for d in preds_dicts]

    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    precision_none = precision_score(y_test, y_pred, average=None, zero_division=0)
    recall_none = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1_none = f1_score(y_test, y_pred, average=None, zero_division=0)
    
    cm = confusion_matrix(y_test, y_pred)
    classes = analyzer.pipeline.classes_

    metrics = {
        'accuracy': accuracy,
        'macro_avg': {
            'precision': precision_macro,
            'recall': recall_macro,
            'f1_score': f1_macro
        },
        'per_class': {
            str(cls): {
                'precision': float(precision_none[i]),
                'recall': float(recall_none[i]),
                'f1_score': float(f1_none[i])
            } for i, cls in enumerate(classes)
        },
        'confusion_matrix': cm.tolist(),
        'classes': [str(c) for c in classes]
    }

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {f1_macro:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=classes, zero_division=0))

    os.makedirs(model_dir, exist_ok=True)
    analyzer.save(model_dir)
    print(f"Model saved to {model_dir}")

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")

if __name__ == '__main__':
    main()
