import os
import sys
import json
import pandas as pd
from collections import Counter
from sklearn.metrics import silhouette_score

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.clustering.clusterer import ComplaintClusterer
from src.preprocessing.cleaner import preprocess_for_ml


def main():
    data_path = "data/synthetic/comments.csv"
    if not os.path.exists(data_path):
        print(f"Dataset {data_path} not found.")
        return

    df = pd.read_csv(data_path)
    if "sentiment" not in df.columns or "text" not in df.columns:
        print("Missing required columns in dataset.")
        return

    neg_df = df[df["sentiment"] == "negative"].copy()
    if neg_df.empty:
        print("No negative comments found.")
        return

    raw_texts = neg_df["text"].dropna().tolist()
    print(f"Loaded {len(raw_texts)} negative comments for clustering.")

    best_k = 6
    best_score = -1.0
    best_clusterer = None

    for k in range(3, 11):
        clusterer = ComplaintClusterer()
        clusterer.train(raw_texts, n_clusters=k)

        processed = [preprocess_for_ml(t) for t in raw_texts]
        X = clusterer.vectorizer.transform(processed)
        score = float(silhouette_score(X, clusterer.kmeans.labels_))
        print(f"k={k}, silhouette score={score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k
            best_clusterer = clusterer

    print(f"\nOptimal k: {best_k} with silhouette score: {best_score:.4f}")

    labels = best_clusterer.kmeans.labels_
    dist = Counter(labels)
    print("\nCluster Distribution & Identified Themes:")
    for cluster_id, count in dist.items():
        cid_str = str(cluster_id)
        label = best_clusterer.cluster_labels.get(cid_str, cid_str)
        theme = best_clusterer.cluster_themes.get(cid_str, "Unknown")
        print(f"Cluster {cluster_id} [{theme}] (Keywords: {label}): {count} samples")

    metrics = {
        "optimal_k": best_k,
        "silhouette_score": round(best_score, 4),
        "cluster_distribution": {str(k): v for k, v in dist.items()},
        "cluster_labels": best_clusterer.cluster_labels,
        "cluster_themes": best_clusterer.cluster_themes
    }

    model_dir = "models/clustering"
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(model_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    best_clusterer.save(model_dir)
    print(f"\nModel and metrics successfully saved to {model_dir}")


if __name__ == "__main__":
    main()
