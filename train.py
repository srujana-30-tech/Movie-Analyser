"""
ReviewSense - Model Training & Evaluation Pipeline
=================================================
Senior ML Engineer: End-to-end sentiment classification pipeline.

Key Engineering Highlights:
1. Inspects raw dataset: shape, columns, missing values, duplicates, class distribution.
2. Cleans raw data and drops duplicates/nulls BEFORE train/test splitting.
3. Performs Stratified 80/20 train/test split with random_state=42.
4. Uses shared `clean_text()` from `preprocessing.py` on the data.
5. Fits TfidfVectorizer strictly on the training set to prevent data leakage.
6. Trains LogisticRegression (max_iter=1000) and MultinomialNB on identical features.
7. Evaluates both models strictly on the held-out test set (Accuracy, Precision, Recall, F1).
8. Automatically selects the champion model based on the actual test F1-score.
9. Exports model artifacts with joblib and full dashboard telemetry to `models/metrics.json`.
10. Renders dark-themed confusion matrix PNGs matching the "Projection Room" palette.
"""

import json
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Import the single source of truth for text normalization
from preprocessing import clean_text


def run_training_pipeline(dataset_path: str = "dataset/IMDB Dataset.csv", models_dir: str = "models"):
    print("=" * 80)
    print("🎬 REVIEWSENSE: ML TRAINING & EVALUATION PIPELINE")
    print("=" * 80)

    start_total_time = time.time()
    os.makedirs(models_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # STEP 1: Dataset Inspection
    # -------------------------------------------------------------------------
    print("\n[1/6] 📂 Inspecting Dataset...")
    if not os.path.exists(dataset_path):
        print(f"\n❌ ERROR: Dataset not found at '{dataset_path}'!")
        print("Please verify the file path and ensure 'IMDB Dataset.csv' is in 'dataset/'.")
        sys.exit(1)

    df_raw = pd.read_csv(dataset_path)
    raw_shape = df_raw.shape
    raw_null_count = int(df_raw.isnull().sum().sum())
    raw_duplicate_count = int(df_raw.duplicated().sum())

    print(f"  • Raw Dataset Shape:     {raw_shape[0]:,} rows, {raw_shape[1]} columns")
    print(f"  • Columns:               {list(df_raw.columns)}")
    print(f"  • Missing Values (Null): {raw_null_count:,}")
    print(f"  • Duplicate Rows:        {raw_duplicate_count:,}")
    print("  • Raw Sentiment Breakdown:")
    for sentiment, count in df_raw["sentiment"].value_counts().items():
        print(f"      - {sentiment}: {count:,} ({count / len(df_raw) * 100:.2f}%)")

    # -------------------------------------------------------------------------
    # STEP 2: Data Cleaning & Deduplication BEFORE Splitting
    # -------------------------------------------------------------------------
    print("\n[2/6] 🧹 Cleaning & Deduplicating Dataset...")
    df = df_raw.dropna().drop_duplicates().copy()
    clean_shape = df.shape
    print(f"  • Rows after deduplication: {clean_shape[0]:,} (Dropped {raw_shape[0] - clean_shape[0]:,} rows)")
    
    # Map sentiments to binary integers (positive=1, negative=0)
    sentiment_map = {"positive": 1, "negative": 0}
    df["label"] = df["sentiment"].map(lambda x: sentiment_map.get(str(x), 0))
    pos_count = int((df["label"] == 1).sum())
    neg_count = int((df["label"] == 0).sum())
    print(f"  • Class Balance: Positive={pos_count:,} ({pos_count/len(df)*100:.2f}%), Negative={neg_count:,} ({neg_count/len(df)*100:.2f}%)")

    # Clean reviews using shared clean_text
    print("  • Normalizing review text via shared clean_text() pipeline...")
    t0 = time.time()
    df["cleaned_review"] = df["review"].apply(clean_text)
    print(f"  • Text normalization completed in {time.time() - t0:.2f}s")

    # -------------------------------------------------------------------------
    # STEP 3: Stratified 80/20 Train-Test Split
    # -------------------------------------------------------------------------
    print("\n[3/6] ✂️ Performing Stratified 80/20 Train/Test Split (random_state=42)...")
    X_list = df["cleaned_review"].tolist()
    y_list = [int(v) for v in df["label"].tolist()]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_list,
        y_list,
        test_size=0.20,
        random_state=42,
        stratify=y_list
    )
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    print(f"  • Training set size: {len(X_train_raw):,} samples (Pos: {int((y_train == 1).sum()):,}, Neg: {int((y_train == 0).sum()):,})")
    print(f"  • Testing set size:  {len(X_test_raw):,} samples (Pos: {int((y_test == 1).sum()):,}, Neg: {int((y_test == 0).sum()):,})")

    # -------------------------------------------------------------------------
    # STEP 4: TF-IDF Vectorization (Fit on TRAIN ONLY to prevent Data Leakage)
    # -------------------------------------------------------------------------
    print("\n[4/6] 🔠 Fitting TfidfVectorizer on Training Set Only...")
    t0 = time.time()
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50000,
        min_df=2,
        sublinear_tf=True
    )
    X_train_tfidf = vectorizer.fit_transform(X_train_raw)
    X_test_tfidf = vectorizer.transform(X_test_raw)
    vocab_size = len(vectorizer.vocabulary_)
    train_shape = getattr(X_train_tfidf, "shape", (len(X_train_raw), vocab_size))
    test_shape = getattr(X_test_tfidf, "shape", (len(X_test_raw), vocab_size))
    print(f"  • Vocabulary size extracted: {vocab_size:,} features")
    print(f"  • Training matrix shape:    {train_shape}")
    print(f"  • Testing matrix shape:     {test_shape}")
    print(f"  • Vectorization finished in {time.time() - t0:.2f}s")

    # -------------------------------------------------------------------------
    # STEP 5: Train Models & Evaluate on Held-Out Test Set
    # -------------------------------------------------------------------------
    print("\n[5/6] 🧠 Training & Evaluating Sentiment Classifiers...")
    
    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "MultinomialNB": MultinomialNB(alpha=1.0)
    }

    eval_results = {}
    fitted_models = {}

    for name, clf in candidates.items():
        print(f"\n  ▶ Training {name}...")
        t_clf = time.time()
        clf.fit(X_train_tfidf, y_train)
        train_time = time.time() - t_clf
        fitted_models[name] = clf

        # Predictions on held-out test set
        y_pred = clf.predict(X_test_tfidf)
        
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average="binary"))
        rec = float(recall_score(y_test, y_pred, average="binary"))
        f1 = float(f1_score(y_test, y_pred, average="binary"))
        cm = confusion_matrix(y_test, y_pred).tolist()
        report_dict = classification_report(y_test, y_pred, target_names=["Negative", "Positive"], output_dict=True)

        eval_results[name] = {
            "name": name,
            "training_time_sec": round(train_time, 2),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": cm,
            "classification_report": report_dict
        }

        print(f"    ✔ {name} Results on Held-out Test Set:")
        print(f"       - Accuracy:  {acc * 100:.2f}%")
        print(f"       - Precision: {prec * 100:.2f}%")
        print(f"       - Recall:    {rec * 100:.2f}%")
        print(f"       - F1-Score:  {f1 * 100:.2f}%")
        print(f"       - Trained in: {train_time:.2f}s")

    # -------------------------------------------------------------------------
    # STEP 6: Champion Selection & Artifact Export
    # -------------------------------------------------------------------------
    print("\n[6/6] 🏆 Selecting Champion Model & Saving Artifacts...")
    
    # Pick best model strictly based on held-out test F1-score
    best_model_name = max(eval_results.keys(), key=lambda k: eval_results[k]["f1_score"])
    best_model_obj = fitted_models[best_model_name]
    best_metrics = eval_results[best_model_name]

    print(f"  ★ CHAMPION MODEL: {best_model_name} (Test F1: {best_metrics['f1_score']*100:.2f}%, Accuracy: {best_metrics['accuracy']*100:.2f}%)")

    # Save best model and vectorizer
    model_save_path = os.path.join(models_dir, "best_sentiment_model.pkl")
    vectorizer_save_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")

    joblib.dump(best_model_obj, model_save_path, compress=3)
    joblib.dump(vectorizer, vectorizer_save_path, compress=3)
    print(f"  • Saved Champion Model to:  {model_save_path} ({os.path.getsize(model_save_path)/(1024*1024):.2f} MB)")
    print(f"  • Saved TF-IDF Vectorizer to: {vectorizer_save_path} ({os.path.getsize(vectorizer_save_path)/(1024*1024):.2f} MB)")

    # Generate and save styled confusion matrix PNGs
    from matplotlib.colors import LinearSegmentedColormap
    amber_cmap = LinearSegmentedColormap.from_list("projection_amber", ["#14141B", "#3D2B0F", "#996300", "#FFB020", "#FFE4A0"])

    for name, res in eval_results.items():
        cm = np.array(res["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(6, 5), facecolor="#0B0B0F")
        ax.set_facecolor("#14141B")

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=amber_cmap,
            cbar=False,
            ax=ax,
            annot_kws={"size": 14, "weight": "bold", "color": "#F4EFE6"},
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
            linewidths=1.5,
            linecolor="#1C1C26"
        )
        ax.set_title(f"{name} Confusion Matrix", fontsize=14, color="#FFB020", pad=15, weight="bold")
        ax.set_xlabel("Predicted Label", fontsize=11, color="#9A97A3", labelpad=10)
        ax.set_ylabel("True Label", fontsize=11, color="#9A97A3", labelpad=10)
        ax.tick_params(colors="#F4EFE6", labelsize=10)

        # Subtle spine styling
        for spine in ax.spines.values():
            spine.set_edgecolor("#333344")

        cm_filename = f"cm_{name.lower()}.png"
        cm_path = os.path.join(models_dir, cm_filename)
        plt.tight_layout()
        plt.savefig(cm_path, dpi=200, facecolor="#0B0B0F", edgecolor="none")
        plt.close(fig)
        print(f"  • Saved Confusion Matrix plot: {cm_path}")

    # Export complete telemetry to models/metrics.json
    telemetry = {
        "dataset_stats": {
            "dataset_filename": os.path.basename(dataset_path),
            "total_raw": raw_shape[0],
            "duplicates_removed": raw_duplicate_count,
            "nulls_removed": raw_null_count,
            "total_clean": clean_shape[0],
            "train_count": len(X_train_raw),
            "test_count": len(X_test_raw),
            "positive_count": pos_count,
            "negative_count": neg_count,
            "positive_pct": round(pos_count / clean_shape[0] * 100, 2),
            "negative_pct": round(neg_count / clean_shape[0] * 100, 2)
        },
        "tfidf_config": {
            "ngram_range": [1, 2],
            "max_features": 50000,
            "min_df": 2,
            "sublinear_tf": True,
            "features_extracted": vocab_size
        },
        "models": eval_results,
        "best_model": {
            "name": best_model_name,
            "criterion": "F1-Score on held-out test set",
            "accuracy": best_metrics["accuracy"],
            "f1_score": best_metrics["f1_score"],
            "precision": best_metrics["precision"],
            "recall": best_metrics["recall"],
            "model_filename": "best_sentiment_model.pkl",
            "vectorizer_filename": "tfidf_vectorizer.pkl"
        },
        "pipeline_metadata": {
            "split_ratio": "80/20 Stratified",
            "random_state": 42,
            "completed_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_runtime_seconds": round(time.time() - start_total_time, 2)
        }
    }

    metrics_json_path = os.path.join(models_dir, "metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=4)
    print(f"  • Exported Dashboard Telemetry to: {metrics_json_path}")

    print("\n" + "=" * 80)
    print(f"🎉 PIPELINE COMPLETED IN {time.time() - start_total_time:.2f} SECONDS")
    print("=" * 80)
    return telemetry


if __name__ == "__main__":
    run_training_pipeline()
