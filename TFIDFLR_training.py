from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from nltk.corpus import stopwords


# ---------------------------------------------------------------------------#
def find_split_files(data_root: Path, ds_id: int) -> Dict[str, Path]:
    """Return {'train': Path, 'test': Path} for dataset <ds_id>."""
    train_files = list(data_root.glob(f"dataset_{ds_id}_*_train.csv"))
    test_files = list(data_root.glob(f"dataset_{ds_id}_*_test.csv"))
    if len(train_files) != 1 or len(test_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one train/test split for dataset {ds_id} in '{data_root}'."
        )
    return {"train": train_files[0], "test": test_files[0]}


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    p_micro, r_micro, f_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": acc,
        "precision_micro": p_micro, "recall_micro": r_micro, "f1_micro": f_micro,
        "precision_macro": p_macro, "recall_macro": r_macro, "f1_macro": f_macro,
    }


# ---------------------------------------------------------------------------#
def train_dataset(
    ds_id: int,
    data_root: Path,
    output_root: Path,
    max_features: int = 50_000,
) -> None:
    print(f"\n=== DATASET {ds_id}  (TF-IDF + LR) ===")

    paths = find_split_files(data_root, ds_id)

    # 1) Load data ----------------------------------------------------------
    train_df = pd.read_csv(paths["train"])
    test_df = pd.read_csv(paths["test"])

    # Replace placeholder for commas (keeps parity with other scripts)
    train_df["text"] = train_df["text"].str.replace("\\comma", ",")
    test_df["text"] = test_df["text"].str.replace("\\comma", ",")

    # 2) Vectorisation ------------------------------------------------------
    german_stopwords = stopwords.words("german")

    vectoriser = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 3),       # unigrams + bigrams + trigrams
        stop_words=german_stopwords,
        lowercase=True,
        strip_accents="unicode",
        sublinear_tf=True,        # sublinear TF scaling
        min_df=3,                 # discard very rare terms
        max_df=0.85,              # ignore extremely common terms
        norm="l2"                 # L2 normalization
    )
    X_train = vectoriser.fit_transform(train_df["text"])
    X_test = vectoriser.transform(test_df["text"])

    y_train = train_df["label"].astype(int).to_numpy()
    y_test = test_df["label"].astype(int).to_numpy()

    # 3) Classifier ---------------------------------------------------------
    clf = LogisticRegression(
        solver="liblinear",       # robust solver for small/medium data
        penalty="l2",             # L2 regularization
        C=1.0,                    # default inverse regularization strength
        class_weight="balanced",  # handle class imbalance
        max_iter=1000,            # iterations for convergence
        multi_class="ovr",        # one-vs-rest approach
        n_jobs=-1,                # use all CPU cores
    )
    clf.fit(X_train, y_train)

    # 4) Evaluation ---------------------------------------------------------
    y_pred = clf.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        labels=list(range(5)),
        target_names=[str(i) for i in range(5)],
        digits=4,
        zero_division=0,
    )
    print(report)

    # 5) Persist artefacts --------------------------------------------------
    out_dir = output_root / f"dataset_{ds_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(vectoriser, out_dir / "tfidf_vectoriser.joblib")
    joblib.dump(clf, out_dir / "logreg_model.joblib")

    (out_dir / "classification_report.txt").write_text(report)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved artefacts to '{out_dir}'.")
    print("\n".join(f"  {k}: {v:.4f}" for k, v in metrics.items()))


# ---------------------------------------------------------------------------#
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data",
                        help="Directory containing dataset_*/ CSVs")
    parser.add_argument("--output_root", type=str, default="tfidf_lr_outputs",
                        help="Where to store fitted models and reports")
    parser.add_argument("--max_features", type=int, default=100_000,
                        help="Maximum vocabulary size for the TF-IDF vectoriser")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    for ds_id in (1, 2, 3, 4):
        train_dataset(ds_id, data_root, output_root, args.max_features)

    print("\nAll datasets processed.")


if __name__ == "__main__":
    main()