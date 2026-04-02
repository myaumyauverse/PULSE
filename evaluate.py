import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_roc_auc(y_true: pd.Series, y_prob: pd.Series) -> float | None:
    if y_true.nunique() < 2:
        return None
    return float(roc_auc_score(y_true, y_prob))


def _scores(model, x: pd.DataFrame, y: pd.Series) -> dict:
    y_pred = model.predict(x)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(x)[:, 1]
    else:
        y_prob = y_pred

    return {
        "accuracy": accuracy_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y, y_pred),
        "roc_auc": _safe_roc_auc(y, y_prob),
        "cm": confusion_matrix(y, y_pred),
    }


def _ensemble_scores(models: dict, x: pd.DataFrame, y: pd.Series, threshold: float) -> dict:
    probabilities = []
    for model in models.values():
        if hasattr(model, "predict_proba"):
            probabilities.append(pd.Series(model.predict_proba(x)[:, 1], index=x.index))
        else:
            probabilities.append(pd.Series(model.predict(x), index=x.index).astype(float))

    ensemble_prob = probabilities[0].copy()
    for series in probabilities[1:]:
        ensemble_prob = ensemble_prob.add(series, fill_value=0.0)
    ensemble_prob = ensemble_prob / len(probabilities)

    y_pred = (ensemble_prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y, y_pred),
        "roc_auc": _safe_roc_auc(y, ensemble_prob),
        "cm": confusion_matrix(y, y_pred),
    }


def evaluate_models(data_path: Path, models_dir: Path, test_size: float) -> None:
    """Load trained models and evaluate on labeled data."""
    df = pd.read_csv(data_path)

    summary_path = models_dir / "training_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        feature_columns = summary.get("feature_columns", [])
        threshold = float(summary.get("calibrated_threshold", 0.5))
        split_indices = summary.get("split_indices", {})
        split_type = summary.get("split_type", "chronological")
    else:
        feature_columns = [
            col
            for col in df.select_dtypes(include=["number"]).columns
            if col != "spike"
        ]
        threshold = 0.5
        split_indices = {}
        split_type = "chronological"

    if split_indices and split_indices.get("train") and split_indices.get("test"):
        train_idx = [idx for idx in split_indices["train"] if idx in df.index]
        test_idx = [idx for idx in split_indices["test"] if idx in df.index]
        if not train_idx or not test_idx:
            raise ValueError("PULSE: Stored split indices are incompatible with current dataset.")
        train_df = df.loc[train_idx].copy()
        test_df = df.loc[test_idx].copy()
    else:
        split_index = int(len(df) * (1.0 - test_size))
        if split_index <= 0 or split_index >= len(df):
            raise ValueError("PULSE: Invalid split index for evaluation. Collect more data.")
        train_df = df.iloc[:split_index].copy()
        test_df = df.iloc[split_index:].copy()

    x_train = train_df[feature_columns]
    y_train = train_df["spike"]
    x_test = test_df[feature_columns]
    y_test = test_df["spike"]

    models = {
        "Logistic Regression": models_dir / "logistic_regression.pkl",
        "Random Forest": models_dir / "random_forest.pkl",
        "SVM": models_dir / "svm.pkl",
    }
    loaded_models = {}

    print("=" * 60)
    print("PULSE: Model Evaluation Results")
    print(f"PULSE: Split strategy = {split_type}")
    print("=" * 60)

    for model_name, model_path in models.items():
        if not model_path.exists():
            print(f"PULSE: ⚠️ {model_name} model not found at {model_path}")
            continue

        model = joblib.load(model_path)
        loaded_models[model_name] = model
        train_scores = _scores(model, x_train, y_train)
        test_scores = _scores(model, x_test, y_test)

        print(f"\n{model_name}:")
        print("  Train Metrics:")
        print(f"    Accuracy:  {train_scores['accuracy']:.4f}")
        print(f"    Precision: {train_scores['precision']:.4f}")
        print(f"    Recall:    {train_scores['recall']:.4f}")
        print(f"    F1:        {train_scores['f1']:.4f}")
        print(f"    Balanced Acc: {train_scores['balanced_accuracy']:.4f}")
        print(f"    MCC:       {train_scores['mcc']:.4f}")
        train_auc = train_scores["roc_auc"]
        print(f"    ROC-AUC:   {train_auc:.4f}" if train_auc is not None else "    ROC-AUC:   n/a")

        print("  Test Metrics (held-out):")
        print(f"    Accuracy:  {test_scores['accuracy']:.4f}")
        print(f"    Precision: {test_scores['precision']:.4f}")
        print(f"    Recall:    {test_scores['recall']:.4f}")
        print(f"    F1:        {test_scores['f1']:.4f}")
        print(f"    Balanced Acc: {test_scores['balanced_accuracy']:.4f}")
        print(f"    MCC:       {test_scores['mcc']:.4f}")
        test_auc = test_scores["roc_auc"]
        print(f"    ROC-AUC:   {test_auc:.4f}" if test_auc is not None else "    ROC-AUC:   n/a")
        print(f"    Confusion Matrix:\n{test_scores['cm']}")

        gap = train_scores["f1"] - test_scores["f1"]
        print(f"  F1 Gap (train-test): {gap:.4f}")

    if loaded_models:
        ensemble_train = _ensemble_scores(loaded_models, x_train, y_train, threshold)
        ensemble_test = _ensemble_scores(loaded_models, x_test, y_test, threshold)
        print("\n  Calibrated Ensemble Metrics:")
        print(f"    Threshold: {threshold:.2f}")
        print(f"    Train F1: {ensemble_train['f1']:.4f}")
        print(f"    Test F1:  {ensemble_test['f1']:.4f}")
        print(f"    Test Balanced Acc: {ensemble_test['balanced_accuracy']:.4f}")
        print(f"    Test MCC: {ensemble_test['mcc']:.4f}")
        print(f"    Test ROC-AUC: {ensemble_test['roc_auc']:.4f}" if ensemble_test['roc_auc'] is not None else "    Test ROC-AUC: n/a")

    print("\n" + "=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PULSE model evaluation")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/system_metrics_labeled.csv"),
        help="Labeled data CSV path",
    )
    parser.add_argument(
        "--models",
        type=Path,
        default=Path("models"),
        help="Models directory",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of latest rows as held-out test set (used if no summary file)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_models(args.data, args.models, args.test_size)
