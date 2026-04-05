import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import train_test_split


def load_artifacts(models_dir: Path):
    lr = joblib.load(models_dir / "logistic_regression.pkl")
    rf = joblib.load(models_dir / "random_forest.pkl")
    svm = joblib.load(models_dir / "svm.pkl")
    with (models_dir / "training_summary.json").open("r", encoding="utf-8") as f:
        summary = json.load(f)
    return lr, rf, svm, summary


def get_probabilities(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=float)
        min_s, max_s = scores.min(), scores.max()
        if max_s - min_s == 0:
            return np.full_like(scores, 0.5, dtype=float)
        return (scores - min_s) / (max_s - min_s)

    return np.asarray(model.predict(X), dtype=float)


def resolve_features(df: pd.DataFrame, target_col: str, summary: dict) -> list[str]:
    feature_cols = summary.get("feature_columns")
    if not feature_cols:
        feature_cols = [c for c in df.columns if c not in [target_col, "timestamp"]]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in dataset: {missing}")

    return feature_cols


def resolve_target_column(df: pd.DataFrame, target_col: str, summary: dict) -> str:
    if target_col in df.columns:
        return target_col

    summary_target = summary.get("target_column")
    if summary_target and summary_target in df.columns:
        print(
            f"[info] target column '{target_col}' not found; using summary target '{summary_target}'."
        )
        return summary_target

    candidates = ["spike_forecast", "spike", "spike_label", "label", "target"]
    for cand in candidates:
        if cand in df.columns:
            print(
                f"[info] target column '{target_col}' not found; using detected target '{cand}'."
            )
            return cand

    raise ValueError(
        f"Target column '{target_col}' not found in {list(df.columns)} and no fallback target detected."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ROC and 2x2 confusion matrix plots for PULSE models."
    )
    parser.add_argument("--data", default="data/system_metrics_labeled.csv")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--target-col", default="spike")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()

    data_path = Path(args.data)
    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    lr, rf, svm, summary = load_artifacts(models_dir)

    target_col = resolve_target_column(df, args.target_col, summary)

    feature_cols = resolve_features(df, target_col, summary)
    X = df[feature_cols].copy()
    y = df[target_col].astype(int).copy()

    split_type = summary.get("split_type", "stratified")
    stratify = y if split_type == "stratified" else None

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify,
    )

    probs_lr = get_probabilities(lr, X_test)
    probs_rf = get_probabilities(rf, X_test)
    probs_svm = get_probabilities(svm, X_test)
    probs_ens = (probs_lr + probs_rf + probs_svm) / 3.0

    threshold = float(summary.get("calibrated_threshold", 0.5))

    # ROC curves
    roc_path = output_dir / "roc_curves.png"
    plt.figure(figsize=(9, 7))
    for name, probs in [
        ("Logistic Regression", probs_lr),
        ("Random Forest", probs_rf),
        ("SVM", probs_svm),
        ("Ensemble", probs_ens),
    ]:
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("PULSE ROC Curves")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(roc_path, dpi=160)
    plt.close()

    # 2x2 confusion matrices
    yhat_lr = (probs_lr >= threshold).astype(int)
    yhat_rf = (probs_rf >= threshold).astype(int)
    yhat_svm = (probs_svm >= threshold).astype(int)
    yhat_ens = (probs_ens >= threshold).astype(int)

    cm_items = [
        ("Logistic Regression", yhat_lr),
        ("Random Forest", yhat_rf),
        ("SVM", yhat_svm),
        ("Ensemble", yhat_ens),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.ravel()

    for ax, (name, yhat) in zip(axes, cm_items):
        cm = confusion_matrix(y_test, yhat, labels=[0, 1])
        disp = ConfusionMatrixDisplay(cm, display_labels=["Stable", "Spike"])
        disp.plot(ax=ax, values_format="d", colorbar=False)
        ax.set_title(name)

    fig.suptitle(f"PULSE Confusion Matrices (Threshold={threshold:.2f})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    cm_grid_path = output_dir / "confusion_matrices_2x2.png"
    fig.savefig(cm_grid_path, dpi=160)
    plt.close(fig)

    print("Saved plots:")
    print(f"- {roc_path}")
    print(f"- {cm_grid_path}")

    print("\nClassification reports (threshold-applied):")
    for name, yhat in cm_items:
        print(f"\n{name}")
        print(classification_report(y_test, yhat, target_names=["Stable", "Spike"], digits=4))


if __name__ == "__main__":
    main()
