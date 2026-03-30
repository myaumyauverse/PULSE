import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


def evaluate_models(data_path: Path, models_dir: Path) -> None:
    """Load trained models and evaluate on labeled data."""
    df = pd.read_csv(data_path)

    X = df[["cpu_percent", "ram_percent", "process_count"]]
    y = df["spike"]

    models = {
        "Logistic Regression": models_dir / "logistic_regression.pkl",
        "Random Forest": models_dir / "random_forest.pkl",
        "SVM": models_dir / "svm.pkl",
    }

    print("=" * 60)
    print("PULSE: Model Evaluation Results")
    print("=" * 60)

    for model_name, model_path in models.items():
        if not model_path.exists():
            print(f"PULSE: ⚠️ {model_name} model not found at {model_path}")
            continue

        model = joblib.load(model_path)
        y_pred = model.predict(X)

        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        cm = confusion_matrix(y, y_pred)

        print(f"\n{model_name}:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  Confusion Matrix:\n{cm}")

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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_models(args.data, args.models)
