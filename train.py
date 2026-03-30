import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def prepare_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the training set has both classes so all models can train."""
    if df["spike"].nunique() >= 2:
        return df

    print("PULSE: No spike=1 rows found. Creating synthetic spike rows for demo training.")
    top_rows = df.sort_values("cpu_percent", ascending=False).head(min(10, len(df))).copy()
    top_rows["cpu_percent"] = top_rows["cpu_percent"].clip(lower=85)
    top_rows["spike"] = 1
    return pd.concat([df, top_rows], ignore_index=True)


def train_models(input_path: Path, model_dir: Path) -> None:
    df = pd.read_csv(input_path)

    required = ["cpu_percent", "ram_percent", "process_count", "spike"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"PULSE: Missing required columns: {missing}")

    df = prepare_training_data(df)

    x = df[["cpu_percent", "ram_percent", "process_count"]]
    y = df["spike"]

    models = {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "svm": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf", probability=True)),
            ]
        ),
    }

    model_dir.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        model.fit(x, y)
        model_path = model_dir / f"{name}.pkl"
        joblib.dump(model, model_path)
        print(f"PULSE: Trained and saved {name} to {model_path}")

    print("PULSE: Model training phase complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PULSE model training")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/system_metrics_labeled.csv"),
        help="Input labeled CSV path",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models"),
        help="Directory to save trained models",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_models(input_path=args.input, model_dir=args.model_dir)


def _ensure_two_classes(df: pd.DataFrame) -> pd.DataFrame:
    """Create a few synthetic spike rows if data contains only one class.

    This keeps the project runnable for beginners when their first dataset is too small.
    """
    if df[TARGET_COLUMN].nunique() > 1:
        return df

    top_rows = df.sort_values("cpu_percent", ascending=False).head(min(5, len(df))).copy()
    top_rows["cpu_percent"] = top_rows["cpu_percent"].clip(lower=85)
    top_rows[TARGET_COLUMN] = 1
    out = pd.concat([df, top_rows], ignore_index=True)

    print("PULSE: Only one label class found. Added demo spike rows for training.")
    return out


def train_models(input_path: Path, models_dir: Path) -> None:
    df = pd.read_csv(input_path)

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"PULSE: Missing required columns for training: {missing}")

    df = _ensure_two_classes(df)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        "random_forest": RandomForestClassifier(n_estimators=150, random_state=42),
        "svm": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf", probability=True, random_state=42)),
            ]
        ),
    }

    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows = []

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        out_path = models_dir / f"{model_name}.pkl"
        joblib.dump(model, out_path)

        metrics_rows.append({"model": model_name, "accuracy": round(float(accuracy), 4)})
        print(f"PULSE: Trained {model_name} | accuracy={accuracy:.4f} | saved={out_path}")

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = models_dir / "training_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"PULSE: Training summary saved to {metrics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PULSE model training")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/system_metrics_labeled.csv"),
        help="Path to labeled training data",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directory to save trained models",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_models(input_path=args.input, models_dir=args.models_dir)