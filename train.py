import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


FEATURE_COLUMNS = ["cpu_percent", "ram_percent", "process_count"]
TARGET_COLUMN = "spike"


def prepare_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the training set has both classes so all models can train."""
    if df[TARGET_COLUMN].nunique() >= 2:
        return df

    print("PULSE: No spike=1 rows found. Creating synthetic spike rows for demo training.")
    top_rows = df.sort_values("cpu_percent", ascending=False).head(min(10, len(df))).copy()
    top_rows["cpu_percent"] = top_rows["cpu_percent"].clip(lower=85)
    top_rows[TARGET_COLUMN] = 1
    return pd.concat([df, top_rows], ignore_index=True)


def train_models(input_path: Path, model_dir: Path) -> None:
    df = pd.read_csv(input_path)

    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"PULSE: Missing required columns: {missing}")

    df = prepare_training_data(df)

    x = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

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
                ("model", SVC(kernel="rbf", probability=True, random_state=42)),
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
