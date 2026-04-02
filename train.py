import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


TARGET_COLUMN = "spike"


def _select_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"timestamp", TARGET_COLUMN}
    numeric_cols = list(df.select_dtypes(include=["number"]).columns)
    return [col for col in numeric_cols if col not in exclude]


def _ensemble_probability(model_map: dict, x_frame: pd.DataFrame) -> pd.Series:
    probabilities = []
    for model in model_map.values():
        if hasattr(model, "predict_proba"):
            probabilities.append(pd.Series(model.predict_proba(x_frame)[:, 1], index=x_frame.index))
        else:
            probabilities.append(pd.Series(model.predict(x_frame), index=x_frame.index).astype(float))

    if not probabilities:
        raise ValueError("PULSE: No trained models available for calibration.")

    combined = probabilities[0].copy()
    for series in probabilities[1:]:
        combined = combined.add(series, fill_value=0.0)
    return combined / len(probabilities)


def _calibrate_threshold(y_true: pd.Series, y_prob: pd.Series) -> dict:
    best_threshold = 0.5
    best_f1 = -1.0
    best_precision = 0.0
    best_recall = 0.0
    best_mcc = -1.0

    for threshold in [round(step / 100, 2) for step in range(10, 91)]:
        y_pred = (y_prob >= threshold).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)

        # Prefer thresholds that improve both-class separation (MCC), then F1.
        if (mcc > best_mcc) or (mcc == best_mcc and score > best_f1):
            best_f1 = score
            best_threshold = threshold
            best_precision = precision
            best_recall = recall
            best_mcc = mcc

    return {
        "threshold": best_threshold,
        "f1": best_f1,
        "precision": best_precision,
        "recall": best_recall,
        "mcc": best_mcc,
    }


def train_models(
    input_path: Path,
    model_dir: Path,
    test_size: float,
    horizon_steps: int,
    split_type: str,
    random_state: int,
) -> None:
    df = pd.read_csv(input_path)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"PULSE: Missing required target column: {TARGET_COLUMN}")

    feature_columns = _select_feature_columns(df)
    if not feature_columns:
        raise ValueError("PULSE: No numeric feature columns found for training.")

    if not 0.0 < test_size < 0.5:
        raise ValueError("PULSE: test_size must be between 0 and 0.5")

    if split_type == "chronological":
        split_index = int(len(df) * (1.0 - test_size))
        if split_index <= 0 or split_index >= len(df):
            raise ValueError("PULSE: Not enough rows for train/test split. Collect more data.")
        train_df = df.iloc[:split_index].copy()
        test_df = df.iloc[split_index:].copy()
    elif split_type == "stratified":
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df[TARGET_COLUMN],
            random_state=random_state,
        )
    else:
        raise ValueError(f"PULSE: Unsupported split type: {split_type}")

    if split_type == "stratified":
        fit_df, val_df = train_test_split(
            train_df,
            test_size=0.2,
            stratify=train_df[TARGET_COLUMN],
            random_state=random_state,
        )
    else:
        val_size = max(1, int(len(train_df) * 0.2))
        if len(train_df) <= val_size:
            raise ValueError("PULSE: Not enough rows left for validation. Collect more data.")
        fit_df = train_df.iloc[:-val_size].copy()
        val_df = train_df.iloc[-val_size:].copy()

    y_train = fit_df[TARGET_COLUMN]
    if y_train.nunique() < 2:
        raise ValueError(
            "PULSE: Training split has only one class. Collect more diverse data "
            "(idle + stress sessions) and preprocess again."
        )

    x_train = fit_df[feature_columns]
    y_train = fit_df[TARGET_COLUMN]
    x_val = val_df[feature_columns]
    y_val = val_df[TARGET_COLUMN]

    models = {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=42,
                        class_weight="balanced",
                        C=0.5,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
        ),
        "svm": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        probability=True,
                        random_state=42,
                        class_weight="balanced",
                        C=1.0,
                        gamma="scale",
                    ),
                ),
            ]
        ),
    }

    model_dir.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        model.fit(x_train, y_train)
        model_path = model_dir / f"{name}.pkl"
        joblib.dump(model, model_path)
        print(f"PULSE: Trained and saved {name} to {model_path}")

    calibrated_models = {}
    for name, model in models.items():
        calibrated_models[name] = model

    val_prob = _ensemble_probability(calibrated_models, x_val)
    calibration = _calibrate_threshold(y_val, val_prob)

    summary = {
        "target_column": TARGET_COLUMN,
        "feature_columns": feature_columns,
        "rows_total": int(len(df)),
        "rows_train": int(len(train_df)),
        "rows_fit": int(len(fit_df)),
        "rows_val": int(len(val_df)),
        "rows_test": int(len(test_df)),
        "test_size": float(test_size),
        "horizon_steps": int(horizon_steps),
        "split_type": split_type,
        "random_state": int(random_state),
        "split_indices": {
            "train": [int(i) for i in train_df.index.tolist()],
            "fit": [int(i) for i in fit_df.index.tolist()],
            "val": [int(i) for i in val_df.index.tolist()],
            "test": [int(i) for i in test_df.index.tolist()],
        },
        "class_counts": {
            "train": {str(k): int(v) for k, v in train_df[TARGET_COLUMN].value_counts().to_dict().items()},
            "fit": {str(k): int(v) for k, v in fit_df[TARGET_COLUMN].value_counts().to_dict().items()},
            "val": {str(k): int(v) for k, v in val_df[TARGET_COLUMN].value_counts().to_dict().items()},
            "test": {str(k): int(v) for k, v in test_df[TARGET_COLUMN].value_counts().to_dict().items()},
        },
        "calibrated_threshold": float(calibration["threshold"]),
        "calibration_f1": float(calibration["f1"]),
        "calibration_precision": float(calibration["precision"]),
        "calibration_recall": float(calibration["recall"]),
        "calibration_mcc": float(calibration["mcc"]),
    }
    summary_path = model_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"PULSE: Saved training summary to {summary_path}")
    print(
        "PULSE: Calibration complete "
        f"(threshold={calibration['threshold']:.2f}, F1={calibration['f1']:.4f}, "
        f"precision={calibration['precision']:.4f}, recall={calibration['recall']:.4f}, "
        f"MCC={calibration['mcc']:.4f})"
    )

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
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of latest rows reserved as held-out test split",
    )
    parser.add_argument(
        "--horizon-steps",
        type=int,
        default=5,
        help="Forecast horizon steps used in preprocessing (stored for inference metadata)",
    )
    parser.add_argument(
        "--split-type",
        choices=["stratified", "chronological"],
        default="stratified",
        help="Data split strategy for train/validation/test",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for stratified splits",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_models(
        input_path=args.input,
        model_dir=args.model_dir,
        test_size=args.test_size,
        horizon_steps=args.horizon_steps,
        split_type=args.split_type,
        random_state=args.random_state,
    )
