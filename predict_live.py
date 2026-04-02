import argparse
import json
from pathlib import Path

import joblib
import psutil

from feature_engineering import BASE_FEATURE_COLUMNS, build_live_feature_frame, parse_window_sizes


def get_current_metrics() -> dict:
    """Capture current system metrics."""
    return {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "ram_percent": psutil.virtual_memory().percent,
        "process_count": len(psutil.pids()),
    }


def _load_training_summary(models_dir: Path) -> dict:
    summary_path = models_dir / "training_summary.json"
    if not summary_path.exists():
        return {
            "feature_columns": BASE_FEATURE_COLUMNS,
            "horizon_steps": 0,
            "calibrated_threshold": 0.5,
        }
    return json.loads(summary_path.read_text(encoding="utf-8"))


def predict_spike(models_dir: Path, num_predictions: int = 5, window_sizes: list[int] | None = None) -> None:
    """Load models and predict near-future spike probability on live metrics."""
    if window_sizes is None:
        window_sizes = [3, 5]

    models = {
        "Logistic Regression": models_dir / "logistic_regression.pkl",
        "Random Forest": models_dir / "random_forest.pkl",
        "SVM": models_dir / "svm.pkl",
    }
    summary = _load_training_summary(models_dir)
    feature_columns = summary.get("feature_columns", BASE_FEATURE_COLUMNS)
    horizon_steps = int(summary.get("horizon_steps", 0))
    calibrated_threshold = float(summary.get("calibrated_threshold", 0.5))
    history_rows: list[dict] = []

    print("=" * 70)
    print("PULSE: Real-Time Forecast Spike Prediction System")
    print("=" * 70)

    for i in range(num_predictions):
        metrics = get_current_metrics()
        history_rows.append(metrics)
        cpu = metrics["cpu_percent"]
        ram = metrics["ram_percent"]
        procs = metrics["process_count"]

        print(f"\n[Sample {i + 1}] CPU={cpu:.1f}% | RAM={ram:.1f}% | Processes={procs}")

        x_live = build_live_feature_frame(
            history_rows=history_rows,
            feature_columns=feature_columns,
            window_sizes=window_sizes,
        )
        if x_live is None:
            print("  ℹ️ Warming up live feature window...")
            continue

        probabilities = []
        for model_name, model_path in models.items():
            if not model_path.exists():
                print(f"  ⚠️ {model_name} not found")
                continue

            model = joblib.load(model_path)
            if hasattr(model, "predict_proba"):
                spike_prob = float(model.predict_proba(x_live)[0][1])
            else:
                spike_prob = float(model.predict(x_live)[0])
            probabilities.append(spike_prob)
            print(f"  {model_name}: spike_prob={spike_prob:.2f}")

        if probabilities:
            ensemble_score = sum(probabilities) / len(probabilities)
            if ensemble_score >= calibrated_threshold:
                if horizon_steps > 0:
                    print(f"  🚨 PULSE Alert: High spike chance within next {horizon_steps} steps")
                else:
                    print("  🚨 PULSE Alert: High spike chance")
            else:
                print("  ✅ PULSE Status: Near-future forecast stable")
            print(f"  Calibrated threshold = {calibrated_threshold:.2f}")

    print("\n" + "=" * 70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PULSE real-time spike prediction")
    parser.add_argument(
        "--models",
        type=Path,
        default=Path("models"),
        help="Models directory",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of predictions to run",
    )
    parser.add_argument(
        "--window-sizes",
        type=str,
        default="3,5",
        help="Rolling window sizes for live temporal features, comma-separated",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict_spike(
        models_dir=args.models,
        num_predictions=args.samples,
        window_sizes=parse_window_sizes(args.window_sizes),
    )
