import argparse
from pathlib import Path

import joblib
import psutil


def get_current_metrics() -> dict:
    """Capture current system metrics."""
    return {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "ram_percent": psutil.virtual_memory().percent,
        "process_count": len(psutil.pids()),
    }


def predict_spike(models_dir: Path, num_predictions: int = 5) -> None:
    """Load models and predict spike probability on live metrics."""
    models = {
        "Logistic Regression": models_dir / "logistic_regression.pkl",
        "Random Forest": models_dir / "random_forest.pkl",
        "SVM": models_dir / "svm.pkl",
    }

    print("=" * 70)
    print("PULSE: Real-Time Spike Prediction System")
    print("=" * 70)

    for i in range(num_predictions):
        metrics = get_current_metrics()
        cpu = metrics["cpu_percent"]
        ram = metrics["ram_percent"]
        procs = metrics["process_count"]

        print(f"\n[Sample {i + 1}] CPU={cpu:.1f}% | RAM={ram:.1f}% | Processes={procs}")

        predictions = []
        for model_name, model_path in models.items():
            if not model_path.exists():
                print(f"  ⚠️ {model_name} not found")
                continue

            model = joblib.load(model_path)
            X = [[cpu, ram, procs]]
            pred = model.predict(X)[0]
            predictions.append(pred)

        if predictions:
            avg_pred = sum(predictions) / len(predictions)
            if avg_pred > 0.5:
                print("  🚨 PULSE Alert: HIGH chance of system spike incoming!")
            else:
                print("  ✅ PULSE Status: System stable")

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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict_spike(models_dir=args.models, num_predictions=args.samples)
