import argparse
from pathlib import Path

import pandas as pd

from feature_engineering import BASE_FEATURE_COLUMNS, parse_window_sizes, prepare_forecast_dataframe


def preprocess_data(
    input_path: Path,
    output_path: Path,
    label_mode: str,
    cpu_threshold: float,
    horizon_steps: int,
    window_sizes: list[int],
) -> None:
    df = pd.read_csv(input_path)

    required_columns = ["timestamp", *BASE_FEATURE_COLUMNS]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"PULSE: Missing required columns: {missing}")

    if label_mode == "simple":
        df["spike"] = (df["cpu_percent"] > cpu_threshold).astype(int)
    elif label_mode == "forecast":
        df = prepare_forecast_dataframe(
            df=df,
            horizon_steps=horizon_steps,
            cpu_threshold=cpu_threshold,
            window_sizes=window_sizes,
        )
    else:
        raise ValueError(f"PULSE: Unsupported label mode: {label_mode}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    spike_count = int(df["spike"].sum())
    print(f"PULSE: Preprocessing complete. Total rows: {len(df)}")
    print(f"PULSE: Rows labeled as spike=1: {spike_count}")
    if label_mode == "forecast":
        print(
            "PULSE: Forecast mode enabled "
            f"(target is spike at t+{horizon_steps} steps, CPU threshold={cpu_threshold})."
        )
    print(f"PULSE: Saved preprocessed data to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PULSE preprocessing")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/system_metrics.csv"),
        help="Input CSV path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/system_metrics_labeled.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--label-mode",
        choices=["simple", "forecast"],
        default="forecast",
        help="Labeling mode: simple current-state or forecast future-state",
    )
    parser.add_argument(
        "--cpu-threshold",
        type=float,
        default=80.0,
        help="CPU threshold used for spike labeling",
    )
    parser.add_argument(
        "--horizon-steps",
        type=int,
        default=5,
        help="Forecast horizon in rows for forecast mode",
    )
    parser.add_argument(
        "--window-sizes",
        type=str,
        default="3,5",
        help="Rolling window sizes for temporal features, comma-separated",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess_data(
        input_path=args.input,
        output_path=args.output,
        label_mode=args.label_mode,
        cpu_threshold=args.cpu_threshold,
        horizon_steps=args.horizon_steps,
        window_sizes=parse_window_sizes(args.window_sizes),
    )