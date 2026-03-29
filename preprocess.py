import argparse
from pathlib import Path

import pandas as pd


def preprocess_data(input_path: Path, output_path: Path) -> None:
    df = pd.read_csv(input_path)

    required_columns = ["timestamp", "cpu_percent", "ram_percent", "process_count"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"PULSE: Missing required columns: {missing}")

    df["spike"] = (df["cpu_percent"] > 80).astype(int)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    spike_count = int(df["spike"].sum())
    print(f"PULSE: Preprocessing complete. Total rows: {len(df)}")
    print(f"PULSE: Rows labeled as spike=1: {spike_count}")
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess_data(args.input, args.output)