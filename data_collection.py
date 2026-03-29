import argparse
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import psutil


def capture_metrics() -> dict:
    """Collect one snapshot of system metrics."""
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cpu_percent": psutil.cpu_percent(interval=1),
        "ram_percent": psutil.virtual_memory().percent,
        "process_count": len(psutil.pids()),
    }


def collect_data(samples: int, delay_seconds: float, output_path: Path) -> None:
    """Collect multiple metric snapshots and save them to CSV."""
    rows = []

    print("PULSE: Starting data collection...")
    for i in range(samples):
        row = capture_metrics()
        rows.append(row)
        print(
            f"PULSE: Sample {i + 1}/{samples} | "
            f"CPU={row['cpu_percent']}% RAM={row['ram_percent']}% "
            f"PROCESSES={row['process_count']}"
        )

        if i < samples - 1:
            time.sleep(delay_seconds)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    new_df = pd.DataFrame(rows)
    if output_path.exists():
        old_df = pd.read_csv(output_path)
        full_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        full_df = new_df

    full_df.to_csv(output_path, index=False)
    print(f"PULSE: Saved {len(rows)} rows to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PULSE system metrics data collection")
    parser.add_argument("--samples", type=int, default=30, help="Number of rows to collect")
    parser.add_argument(
        "--delay", type=float, default=2.0, help="Delay (seconds) between samples"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/system_metrics.csv"),
        help="CSV output path",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    collect_data(samples=args.samples, delay_seconds=args.delay, output_path=args.output)
