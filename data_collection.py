import argparse
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import psutil


def capture_metrics(scenario: str, source_tag: str) -> dict:
    """Collect one snapshot of system metrics."""
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cpu_percent": psutil.cpu_percent(interval=1),
        "ram_percent": psutil.virtual_memory().percent,
        "process_count": len(psutil.pids()),
        "scenario": scenario,
        "source_tag": source_tag,
    }


def collect_data(
    samples: int,
    delay_seconds: float,
    output_path: Path,
    scenario: str,
    source_tag: str,
) -> None:
    """Collect multiple metric snapshots and save them to CSV."""
    rows = []

    print(f"PULSE: Starting data collection (scenario={scenario}, source={source_tag})...")
    for i in range(samples):
        row = capture_metrics(scenario=scenario, source_tag=source_tag)
        rows.append(row)
        print(
            f"PULSE: Sample {i + 1}/{samples} | "
            f"CPU={row['cpu_percent']}% RAM={row['ram_percent']}% "
            f"PROCESSES={row['process_count']} SCENARIO={row['scenario']}"
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
    parser.add_argument(
        "--scenario",
        type=str,
        default="normal",
        choices=["idle", "normal", "cpu_stress", "ram_stress", "mixed"],
        help="Scenario label to tag collected rows for dataset diversity",
    )
    parser.add_argument(
        "--source-tag",
        type=str,
        default="local_psutil",
        help="Data source tag written with each row",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    collect_data(
        samples=args.samples,
        delay_seconds=args.delay,
        output_path=args.output,
        scenario=args.scenario,
        source_tag=args.source_tag,
    )
