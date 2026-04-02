from __future__ import annotations

from typing import Iterable

import pandas as pd


BASE_FEATURE_COLUMNS = ["cpu_percent", "ram_percent", "process_count"]


def parse_window_sizes(window_sizes_text: str) -> list[int]:
    values = []
    for token in window_sizes_text.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value > 1:
            values.append(value)
    return sorted(set(values)) or [3, 5]


def add_temporal_features(
    df: pd.DataFrame,
    base_columns: Iterable[str] = BASE_FEATURE_COLUMNS,
    window_sizes: Iterable[int] = (3, 5),
) -> pd.DataFrame:
    engineered = df.copy()

    for col in base_columns:
        engineered[col] = pd.to_numeric(engineered[col], errors="coerce")
        engineered[f"{col}_lag1"] = engineered[col].shift(1)
        engineered[f"{col}_lag2"] = engineered[col].shift(2)
        engineered[f"{col}_delta1"] = engineered[col] - engineered[col].shift(1)

        for window in window_sizes:
            engineered[f"{col}_roll_mean_{window}"] = engineered[col].rolling(window).mean()
            engineered[f"{col}_roll_std_{window}"] = engineered[col].rolling(window).std()

    return engineered


def prepare_forecast_dataframe(
    df: pd.DataFrame,
    horizon_steps: int,
    cpu_threshold: float,
    window_sizes: Iterable[int] = (3, 5),
) -> pd.DataFrame:
    if horizon_steps < 1:
        raise ValueError("PULSE: horizon_steps must be >= 1")

    engineered = add_temporal_features(df, BASE_FEATURE_COLUMNS, window_sizes)
    engineered["spike"] = (engineered["cpu_percent"].shift(-horizon_steps) > cpu_threshold).astype("float")

    # Rows with NaNs are expected at the start (lags/rolling) and tail (future shift).
    # Drop them to keep training/inference features consistent.
    engineered = engineered.dropna().reset_index(drop=True)
    engineered["spike"] = engineered["spike"].astype(int)

    return engineered


def build_live_feature_frame(
    history_rows: list[dict],
    feature_columns: list[str],
    window_sizes: Iterable[int] = (3, 5),
) -> pd.DataFrame | None:
    if not history_rows:
        return None

    history_df = pd.DataFrame(history_rows)
    engineered = add_temporal_features(history_df, BASE_FEATURE_COLUMNS, window_sizes)
    latest = engineered.iloc[-1]

    missing = []
    row_data = {}
    for feature in feature_columns:
        if feature not in latest.index:
            row_data[feature] = 0.0
            continue

        value = latest[feature]
        if pd.isna(value):
            missing.append(feature)
        row_data[feature] = float(value) if not pd.isna(value) else 0.0

    if missing:
        return None

    return pd.DataFrame([row_data], columns=feature_columns)
