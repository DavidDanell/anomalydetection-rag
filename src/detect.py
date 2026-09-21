from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def rolling_zscore(
    values: pd.Series,
    window: int,
    min_periods: int,
) -> pd.Series:
    """Compute the rolling z-score against a trailing window.

    The window excludes the point being scored, so a value cannot inflate
    the statistics it is measured against.

    Args:
        values: Sensor signal, indexed by timestamp.
        window: Number of past samples in the window.
        min_periods: Minimum samples required before a score is produced.

    Returns:
        Z-scores, NaN where the window is incomplete or has zero spread.
    """

    past = values.shift(1).rolling(window=window, min_periods=min_periods)

    mean = past.mean()
    std = past.std()

    return (values - mean) / std.replace(0.0, np.nan)


# s = pd.Series([10.0] * 100 + [20.0] + [10.0] * 10)
# z = rolling_zscore(s, window=50, min_periods=20)
# print(z.iloc[95:105])

# rng = np.random.default_rng(0)
# s = pd.Series(rng.normal(10, 0.5, 200))
# s.iloc[150] += 5.0

# z = rolling_zscore(s, window=50, min_periods=20)
# print('\n',z.iloc[148:153].round(2))

ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    """Load and parse config.yaml from the repo root."""
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def detect_all(sensors: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Score every sensor and flag samples above the threshold.

    Args:
        sensors: Signal data, one column per sensor.
        cfg: Parsed config.yaml.

    Returns:
        Long-format frame with one row per flagged sample.
    """
    params = cfg["detection"]["zscore"]

    rows = []
    for sensor_id in sensors.columns:
        z = rolling_zscore(
            sensors[sensor_id],
            window=params["window"],
            min_periods=params["min_periods"],
        )
        flagged = z[z.abs() > params["threshold"]]

        for timestamp, score in flagged.items():
            rows.append(
                {
                    "timestamp": timestamp,
                    "sensor_id": sensor_id,
                    "method": "zscore",
                    "value": round(sensors.at[timestamp, sensor_id], 3),
                    "score": round(score, 3)
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    """Run detection and write the flagged samples to disk."""
    cfg = load_config()

    sensors = pd.read_csv(
        ROOT / cfg["data"]["output"], index_col="timestamp", parse_dates=True
    )

    detections = detect_all(sensors, cfg)
    out = ROOT / cfg["detection"]["output"]
    detections.to_csv(out, index=False)

    print(f"{len(detections)} flaggade samples -> {out}")
    print()
    print(detections["sensor_id"].value_counts())


if __name__ == "__main__":
    main()
