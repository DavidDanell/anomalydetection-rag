from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import IsolationForest

ROOT = Path(__file__).resolve().parents[1]


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


def load_config() -> dict:
    """Load and parse config.yaml from the repo root."""
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def detect_zscore(sensors: pd.DataFrame, cfg: dict) -> pd.DataFrame:
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
                    "score": round(score, 3),
                }
            )

    return pd.DataFrame(rows)


def build_features(
    values: pd.Series,
    samples_per_day: int,
    std_window: int,
    level_window: int,
) -> pd.DataFrame:
    """Build context features for one sensor.

    Each feature turns a contextual anomaly into a point anomaly. The raw
    value is deliberately excluded — it would reintroduce the daily cycle.

    Args:
        values: Sensor signal, indexed by timestamp.
        samples_per_day: Samples in 24 hours.
        std_window: Samples in the rolling std window.
        level_window: Samples in the rolling mean window.

    Returns:
        One row per timestamp with a complete feature set.
    """
    features = pd.DataFrame(
        {
            "diff_24h": values - values.shift(samples_per_day),
            "std_60": values.rolling(std_window).std(),
            "dev_6h": values - values.rolling(level_window).mean(),
        }
    )
    return features.dropna()


def detect_iforest(sensors: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Flag anomalies with one Isolation Forest per sensor.

    Each model is fitted on the data it scores — the realistic
    unsupervised setting, where no fault-free period is known.

    Args:
        sensors: Signal data, one column per sensor.
        cfg: Parsed config.yaml.

    Returns:
        Long-format frame with one row per flagged sample.
    """
    params = cfg["detection"]["iforest"]
    samples_per_day = 24 * 3600 // cfg["data"]["sample_rate_seconds"]

    rows = []
    for sensor_id in sensors.columns:
        X = build_features(
            sensors[sensor_id],
            samples_per_day,
            params["std_window"],
            params["level_window"],
        )
        model = IsolationForest(
            n_estimators=params["n_estimators"],
            contamination=params["contamination"],
            random_state=params["seed"],
        ).fit(X)

        is_anomaly = model.predict(X) == -1
        scores = -model.decision_function(X)

        for timestamp, score in zip(X.index[is_anomaly], scores[is_anomaly]):
            rows.append(
                {
                    "timestamp": timestamp,
                    "sensor_id": sensor_id,
                    "method": "iforest",
                    "value": round(sensors.at[timestamp, sensor_id], 3),
                    "score": round(score, 3),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    """Run both detectors and write all flagged samples to disk."""
    cfg = load_config()
    sensors = pd.read_csv(
        ROOT / cfg["data"]["output"], index_col="timestamp", parse_dates=True
    )

    detections = pd.concat(
        [detect_zscore(sensors, cfg), detect_iforest(sensors, cfg)],
        ignore_index=True,
    )
    out = ROOT / cfg["detection"]["output"]
    detections.to_csv(out, index=False)

    print(f"{len(detections)} flaggade samples -> {out}")
    print(detections.groupby(["method", "sensor_id"]).size())


if __name__ == "__main__":
    main()
