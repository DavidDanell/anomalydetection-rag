"""Generate synthetic sensor data with injected faults."""


from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config():
    """Load and parse config.yaml from the repo root."""
    with open(ROOT / "config.yaml", encoding="UTF-8") as f:
        return yaml.safe_load(f)


def generate_baseline(
    n: int,
    mean: float,
    noise_std: float,
    daily_amp: float,
    samples_per_day: float,
    rng: np.random.Generator,
    phi: float,
    slow_std: float,
) -> np.ndarray:
    """Generate one sensor's baseline signal — normal operation, no faults.

    Args:
        n: Number of samples to generate.
        mean: The sensor's normal operating value.
        noise_std: Standard deviation of the fast measurement noise.
        daily_amp: Amplitude of the 24-hour cycle, in sensor units.
        samples_per_day: Samples in one day. Sets the period of the cycle.
        rng: Random generator. Passed in so the caller controls the seed.
        phi: AR(1) pull-back toward the mean. Near 1 = slow return.
        slow_std: Step size of the slow drift.

    Returns:
        Array of length n.
    """
    t = np.arange(n)

    daily = daily_amp * np.sin(2 * np.pi * t / samples_per_day)

    steps = rng.normal(0, slow_std, n)
    slow = np.empty(n)
    slow[0] = 0.0
    for i in range(1, n):
        slow[i] = phi * slow[i - 1] + steps[i]

    noise = rng.normal(0, noise_std, n)

    return mean + daily + slow + noise

def inject_spike(
    values: np.ndarray,
    start_idx: int,
    end_idx: int,
    magnitude: float,
) -> None:
    """Add a large offset to a short window. Simulates a sensor glitch.

    Args:
        values: Signal to modify, in place.
        start_idx: First affected sample.
        end_idx: One past the last affected sample.
        magnitude: Offset in sensor units.
    """
    values[start_idx:end_idx] += magnitude





def inject_flatline(values: np.ndarray, start_idx: int, end_idx: int) -> None:
    """Freeze the signal at its last value before the window. Simulates faulty sensor.

    Args:
        values: Signal to modify, in place.
        start_idx: First affected sample.
        end_idx: One past the last affected sample.
    """
    if start_idx < 1:
        raise ValueError("start_idx must be >= 1")

    values[start_idx:end_idx] = values[start_idx - 1]


def inject_drift(
    values: np.ndarray,
    start_idx: int,
    peak_idx: int,
    end_idx: int,
    magnitude: float,
) -> None:
    """Ramp linearly to magnitude, then decay exponentially back to baseline.

    Args:
        values: Signal to modify, in place.
        start_idx: First affected sample.
        peak_idx: Sample where the drift reaches full magnitude.
        end_idx: One past the last affected sample, including recovery.
        magnitude: Peak offset in sensor units.
    """
    ramp = np.linspace(0, 1, peak_idx - start_idx)
    values[start_idx:peak_idx] += magnitude * ramp

    n_decay = end_idx - peak_idx
    if n_decay > 0:
        decay = np.exp(-np.linspace(0, 4, n_decay))
        values[peak_idx:end_idx] += magnitude * decay


def inject_variance(
    values: np.ndarray,
    start_idx: int,
    end_idx: int,
    magnitude: float,
    rng: np.random.Generator,
    base_noise_std: float,
) -> None:
    """Add noise with linearly growing spread. Mean stays unchanged.

    Simulates developing mechanical imbalance.

    Args:
        values: Signal to modify, in place.
        start_idx: First affected sample.
        end_idx: One past the last affected sample.
        magnitude: Peak extra spread as a multiple of base_noise_std.
        rng: Random generator.
        base_noise_std: The sensor's normal noise level.
    """
    n_samples = end_idx - start_idx
    ramp = np.linspace(0, 1, n_samples)
    extra = rng.normal(0, 1, n_samples) * ramp * magnitude * base_noise_std
    values[start_idx:end_idx] += extra


def resolve_window(
    start: str,
    duration_minutes: int,
    index: pd.DatetimeIndex,
    rate: int,
) -> tuple[int, int]:
    """Convert a fault's start time and duration to array positions.

    Args:
        start: Fault start as a timestamp string.
        duration_minutes: How long the fault lasts.
        index: The dataset's time index.
        rate: Seconds between samples.

    Returns:
        (start_idx, end_idx), where end_idx is exclusive.

    Raises:
        KeyError: If start is not in index.
        ValueError: If the window runs past the end of the data.
    """
    start_idx = index.get_loc(pd.Timestamp(start))
    n_samples = duration_minutes * 60 // rate
    end_idx = start_idx + n_samples

    if end_idx > len(index):
        raise ValueError(f"fault window at {start} runs past end of data")

    return start_idx, end_idx


def apply_faults(
    series: dict[str, np.ndarray],
    cfg: dict,
    index: pd.DatetimeIndex,
) -> list[dict]:
    """Apply all configured faults in place and return the manifest rows.

    Args:
        series: Sensor id -> signal array. Modified in place.
        cfg: Parsed config.yaml.
        index: The dataset's time index.

    Returns:
        One dict per injected fault, ready for a DataFrame.
    """
    rate = cfg["data"]["sample_rate_seconds"]
    noise_by_sensor = {s["id"]: s["noise_std"] for s in cfg["data"]["sensors"]}
    rng = np.random.default_rng(cfg["faults"]["seed"])

    manifest = []
    seen: dict[str, list[tuple[int, int]]] = {}

    for i, fault in enumerate(cfg["faults"]["injections"], start=1):
        sensor_id = fault["sensor"]
        values = series[sensor_id]

        start_idx, end_idx = resolve_window(
            fault["start"], fault["duration_minutes"], index, rate
        )

        kind = fault["type"]
        if kind == "spike":
            inject_spike(values, start_idx, end_idx, fault["magnitude"])

        elif kind == "drift":
            n_recovery = fault.get("recovery_minutes", 0) * 60 // rate
            peak_idx = end_idx
            end_idx = min(end_idx + n_recovery, len(index))
            inject_drift(values, start_idx, peak_idx, end_idx, fault["magnitude"])

        elif kind == "flatline":
            inject_flatline(values, start_idx, end_idx)

        elif kind == "variance":
            inject_variance(
                values,
                start_idx,
                end_idx,
                fault["magnitude"],
                rng,
                noise_by_sensor[sensor_id],
            )
        else:
            raise ValueError(f"unknown fault type: {kind}")

        for s, e in seen.get(sensor_id, []):
            if start_idx < e and s < end_idx:
                raise ValueError(
                    f"fault {i} overlaps an earlier fault on {sensor_id}"
                )
        seen.setdefault(sensor_id, []).append((start_idx, end_idx))

        manifest.append(
            {
                "fault_id": f"F{i:03d}",
                "sensor_id": sensor_id,
                "fault_type": kind,
                "start": index[start_idx],
                "end": index[end_idx - 1],
                "magnitude": fault["magnitude"],
            }
        )

    return manifest

def build_dataset(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the sensor dataset and its fault manifest.

    Args:
        cfg: Parsed config.yaml.

    Returns:
        (sensors, faults) — the signal data and the ground truth.
    """
    d = cfg["data"]
    rate = d["sample_rate_seconds"]
    n = d["days"] * 24 * 3600 // rate
    samples_per_day = 24 * 3600 / rate

    index = pd.date_range(
        start=d["start"], periods=n, freq=f"{rate}s", name="timestamp"
    )

    parent = np.random.default_rng(d["seed"])
    children = parent.spawn(len(d["sensors"]))

    series = {}
    for sensor, child in zip(d["sensors"], children):
        series[sensor["id"]] = generate_baseline(
            n=n,
            mean=sensor["mean"],
            noise_std=sensor["noise_std"],
            daily_amp=sensor["daily_amp"],
            samples_per_day=samples_per_day,
            rng=child,
            phi=d["baseline"]["phi"],
            slow_std=sensor["slow_std"],
        )

    manifest = apply_faults(series, cfg, index)
    sensors = pd.DataFrame(series, index=index)
    faults = pd.DataFrame(manifest)

    return sensors, faults

def main() -> None:
    """Generate the dataset and write it to disk."""
    cfg = load_config()
    sensors, faults = build_dataset(cfg)

    sensors_path = ROOT / cfg["data"]["output"]
    faults_path = ROOT / cfg["faults"]["output"]

    sensors_path.parent.mkdir(parents=True, exist_ok=True)

    sensors.to_csv(sensors_path, float_format="%.3f")
    faults.to_csv(faults_path, index=False)

    print(f"{len(sensors)} rader, {len(sensors.columns)} sensorer -> {sensors_path}")
    print(f"{len(faults)} fel -> {faults_path}")
    print()
    print(faults.to_string(index=False))
    print()
    print(sensors.describe())


if __name__ == "__main__":
    main()
