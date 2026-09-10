from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# generates a basline, no errors

ROOT = Path(__file__).resolve().parents[1]


def load_config():
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


def build_dataframe(cfg: dict) -> pd.DataFrame:
    """Build the full sensor dataset from a loaded config.

    Args:
        cfg: Parsed config.yaml.

    Returns:
        DataFrame with a DatetimeIndex and one column per sensor.
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

    return pd.DataFrame(series, index=index)

def main() -> None:
    """Generate the dataset and write it to disk."""
    cfg = load_config()
    df = build_dataframe(cfg)

    out = ROOT / cfg["data"]["output"]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, float_format="%.3f")

    print(f"{len(df)} rader, {len(df.columns)} sensorer -> {out}")
    print(df.describe())


if __name__ == "__main__":
    main()