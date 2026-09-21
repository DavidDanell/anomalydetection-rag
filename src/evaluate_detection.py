"""Evaluate detected alarms against the fault manifest."""

import copy

import pandas as pd

from data_generation import build_dataset
from detect import ROOT, detect_all, load_config


def group_alarms(detections: pd.DataFrame, gap_minutes: int) -> pd.DataFrame:
    """Merge flagged samples into alarms.

    Flags on the same sensor belong to the same alarm when they are at
    most gap_minutes apart.

    Args:
        detections: One row per flagged sample.
        gap_minutes: Largest gap that still counts as the same alarm.

    Returns:
        One row per alarm: sensor_id, start, end, n_flags, peak_score.
    """
    d = detections.sort_values(["sensor_id", "timestamp"])

    gap = d.groupby("sensor_id")["timestamp"].diff()
    new_alarm = gap.isna() | (gap > pd.Timedelta(minutes=gap_minutes))
    d["alarm_id"] = new_alarm.cumsum()

    return (
        d.groupby("alarm_id")
        .agg(
            sensor_id=("sensor_id", "first"),
            start=("timestamp", "min"),
            end=("timestamp", "max"),
            n_flags=("timestamp", "size"),
            peak_score=("score", lambda s: s.abs().max()),
        )
        .reset_index(drop=True)
    )


def match_alarms(
    alarms: pd.DataFrame,
    faults: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match alarms to faults by alarm onset.

    An alarm is a true positive if it starts inside a fault window on the
    same sensor. A fault is detected if at least one alarm starts inside it.

    Args:
        alarms: One row per alarm, from group_alarms.
        faults: The fault manifest.

    Returns:
        (alarms, faults) — copies with match columns added.
    """
    alarms = alarms.copy()
    faults = faults.copy()
    alarms["fault_id"] = None

    delays = []
    for _, f in faults.iterrows():
        inside = (
            (alarms["sensor_id"] == f["sensor_id"])
            & (alarms["start"] >= f["start"])
            & (alarms["start"] <= f["end"])
        )
        alarms.loc[inside, "fault_id"] = f["fault_id"]
        delays.append(alarms.loc[inside, "start"].min() - f["start"])

    delay = pd.Series(delays, index=faults.index)
    faults["detected"] = delay.notna()
    faults["delay_min"] = delay.dt.total_seconds() / 60

    return alarms, faults


def precision(alarms: pd.DataFrame) -> float:
    """Share of alarms that start inside a fault window."""
    return alarms["fault_id"].notna().mean()


def recall_by_type(faults: pd.DataFrame) -> pd.DataFrame:
    """Recall and median detection delay per fault type.

    Args:
        faults: Matched faults, from match_alarms.

    Returns:
        One row per fault type.
    """
    table = faults.groupby("fault_type").agg(
        n_faults=("fault_id", "size"),
        detected=("detected", "sum"),
        median_delay_min=("delay_min", "median"),
    )
    table["recall"] = table["detected"] / table["n_faults"]
    return table


def chance_recall(cfg: dict, faults: pd.DataFrame) -> pd.Series:
    """Recall the detector scores on the same baseline with no faults.

    Faults use their own seed, so the baseline is identical with and
    without them. Any alarm starting inside a fault window on clean data
    is chance — this is the recall to beat.

    Args:
        cfg: Parsed config.yaml.
        faults: The fault manifest, used only for its windows.

    Returns:
        Chance recall per fault type.
    """
    clean_cfg = copy.deepcopy(cfg)
    clean_cfg["faults"]["injections"] = []

    clean_sensors, _ = build_dataset(clean_cfg)
    detections = detect_all(clean_sensors, clean_cfg)

    alarms = group_alarms(detections, cfg["detection"]["alarm_gap_minutes"])
    _, matched = match_alarms(alarms, faults)

    return recall_by_type(matched)["recall"]


def main() -> None:
    """Evaluate alarms against the manifest and report metrics."""
    cfg = load_config()

    sensors, faults = build_dataset(cfg)
    detections = detect_all(sensors, cfg)

    alarms = group_alarms(detections, cfg["detection"]["alarm_gap_minutes"])
    alarms, faults = match_alarms(alarms, faults)

    print()
    table = recall_by_type(faults)
    table["chance_recall"] = chance_recall(cfg, faults)
    print(table.round(2).to_string())
    n_true = alarms["fault_id"].notna().sum()
    print()
    print(
        faults[
            ["fault_id", "sensor_id", "fault_type", "detected", "delay_min"]
        ].to_string(index=False)
    )
    print(f"\nPrecision: {precision(alarms):.1%}")
    print(f"{len(alarms)} larm: {n_true} sanna, {len(alarms) - n_true} falska")
    print(f"{faults['detected'].sum()} av {len(faults)} fel hittade")


if __name__ == "__main__":
    main()
