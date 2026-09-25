"""Evaluate detected alarms against the fault manifest."""

import copy
from collections.abc import Callable

import pandas as pd

from data_generation import build_dataset
from detect import ROOT, detect_iforest, detect_zscore, load_config

Detector = Callable[[pd.DataFrame, dict], pd.DataFrame]


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


def run_on_clean(cfg: dict, detector: Detector) -> pd.DataFrame:
    """Run a detector on the same baseline with all faults removed."""
    clean_cfg = copy.deepcopy(cfg)
    clean_cfg["faults"]["injections"] = []
    clean_sensors, _ = build_dataset(clean_cfg)
    return detector(clean_sensors, clean_cfg)


def chance_recall(clean: pd.DataFrame, faults: pd.DataFrame, gap: int) -> pd.Series:
    """Recall scored on fault-free data — the level to beat."""
    _, matched = match_alarms(group_alarms(clean, gap), faults)
    return recall_by_type(matched)["recall"]


DETECTORS: dict[str, Detector] = {
    "zscore": detect_zscore,
    "iforest": detect_iforest,
}


def flags_per_fault(detections: pd.DataFrame, faults: pd.DataFrame) -> pd.Series:
    """Count flagged samples inside each fault window."""
    counts = {}
    for _, f in faults.iterrows():
        inside = (
            (detections["sensor_id"] == f["sensor_id"])
            & (detections["timestamp"] >= f["start"])
            & (detections["timestamp"] <= f["end"])
        )
        counts[f["fault_id"]] = int(inside.sum())
    return pd.Series(counts)


def main() -> None:
    """Evaluate every detector against the manifest and against chance."""
    cfg = load_config()
    sensors, faults = build_dataset(cfg)
    gap = cfg["detection"]["alarm_gap_minutes"]

    for name, detector in DETECTORS.items():
        detections = detector(sensors, cfg)
        clean = run_on_clean(cfg, detector)

        alarms, matched = match_alarms(group_alarms(detections, gap), faults)
        table = recall_by_type(matched)
        table["chance_recall"] = chance_recall(clean, faults, gap)

        per_fault = faults.set_index("fault_id")[["fault_type"]]
        per_fault["with_fault"] = flags_per_fault(detections, faults)
        per_fault["without_fault"] = flags_per_fault(clean, faults)

        print(f"\n=== {name} ===")
        print(f"Precision: {precision(alarms):.1%}  ({len(alarms)} larm)")
        print(table.round(2).to_string())
        print()
        print(per_fault.to_string())

if __name__ == "__main__":
    main()
