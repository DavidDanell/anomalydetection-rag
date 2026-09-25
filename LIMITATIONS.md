# Limitations

## Data generation

- **Synthetic faults are cleaner than real ones.** The baseline has stationary
  noise, no operating regimes, no maintenance events disturbing the series, and
  no sensor drift or recalibration. Detection scores here do not transfer as
  performance claims about real equipment.

- **The daily cycle is a perfect sine.** Real load and ambient patterns vary by
  weekday, shift and weather. The cycle was a major source of false alarms here
  — the sensor with the strongest cycle had about twice as many as the others —
  so irregular real patterns would likely make detection harder.

- **Fault magnitudes are capped by physical plausibility.** Vibration cannot go
  negative, which limits how strong a variance fault can be on MTR-03. On
  magnitude alone these faults are harder to detect than severe real ones, but
  other simplifications push the other way. The net effect is unknown.

- **The sensor-to-asset mapping is given here.** Sensor ids match document
  metadata by construction. In a real plant this mapping has to exist and be
  verified before any filtering is possible.

- **Drift recovery is guessed, not calibrated.** The exponential decay back to
  baseline is plausible for a thermal system but is not fitted to real cooling
  behaviour.

## Detection

- **Rolling z-score only detects spikes reliably.** Its apparent recall on
  drift and variance equals chance: false alarms land inside long fault windows
  whether or not a fault is present, as shown by running the same detector on
  the fault-free baseline.

- **Four faults per fault type.** One fault changing outcome moves that type's
  recall by 25 percentage points. Per-type results show direction, not precision.

- **One noise realisation.** All results come from a single seed. Which fault
  windows happen to contain a false alarm would change with another.

- **Event recall cannot separate a good detector from a bad one.** Both
  detectors score 1.0 recall on drift, and both score 1.0 on the fault-free
  baseline. Counting flags inside each fault window shows the real picture:
  z-score has 6 flags against 3 on clean data, Isolation Forest has 156
  against 8. Recall is reported here only alongside the flag counts.

- **Rolling z-score detects only spikes.** 4 of 16 faults produce flag counts
  above the fault-free level. Its apparent recall on drift and variance is
  chance.

- **Isolation Forest detects 11 of 16.** The five it misses are the weakest
  and the shortest faults.

- **contamination is a shared budget.** It flags a fixed share of samples per
  sensor, so a short fault competes with louder faults on the same sensor.
  F011 alone takes 215 of PMP-01's 418 slots, which is part of why the two
  short flatlines are missed.

- **Four faults per fault type.** One fault changing outcome moves that type's
  recall by 25 percentage points. Per-type results show direction, not precision.

- **One noise realisation.** All results come from a single seed.