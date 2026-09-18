# Limitations

## Data generation

- **Synthetic faults are cleaner than real ones.** The baseline has stationary
  noise, no operating regimes, no maintenance events disturbing the series, and
  no sensor drift or recalibration. Detection scores here do not transfer as
  performance claims about real equipment.

- **Fault magnitudes are capped by physical plausibility.** Vibration cannot go
  negative, which limits how strong a variance fault can be on MTR-03. Real
  faults can be far more severe, so these results are pessimistic rather than
  optimistic.

- **The sensor-to-asset mapping is given here.** Sensor ids match document
  metadata by construction. In a real plant this mapping has to exist and be
  verified before any filtering is possible.

- **Drift recovery is guessed, not calibrated.** The exponential decay back to
  baseline is plausible for a thermal system but is not fitted to real cooling
  behaviour.