# Worked examples

All scripts save plots without opening a window. Install the optional packages
with `python -m pip install -e '.[examples,interop]'` from the repository root.

- `quickstart.py` renders one configured Shack–Hartmann exposure.
- `compare_sensors.py` applies the same OPD to the SH and pyramid configurations.
- `moving_atmosphere.py` consumes frozen-flow OPD frames from `pyturb` and sends
  each one through `makewfs` and `getframes`.
- `magnitude_series.py` keeps the detector configuration fixed while comparing
  NGS magnitudes through the `getframes` radiometry path.
- `lgs_thin_beacon.py` demonstrates the current LGS contract: `pyturb` supplies
  cone-effect OPD and the user supplies a detector-surface return rate. Sodium
  range elongation is an explicit roadmap item and is not implied by this plot.

Use `--help` on each script for output and resolution controls.
