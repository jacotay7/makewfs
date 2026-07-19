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
  range elongation is not implied by this thin-beacon plot.
- `lgs_elongation.py` compares thin, centre-launched, and side-launched sodium
  range profiles using the current Shack–Hartmann geometry model.
- `closed_loop_injection.py` shows residual OPD entering a persistent sensor;
  the toy attenuation is explicitly external to makewfs.
- `detector_choices.py` holds the ideal optical map fixed while swapping
  existing CCD, EMCCD, sCMOS, and eAPD getframes presets.
- `sh_design_trade.py`: field-stop and blur sampling choices.
- `pyramid_modulation.py`: unmodulated versus circularly modulated PWFS response.
- `realistic_broadband.py`: rotated segmented pupil and incoherent spectral sensing.
- `precision_throughput.py`: float32/float64 warm-path latency and image agreement.
- `configs/precision_throughput.toml` is the representative 20x20 lenslet
  configuration used by that precision example.
- `configs/shack_hartmann_extended_source.toml` plus its kernel file show a
  measured/arbitrary finite-source morphology without changing the API.

Use `--help` on each script for output and resolution controls.
