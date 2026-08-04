# Worked examples

All scripts save plots (or a GIF) without opening a window. Install the optional
packages with `python -m pip install -e '.[examples,interop]'` from the
repository root. Every raster panel carries a colorbar with units, and each
script accepts `--help` for output and resolution controls.

- `showcase.py` animates four sensor configurations (20×20 SH, 60×60 SH,
  modulated pyramid, broadband LGS SH) watching one wind-blown atmosphere, each
  overlaid with its measured end-to-end throughput, and writes the animated
  **WebP** used in the README. It has its own flags (`--device`, `--frames`,
  `--residual-scale`, `--out`) and picks up a GPU automatically when CuPy is
  installed.
- `quickstart.py` renders one configured Shack–Hartmann exposure.
- `compare_sensors.py` applies the same OPD to the SH and pyramid configurations.
- `moving_atmosphere.py` animates frozen-flow OPD frames from `pyturb` through
  `makewfs` and `getframes` and writes a **GIF**. A residual-scaling factor
  emulates partial correction so the drifting spots stay legible.
- `magnitude_series.py` keeps a fixed real sCMOS preset while comparing NGS
  magnitudes from photon-rich to read-noise-limited.
- `lgs_thin_beacon.py` demonstrates the current LGS contract: `pyturb` supplies
  cone-effect OPD and the user supplies a detector-surface return rate. It
  contrasts the flat point-like beacon with the turbulent one; no range
  elongation is implied.
- `lgs_elongation.py` compares thin, centre-launched, and side-launched sodium
  range profiles; spots elongate radially from the launch, growing with launch
  distance, using a wide subaperture field of view so the streaks are not clipped.
- `closed_loop_injection.py` drives a low-order residual toward zero via a toy
  external loop and tracks the residual RMS and the frame's departure from the
  flat reference; the controller is explicitly outside makewfs.
- `detector_choices.py` holds the ideal optical map fixed at a faint magnitude
  while swapping real CCD, EMCCD, sCMOS, and CMOS getframes presets so their
  noise characters differ.
- `sh_design_trade.py`: field-stop and blur sampling choices.
- `pyramid_modulation.py`: unmodulated versus circularly modulated PWFS response.
- `realistic_broadband.py`: rotated segmented pupil (shown as an amplitude mask)
  and incoherent broadband sensing with a zoom on the chromatic spot smearing.
- `precision_throughput.py`: float32/float64 warm-path latency plus the actual
  images and their difference map.
- `gallery.py`: deterministic six-panel documentation gallery with labelled
  units, color bars, configuration digests, and modeling notes.
- `spectral_qe.py`: wavelength-resolved versus scalar detector QE, shown as a
  blue guide star and a red one on a real CMOS preset whose QE rolls off in the
  red; only the spectral path penalizes the red star.
- `keck_haka/`: a Keck II HAKA 57x57 Shack-Hartmann simulation using a 228x228
  OCAM2K ROI, the supplied magnitude-dependent EM-gain/frame-rate table, and
  open-loop `pyturb` Maunakea turbulence. It writes a GIF or MP4 plus a JSON
  provenance manifest and includes an unscaled real-RTC comparison with the
  measured eight-output OCAM response structure and a fitted secondary shadow.
- `configs/precision_throughput.toml` is the representative 20x20 lenslet
  configuration used by that precision example.
- `configs/shack_hartmann_extended_source.toml` plus its kernel file show a
  measured/arbitrary finite-source morphology without changing the API.

Use `--help` on each script for output and resolution controls.
