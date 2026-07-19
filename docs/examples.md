# Examples

The scripts under `examples/` are deterministic, headless, and save plots. See
[`examples/README.md`](https://github.com/jacotay7/makewfs/tree/main/examples)
for the command list.

The most complete current workflows are:

- `moving_atmosphere.py`: `pyturb` OPD frames into a persistent sensor and
  `getframes` detector;
- `magnitude_series.py`: one detector configuration across NGS magnitudes;
- `compare_sensors.py`: identical injected OPD through SH and pyramid optics;
- `lgs_thin_beacon.py`: explicit thin-beacon LGS return-rate and cone-effect
  boundary;
- `lgs_elongation.py`: centre/side launch and finite sodium-range elongation.
- `closed_loop_injection.py`: external residual update at the closed-loop API
  boundary;
- `detector_choices.py`: identical ideal maps through several getframes presets.
- `sh_design_trade.py`, `pyramid_modulation.py`, `realistic_broadband.py`, and
  `precision_throughput.py` cover the optical design, modulation, broadband
  pupil, and CPU precision trade studies.

The extended-source configuration at
`examples/configs/shack_hartmann_extended_source.toml` demonstrates the
three-column angular-kernel input used for resolved or binary guide sources.

Full-resolution galleries, timing traces, detector-presets comparisons, and a
closed-loop toy injection plot are scheduled in the roadmap.
