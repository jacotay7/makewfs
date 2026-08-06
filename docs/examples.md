# Examples

The scripts under `examples/` are deterministic, headless, and save plots. See
[`examples/README.md`](https://github.com/jacotay7/makewfs/tree/main/examples)
for the command list.

The most complete current workflows are:

- `showcase.py`: four sensor configurations on one wind-blown atmosphere, each
  overlaid with its measured end-to-end throughput, written as an animated WebP;
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
- `cds_readout.py`: an H-band pyramid frame read both as an ordinary integration
  and as correlated double sampling on a C-RED One.
- `keck_haka/`: Keck II HAKA open-loop video across guide-star magnitudes 5--15,
  with a 57x57 Shack-Hartmann, 4x4 pixels per subaperture, OCAM2K mode lookup,
  `pyturb` Maunakea turbulence, a V-normalized 6600 K spectrum over 400--950 nm,
  measured wavelength-dependent Maunakea extinction, three 0.88-reflectivity
  aluminum telescope mirrors, a live-data-fitted circle-plus-hexagon secondary
  shadow, and an unscaled animated comparison to the V=10.16 eng519 RTC cube
  including eight relative amplifier responses.
- `sh_design_trade.py`, `pyramid_modulation.py`, `realistic_broadband.py`, and
  `precision_throughput.py` cover the optical design, modulation, broadband
  pupil, and CPU precision trade studies.

The extended-source configuration at
`examples/configs/shack_hartmann_extended_source.toml` demonstrates the
three-column angular-kernel input used for resolved or binary guide sources.

The labelled [documentation gallery](gallery.md) is generated from the same
shipped configurations and keeps its SVG/manifest outputs under `docs/gallery/`.

The gallery is a compact deterministic documentation artifact; the individual
scripts remain available for full-resolution timing traces, detector-preset
comparisons, and closed-loop toy-injection studies.
