"""Guard the intentionally small top-level API surface."""

import makewfs


def test_top_level_exports_are_intentional() -> None:
    assert makewfs.__all__ == [
        "Config",
        "ConfigError",
        "WFSConfig",
        "WavefrontSensor",
        "__version__",
        "load_config",
        "simulate",
    ]
    assert makewfs.Config is makewfs.WFSConfig
    assert not hasattr(makewfs, "DetectorAdapter")
    assert not hasattr(makewfs, "ShackHartmannEngine")
