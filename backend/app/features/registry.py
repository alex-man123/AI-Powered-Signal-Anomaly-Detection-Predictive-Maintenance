"""TASK 5.1 — feature registry; extended by TASK 5.3 with a second, separate
registry for TASK 5.2's frequency-domain features.

`FEATURE_REGISTRY` is a plain dict mapping feature name -> pure function
(`app.features.time_domain`'s own `f(signal: Sequence[float]) -> float` functions,
unmodified). Extensible by adding a single entry
(`FEATURE_REGISTRY["new_feature"] = new_feature`) -- no dispatch if/elif chain exists
anywhere, so adding a feature never requires touching existing entries or existing
call sites.

`FREQUENCY_FEATURE_REGISTRY` is the same pattern for `app.features.frequency_domain`'s
functions, kept as a SEPARATE dict rather than merged into `FEATURE_REGISTRY` because
the two groups of functions do not share a calling convention: `FEATURE_REGISTRY`
functions take `f(signal)`, while frequency-domain functions take
`f(frequencies, power)` (an already-computed spectrum, per TASK 5.2's own design --
see `app.features.frequency_domain`'s module docstring). TASK 5.3's extractor
explicitly does not force these into one artificial shared signature; instead it
consumes both registries, each according to its own real contract. `FEATURE_REGISTRY`
itself is left completely unmodified here -- this is a pure addition, not a rewrite of
TASK 5.1's own registry.
"""

from __future__ import annotations

from functools import partial
from typing import Callable, Sequence

import numpy as np

from app.features.frequency_domain import (
    band_energy,
    dominant_frequency,
    spectral_bandwidth,
    spectral_centroid,
    spectral_energy,
    spectral_entropy,
)
from app.features.time_domain import (
    crest_factor,
    kurtosis,
    mean,
    peak,
    peak_to_peak,
    rms,
    skewness,
    std,
    variance,
)

FEATURE_REGISTRY: dict[str, Callable[[Sequence[float]], float]] = {
    "mean": mean,
    "std": std,
    "variance": variance,
    "rms": rms,
    "peak": peak,
    "peak_to_peak": peak_to_peak,
    "skewness": skewness,
    "kurtosis": kurtosis,
    "crest_factor": crest_factor,
}

# `band_energy` needs two extra scalar arguments (`low_freq`/`high_freq`) beyond
# `(frequencies, power)`, so it is bound to one fixed band via `functools.partial`
# rather than being registered with a different arity than the other five -- every
# entry in this registry ends up callable as `f(frequencies, power) -> float`, no
# exceptions, without modifying `band_energy` itself.
#
# The band (10-100 Hz) is an explicitly arbitrary, low-frequency default -- NOT a
# claim of MAFAULDA-specific physical significance (docs/dataset_audit/
# recording_mapping.md itself only says filenames are "plausibly" rotation-speed
# related, never confirmed as an exact rotation-frequency range, so no specific
# fault-frequency band could honestly be derived from the audit). It exists only so
# `band_energy` -- one of TASK 5.2's 6 delivered features -- has a concrete default
# entry in this uniform-arity registry; it stays comfortably below Nyquist for both
# the real dataset (fs=50,000 Hz) and any small synthetic test signal (e.g.
# fs=1,000 Hz -> Nyquist=500 Hz). Callers needing a specific, physically meaningful
# band call `band_energy(frequencies, power, low_freq, high_freq)` directly (it
# remains fully exported from `app.features.frequency_domain`, unaffected by this
# registry entry).
FREQUENCY_FEATURE_REGISTRY: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "dominant_frequency": dominant_frequency,
    "spectral_centroid": spectral_centroid,
    "spectral_bandwidth": spectral_bandwidth,
    "spectral_energy": spectral_energy,
    "spectral_entropy": spectral_entropy,
    "band_energy_10_100hz": partial(band_energy, low_freq=10.0, high_freq=100.0),
}
