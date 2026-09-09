"""TASK 5.1 — feature registry.

A plain dict mapping feature name -> pure function (`app.features.time_domain`'s own
`f(signal: Sequence[float]) -> float` functions, unmodified). Extensible by adding a
single entry (`FEATURE_REGISTRY["new_feature"] = new_feature`) -- no dispatch
if/elif chain exists anywhere, so adding a feature never requires touching existing
entries or existing call sites.
"""

from __future__ import annotations

from typing import Callable, Sequence

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
