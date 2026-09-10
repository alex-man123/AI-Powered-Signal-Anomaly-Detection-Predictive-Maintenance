"""TASK 10.4 -- orchestrates HTTP requests into Phase 5.3's existing feature
extractor. No feature formula is reimplemented here: every value comes from
`app.features.extractor.extract_features`, which itself iterates
`app.features.registry.FEATURE_REGISTRY`/`FREQUENCY_FEATURE_REGISTRY` -- the
same registries TASK 5.1/5.2 already defined and tested. `feature_names`/
`values` come directly from that call's own dict (`dict.keys()`/`values()`,
insertion order preserved -- the extractor's own documented ordering
contract), never a separately hard-coded name list, so a future registry
addition is picked up automatically with no change here.
"""

from __future__ import annotations

from app.api.schemas.features import FeatureExtractionRequest, FeatureExtractionResponse
from app.features.extractor import extract_features


def run_feature_extraction(request: FeatureExtractionRequest) -> FeatureExtractionResponse:
    features = extract_features(
        request.signal,
        request.sampling_rate,
        nperseg=request.nperseg,
        noverlap=request.noverlap,
    )
    return FeatureExtractionResponse(
        feature_names=list(features.keys()),
        values=list(features.values()),
    )
