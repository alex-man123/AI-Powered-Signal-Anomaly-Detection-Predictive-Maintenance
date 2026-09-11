"""TASK 12.2 -- response schema for `GET /api/experiments/pca-visualization`.

Mirrors `app.services.pca_service.get_pca_visualization`'s real return shape
field-for-field: one `PCAPoint` per real window (real `recording_id`, real
`label` straight from `app.models.signal.SignalLabel`, real PC1/PC2/PC3
coordinates from TASK 9.1's already-fitted PCA), plus the real, already-
measured `explained_variance_ratio` for exactly those first 3 components --
never a recomputation, never an invented "confidence" figure.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.models.signal import SignalLabel


class PCAPoint(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    recording_id: str = Field(min_length=1)
    label: SignalLabel
    pc1: float = Field(strict=True)
    pc2: float = Field(strict=True)
    pc3: float = Field(strict=True)


class PCAVisualizationResponse(BaseModel):
    points: list[PCAPoint]
    explained_variance_ratio: list[float] = Field(min_length=3, max_length=3)
    total_components: int = Field(gt=0, strict=True)
