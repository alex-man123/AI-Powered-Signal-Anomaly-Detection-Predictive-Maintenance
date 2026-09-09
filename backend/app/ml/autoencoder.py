"""TASK 7.1 — Autoencoder architecture (PyTorch, dense, on feature vectors).

Blueprint.md section 16: "encoder 2-3 straturi dense (dimensiune input = nr.
features -> bottleneck mic), decoder simetric. Suficient de simplă incat sa poti
desena arhitectura pe o coala de hartie intr-un interviu." No concrete
hidden_dim/bottleneck_dim/activation/parameter-count values are mandated there --
this module keeps every dimension a configurable constructor argument (never
hard-coded) and picks the simplest standard choice for what blueprint leaves open,
documented below.

Scope: architecture only.

    feature vector (N, input_dim)
            |
            v
       Linear(input_dim -> hidden_dim)
       ReLU
       Linear(hidden_dim -> bottleneck_dim)     <- encoder
            |
            v
         bottleneck (N, bottleneck_dim)          <- compressed representation
            |
            v
       Linear(bottleneck_dim -> hidden_dim)
       ReLU
       Linear(hidden_dim -> input_dim)           <- decoder (symmetric to encoder)
            |
            v
       reconstruction (N, input_dim)

- `input_dim`: number of DSP features per window (TASK 5.3's feature matrix column
  count) -- this module does not import the feature registry/extractor to compute
  it; the caller passes it in explicitly (see module-level "dependency direction"
  note below).
- `hidden_dim`: intermediate dense layer width.
- `bottleneck_dim`: the compressed representation size -- blueprint's own "bottleneck
  mic" is a qualitative design intent, not a hard numeric ratio, so this module
  validates only `> 0` for each dimension, not a specific ordering constraint (e.g.
  `bottleneck_dim < hidden_dim < input_dim`) -- nothing in blueprint.md mandates
  that as an enforced invariant, and inventing one here would reject configurations
  blueprint itself never ruled out.

Activation: ReLU after each hidden (non-final) Linear layer in both encoder and
decoder -- the simplest, most standard nonlinearity for a small dense autoencoder,
consistent with keeping the architecture "easy to explain" (section 16). The FINAL
decoder layer (bottleneck_dim -> input_dim) has NO activation: this network
reconstructs already-standardized feature vectors (TASK 5.4's `StandardScaler`
output, mean 0/unbounded range, not confined to `[0,1]` or `[-1,1]`) -- a bounded
output activation (Sigmoid/Tanh) would incorrectly clip what the network can ever
reconstruct, so none is added.

Total trainable parameters (every `nn.Linear(a, b)` contributes `a*b + b`, weight
matrix plus bias vector):

    (input_dim * hidden_dim + hidden_dim)          <- encoder layer 1
  + (hidden_dim * bottleneck_dim + bottleneck_dim)  <- encoder layer 2
  + (bottleneck_dim * hidden_dim + hidden_dim)      <- decoder layer 1
  + (hidden_dim * input_dim + input_dim)            <- decoder layer 2

Not implemented here (later tasks, out of this task's scope): training loop,
optimizer, loss function, reconstruction-error-based anomaly scoring, threshold
calibration, `.pt` persistence/metadata sidecar (TASK 6.5's contract already
supports `model_type="autoencoder"` at the schema level -- wiring an actual
Autoencoder artifact through it is TASK 7.3's job), and feature scaling (TASK 5.4's
`StandardScaler` is applied by the caller BEFORE a feature vector ever reaches this
model -- `Autoencoder` never imports `app.ml.scaling` or fits/applies a scaler
itself). Dependency direction is strictly `features -> scaling -> Autoencoder`,
never the reverse -- this module has no import from `app.features` or `app.ml.
scaling` anywhere.

Device-agnostic: no `.cuda()`/`.to(...)` call appears anywhere in this module --
placing the model (and its input tensors) on a given device is entirely the
caller's standard PyTorch responsibility.
"""

from __future__ import annotations

import torch
from torch import nn


class AutoencoderError(ValueError):
    """Raised for invalid Autoencoder configuration (a non-positive `input_dim`/
    `hidden_dim`/`bottleneck_dim`). Never silently corrected -- there is no
    fallback dimension substituted for an invalid one."""


class Autoencoder(nn.Module):
    """Dense encoder/decoder autoencoder for DSP feature vectors.

    Args:
        input_dim: number of features per input vector (the caller's TASK 5.3
            feature matrix column count) -- must be > 0.
        hidden_dim: width of the single intermediate dense layer in both encoder
            and decoder -- must be > 0.
        bottleneck_dim: width of the compressed representation between encoder
            and decoder -- must be > 0.

    Raises:
        AutoencoderError: if `input_dim`, `hidden_dim`, or `bottleneck_dim` is not
            a positive integer.
    """

    def __init__(self, input_dim: int, hidden_dim: int, bottleneck_dim: int) -> None:
        super().__init__()

        if input_dim <= 0:
            raise AutoencoderError(f"input_dim must be > 0, got {input_dim}")
        if hidden_dim <= 0:
            raise AutoencoderError(f"hidden_dim must be > 0, got {hidden_dim}")
        if bottleneck_dim <= 0:
            raise AutoencoderError(f"bottleneck_dim must be > 0, got {bottleneck_dim}")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes `x` to the bottleneck representation and decodes it back.

        Args:
            x: a batch of feature vectors, shape `(N, input_dim)` for any `N`.

        Returns:
            The reconstruction, shape `(N, input_dim)` -- identical shape to `x`.
            Does not modify `x` in place (every operation here is an out-of-place
            `nn.Linear`/`nn.ReLU` call).

        Raises PyTorch's own standard shape-mismatch error (not re-wrapped here)
        if `x`'s last dimension is not `input_dim` -- this method does not attempt
        to reshape/pad/truncate a mismatched input.
        """
        bottleneck = self.encoder(x)
        reconstruction = self.decoder(bottleneck)
        return reconstruction
