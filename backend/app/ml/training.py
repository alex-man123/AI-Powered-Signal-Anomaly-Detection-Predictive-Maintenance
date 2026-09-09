"""TASK 7.2 — Autoencoder training loop: DataLoader, MSE reconstruction loss, Adam,
validation tracking, early stopping, best-checkpoint save/reload.

Blueprint.md does not define concrete learning_rate/batch_size/epochs/patience
values (section 16 only specifies the architecture, not training hyperparameters;
section 25 says seeds are fixed for numpy/torch/sklearn, without a specific seed
value) -- `TrainingConfig` below therefore exposes every hyperparameter as an
explicit, documented, configurable field, never hard-coded inline.

Pipeline this module owns (and nothing more):

    train feature matrix + train labels           validation feature matrix
              |                                              |
              v                                              |
      select_normal_samples()                                |
      (reused UNCHANGED from app.ml.isolation_forest --       |
       identical anti-leakage rule as TASK 6.2's own          |
       Isolation Forest training, not a second                |
       implementation of "keep only label=='normal'")         |
              |                                              |
              v                                              v
         DataLoader (shuffle, seeded generator)      single no-grad forward pass
              |                                              |
              v                                              v
      train_autoencoder(): for each epoch, gradient-update on the DataLoader's
      batches, then compute validation loss (no gradients), track both histories,
      apply early stopping on validation loss, keep the best state_dict, restore it
      into `model` at the end, optionally write a checkpoint to disk.

Reuse, not reimplementation:
- `app.ml.isolation_forest.select_normal_samples` is imported directly for the
  train-only-normal filter -- its exception type is named
  `IsolationForestTrainingError` (a naming artifact of it being written for TASK
  6.2 first), which is intentionally NOT re-wrapped here: it is the exact same
  business rule ("keep rows where label == SignalLabel.NORMAL.value", nothing
  Isolation-Forest-specific about the implementation itself), so raising a second,
  differently-named exception for identical semantics would only add confusion.
- `app.ml.autoencoder.Autoencoder` (TASK 7.1) is imported and trained as-is --
  its architecture is never copied or reimplemented here.
- Feature SCALING (TASK 5.4) is NOT performed here: `train_features`/
  `validation_features` are expected to already be the output of `app.ml.scaling.
  apply_scaler` (fit on train, applied to both) -- this module never calls
  `fit_scaler`/`apply_scaler` itself, and therefore can never accidentally refit a
  scaler on validation data.

CRITICAL data-leakage boundaries:
- Training gradient updates NEVER see anything but the normal-labeled subset of
  `train_features` (see `select_normal_samples` above) -- validated end-to-end by
  this module's own tests via a forward-pass spy, not just by inspecting the
  filter function in isolation.
- Validation data is used ONLY under `torch.no_grad()`, for computing a validation
  loss and for early-stopping decisions -- it never appears in a `loss.backward()`/
  `optimizer.step()` call anywhere in this module.
- There is no `test_features`/`test_labels` parameter anywhere in this module's
  public API -- test data is structurally impossible to pass into training or
  early stopping through this module (mirroring TASK 6.3's own
  `calibrate_threshold` design).

SEED NUANCE (read before writing a reproducibility test): `TrainingConfig.seed`
fixes `random`/`numpy`/`torch`'s global RNGs AND a dedicated `torch.Generator` used
for the DataLoader's shuffling order -- but a PyTorch `nn.Module`'s weights are
randomly initialized at CONSTRUCTION time (inside `Autoencoder.__init__`, called by
the CALLER before this function ever runs). Seeding inside `train_autoencoder`
therefore reproduces the TRAINING PROCESS (data order, any future stochastic
op) exactly, but does NOT retroactively change weights that were already
initialized before this function was called. For full end-to-end reproducibility
(TASK 7.2's AC4), the caller must seed (e.g. `torch.manual_seed(seed)`) BEFORE
constructing each `Autoencoder` instance being compared, in addition to passing
the same `config.seed` to `train_autoencoder` -- this module's own AC4 test does
exactly that.

Checkpoint vs. TASK 6.5's Model Artifact Contract: the checkpoint produced here
(`model_state_dict`/`optimizer_state_dict`/`epoch`/`best_validation_loss`, saved via
plain `torch.save`) is a TRAINING-time artifact, deliberately NOT the same thing as
TASK 6.5's `ModelArtifactMetadata` JSON-sidecar contract -- no second copy of that
schema is created here, and wiring an Autoencoder through TASK 6.5's contract
remains TASK 7.3's job.

Trust boundary: `load_checkpoint` deserializes via `torch.load`, which (like
`joblib`/`pickle`, per TASK 6.4's own documented boundary) can execute arbitrary
code for a maliciously crafted file. This module never loads a path it did not
receive explicitly from its caller.

Not implemented here (later tasks, out of this task's scope): reconstruction-error
anomaly scoring, threshold calibration, TASK 6.5 artifact wiring, API/frontend.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.ml.autoencoder import Autoencoder
from app.ml.isolation_forest import select_normal_samples


class AutoencoderTrainingError(ValueError):
    """Raised for invalid Autoencoder training input or a training-time failure:
    empty train/validation data, a feature-dimension mismatch against
    `model.input_dim`, an invalid `TrainingConfig` value, or a non-finite
    train/validation loss at any epoch. Never silently corrected -- there is no
    `nan_to_num`-style masking anywhere in this module, and training stops the
    instant a non-finite loss is observed rather than continuing with corrupted
    state."""


@dataclass(frozen=True)
class TrainingConfig:
    """Every Autoencoder training hyperparameter, explicit and configurable --
    blueprint.md defines no concrete values for any of these (see module
    docstring), so each default here is a simple, documented, non-"universal
    truth" choice:

    - `learning_rate=1e-3`: the standard default for `torch.optim.Adam` itself
      (Adam's own original paper's default), a reasonable starting point for a
      small dense network.
    - `batch_size=32`: a small, common default suitable for the modest dataset
      sizes this project's feature matrices produce.
    - `max_epochs=100`, `patience=10`: enough headroom for a small network to
      converge while still bounding worst-case training time; `patience=10`
      epochs without improvement is a common, easy-to-explain early-stopping
      default.
    - `min_delta=1e-4`: the minimum validation-loss improvement (in absolute
      terms) that counts as "actually better", so early stopping does not reset
      its patience counter on negligible floating-point noise.
    - `seed=42`: matches this project's own established seed convention (used
      throughout TASK 6.2/6.3's own tests and `data/processed/split_manifest.
      json`'s split seed).
    """

    learning_rate: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 100
    patience: int = 10
    min_delta: float = 1e-4
    seed: int = 42


@dataclass
class TrainingHistory:
    """Everything about one `train_autoencoder` run, accessible programmatically
    (never hidden behind printed log lines -- section 38's own requirement)."""

    train_losses: list[float]
    validation_losses: list[float]
    best_epoch: int
    best_validation_loss: float
    stopped_early: bool
    epochs_completed: int
    checkpoint_path: Path | None = field(default=None)


def _validate_config(config: TrainingConfig) -> None:
    if config.learning_rate <= 0:
        raise AutoencoderTrainingError(f"learning_rate must be > 0, got {config.learning_rate}")
    if config.batch_size <= 0:
        raise AutoencoderTrainingError(f"batch_size must be > 0, got {config.batch_size}")
    if config.max_epochs <= 0:
        raise AutoencoderTrainingError(f"max_epochs must be > 0, got {config.max_epochs}")
    if config.patience < 0:
        raise AutoencoderTrainingError(f"patience must be >= 0, got {config.patience}")
    if config.min_delta < 0:
        raise AutoencoderTrainingError(f"min_delta must be >= 0, got {config.min_delta}")


def _to_validated_tensor(features: pd.DataFrame | np.ndarray, *, name: str, input_dim: int) -> torch.Tensor:
    array = np.asarray(features, dtype=np.float32)

    if array.ndim != 2:
        raise AutoencoderTrainingError(
            f"{name} must be 2-dimensional (n_samples, n_features), got shape {array.shape}"
        )
    if array.shape[0] == 0:
        raise AutoencoderTrainingError(f"{name} must have at least one row (one window/sample)")
    if not np.isfinite(array).all():
        raise AutoencoderTrainingError(
            f"{name} contains NaN/Inf values -- refusing to silently mask them; fix the "
            "upstream (scaled) feature matrix instead"
        )
    if array.shape[1] != input_dim:
        raise AutoencoderTrainingError(
            f"{name} has {array.shape[1]} columns, but model.input_dim is {input_dim}"
        )

    return torch.from_numpy(array)


def train_autoencoder(
    model: Autoencoder,
    train_features: pd.DataFrame | np.ndarray,
    train_labels: Sequence[str],
    validation_features: pd.DataFrame | np.ndarray,
    *,
    config: TrainingConfig = TrainingConfig(),
    checkpoint_path: str | Path | None = None,
) -> TrainingHistory:
    """Trains `model` (in place) on the normal-labeled subset of `train_features`,
    tracking a separate validation loss every epoch and stopping early on
    validation-loss stagnation. See module docstring for the seed nuance, the
    checkpoint/TASK-6.5 separation, and the train/validation/test data boundaries.

    Args:
        model: an already-constructed `Autoencoder` (TASK 7.1) -- trained in
            place; at the end of this call, `model`'s weights are the BEST
            (lowest validation loss) weights observed, not necessarily the last
            epoch's.
        train_features: the TRAIN split's feature matrix (already scaled via TASK
            5.4's `apply_scaler`) -- may still contain non-normal rows; this
            function filters them out (see `select_normal_samples`).
        train_labels: one label per row of `train_features`, aligned
            positionally (same contract as TASK 6.2's `train_isolation_forest`).
        validation_features: the VALIDATION split's feature matrix (already
            scaled with the SAME train-fitted scaler) -- used AS GIVEN, never
            filtered to normal-only (this project's validation sets legitimately
            contain fault examples too, e.g. for TASK 6.3-style calibration
            elsewhere; this function does not presume otherwise).
        config: hyperparameters -- see `TrainingConfig`.
        checkpoint_path: if given, the best checkpoint (`model_state_dict`,
            `optimizer_state_dict`, `epoch`, `best_validation_loss`) is written
            here via `save_checkpoint` once, after training completes.

    Returns:
        A `TrainingHistory` with per-epoch train/validation loss lists (same
        length, one entry per completed epoch), the best epoch/validation loss,
        whether early stopping triggered, and the checkpoint path if one was
        written.

    Raises:
        AutoencoderTrainingError: for an invalid `config`; for
            `train_features`/`validation_features` that are empty, not
            2-dimensional, contain NaN/Inf, or whose column count does not match
            `model.input_dim`; if `select_normal_samples` finds zero
            normal-labeled rows (via `app.ml.isolation_forest.
            IsolationForestTrainingError` -- see module docstring for why this is
            not re-wrapped); or if any epoch's train/validation loss is
            non-finite.
    """
    _validate_config(config)

    normal_train_features = select_normal_samples(train_features, train_labels)

    train_tensor = _to_validated_tensor(
        normal_train_features, name="train_features (normal-only)", input_dim=model.input_dim
    )
    validation_tensor = _to_validated_tensor(
        validation_features, name="validation_features", input_dim=model.input_dim
    )

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)

    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()

    train_losses: list[float] = []
    validation_losses: list[float] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    best_model_state = copy.deepcopy(model.state_dict())
    best_optimizer_state = copy.deepcopy(optimizer.state_dict())
    patience_counter = 0
    stopped_early = False
    epochs_completed = 0

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_sample_count = 0
        for (batch,) in train_loader:
            optimizer.zero_grad()
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()

            batch_size = batch.shape[0]
            epoch_loss_sum += loss.item() * batch_size
            epoch_sample_count += batch_size

        train_loss = epoch_loss_sum / epoch_sample_count

        model.eval()
        with torch.no_grad():
            validation_reconstruction = model(validation_tensor)
            validation_loss = loss_fn(validation_reconstruction, validation_tensor).item()

        if not math.isfinite(train_loss) or not math.isfinite(validation_loss):
            raise AutoencoderTrainingError(
                f"Non-finite loss at epoch {epoch}: train_loss={train_loss}, "
                f"validation_loss={validation_loss} -- stopping immediately, not masking"
            )

        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        epochs_completed = epoch

        if validation_loss < best_validation_loss - config.min_delta:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            stopped_early = True
            break

    model.load_state_dict(best_model_state)

    resolved_checkpoint_path: Path | None = None
    if checkpoint_path is not None:
        resolved_checkpoint_path = save_checkpoint(
            checkpoint_path,
            model_state_dict=best_model_state,
            optimizer_state_dict=best_optimizer_state,
            epoch=best_epoch,
            best_validation_loss=best_validation_loss,
        )

    return TrainingHistory(
        train_losses=train_losses,
        validation_losses=validation_losses,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        stopped_early=stopped_early,
        epochs_completed=epochs_completed,
        checkpoint_path=resolved_checkpoint_path,
    )


def save_checkpoint(
    path: str | Path,
    *,
    model_state_dict: dict,
    optimizer_state_dict: dict,
    epoch: int,
    best_validation_loss: float,
) -> Path:
    """Writes a training checkpoint via `torch.save` -- NOT TASK 6.5's Model
    Artifact Contract (see module docstring). Creates parent directories if
    needed."""
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
            "epoch": epoch,
            "best_validation_loss": best_validation_loss,
        },
        resolved_path,
    )
    return resolved_path


def load_checkpoint(path: str | Path) -> dict:
    """Deserializes a checkpoint written by `save_checkpoint`. See module
    docstring's trust-boundary note -- only load a path this application itself
    produced via `save_checkpoint`."""
    return torch.load(Path(path), map_location="cpu")
