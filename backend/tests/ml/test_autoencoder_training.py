from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from app.datasets.loader import DATASET_ROOT as REAL_DATASET_ROOT
from app.datasets.loader import load_all_splits
from app.datasets.validators import SAMPLING_RATE_HZ
from app.features.extractor import extract_feature_matrix
from app.ml import training as training_module
from app.ml.autoencoder import Autoencoder
from app.ml.isolation_forest import IsolationForestTrainingError
from app.ml.scaling import apply_scaler, fit_scaler
from app.ml.training import (
    AutoencoderTrainingError,
    TrainingConfig,
    load_checkpoint,
    save_checkpoint,
    train_autoencoder,
)
from app.signal_processing.windowing import Window, create_windows


def _learnable_dataset(n_train: int = 300, n_val: int = 60, num_features: int = 6, seed: int = 0):
    """2 latent factors linearly mixed into `num_features` observed features, plus
    tiny noise -- deterministic and easy for a bottleneck_dim=2 autoencoder to
    reconstruct well, so AC1's convergence claim is a real, non-marginal effect,
    not a coincidence of an unlearnable random dataset."""
    rng = np.random.default_rng(seed)
    mixing = rng.normal(0, 1, size=(2, num_features))

    train_latent = rng.normal(0, 1, size=(n_train, 2))
    train_data = train_latent @ mixing + rng.normal(0, 0.01, size=(n_train, num_features))
    train_features = pd.DataFrame(train_data, columns=[f"f{i}" for i in range(num_features)])

    val_latent = rng.normal(0, 1, size=(n_val, 2))
    val_data = val_latent @ mixing + rng.normal(0, 0.01, size=(n_val, num_features))
    validation_features = pd.DataFrame(val_data, columns=[f"f{i}" for i in range(num_features)])

    train_labels = ["normal"] * n_train
    return train_features, train_labels, validation_features


# --- AC1: convergence, no NaN/Inf ---


def test_ac1_final_train_loss_significantly_below_initial_train_loss() -> None:
    train_features, train_labels, validation_features = _learnable_dataset()
    torch.manual_seed(42)
    model = Autoencoder(input_dim=6, hidden_dim=5, bottleneck_dim=2)

    history = train_autoencoder(
        model,
        train_features,
        train_labels,
        validation_features,
        config=TrainingConfig(max_epochs=60, patience=60, learning_rate=1e-2, seed=42),
    )

    print(f"\ninitial train_loss={history.train_losses[0]:.6f} final train_loss={history.train_losses[-1]:.6f}")

    # Not marginal -- a real, learnable dataset should let the autoencoder reduce
    # loss by at least 10x, not just "slightly lower".
    assert history.train_losses[-1] < 0.1 * history.train_losses[0]


def test_ac1_no_nan_or_inf_across_entire_training_history() -> None:
    train_features, train_labels, validation_features = _learnable_dataset()
    torch.manual_seed(42)
    model = Autoencoder(input_dim=6, hidden_dim=5, bottleneck_dim=2)

    history = train_autoencoder(
        model,
        train_features,
        train_labels,
        validation_features,
        config=TrainingConfig(max_epochs=30, patience=30, seed=42),
    )

    assert all(np.isfinite(loss) for loss in history.train_losses)
    assert all(np.isfinite(loss) for loss in history.validation_losses)


def test_convergence_is_not_required_to_be_monotonic() -> None:
    """AC1 explicitly does not require monotonic decrease -- confirms the
    implementation doesn't reject/artificially smooth a fluctuating loss curve."""
    train_features, train_labels, validation_features = _learnable_dataset(seed=3)
    torch.manual_seed(7)
    model = Autoencoder(input_dim=6, hidden_dim=5, bottleneck_dim=2)

    history = train_autoencoder(
        model,
        train_features,
        train_labels,
        validation_features,
        config=TrainingConfig(max_epochs=40, patience=40, learning_rate=3e-2, seed=7),
    )

    non_monotonic_steps = sum(
        1 for i in range(len(history.train_losses) - 1) if history.train_losses[i + 1] > history.train_losses[i]
    )
    print(f"\nnon-monotonic epoch-to-epoch steps: {non_monotonic_steps} / {len(history.train_losses) - 1}")
    # The model is still perfectly valid regardless of this count -- this test only
    # documents that fluctuation happens and training tolerates it (no assertion
    # requiring zero fluctuation).
    assert history.epochs_completed == 40


# --- AC2: validation tracked separately, every epoch ---


def test_ac2_train_and_validation_histories_have_equal_length_per_epoch() -> None:
    train_features, train_labels, validation_features = _learnable_dataset()
    torch.manual_seed(42)
    model = Autoencoder(input_dim=6, hidden_dim=5, bottleneck_dim=2)

    history = train_autoencoder(
        model,
        train_features,
        train_labels,
        validation_features,
        config=TrainingConfig(max_epochs=15, patience=15, seed=42),
    )

    assert len(history.train_losses) == len(history.validation_losses) == history.epochs_completed == 15


def test_ac2_validation_forward_pass_does_not_change_model_parameters() -> None:
    """Validation must never trigger a gradient update -- verified by training for
    exactly one epoch on a single training batch, then confirming that swapping in
    a much larger validation set (more forward-pass work) does not change the
    resulting trained parameters at all (only the recorded validation_loss would
    differ)."""
    train_features, train_labels, small_validation_features = _learnable_dataset(n_train=32, n_val=1)
    _, _, big_validation_features = _learnable_dataset(n_train=32, n_val=500, seed=99)

    torch.manual_seed(1)
    model_small_val = Autoencoder(input_dim=6, hidden_dim=5, bottleneck_dim=2)
    train_autoencoder(
        model_small_val,
        train_features,
        train_labels,
        small_validation_features,
        config=TrainingConfig(max_epochs=1, patience=1, batch_size=32, seed=1),
    )

    torch.manual_seed(1)
    model_big_val = Autoencoder(input_dim=6, hidden_dim=5, bottleneck_dim=2)
    train_autoencoder(
        model_big_val,
        train_features,
        train_labels,
        big_validation_features,
        config=TrainingConfig(max_epochs=1, patience=1, batch_size=32, seed=1),
    )

    for p1, p2 in zip(model_small_val.parameters(), model_big_val.parameters()):
        torch.testing.assert_close(p1, p2)


# --- AC3 (CRITICAL): no non-normal example ever reaches a training batch ---


def test_ac3_only_normal_labeled_rows_reach_training_batches(monkeypatch) -> None:
    rng = np.random.default_rng(0)
    normal = pd.DataFrame(rng.normal(0.0, 1.0, size=(40, 4)), columns=["a", "b", "c", "d"])
    fault_high = pd.DataFrame(rng.normal(500.0, 1.0, size=(5, 4)), columns=["a", "b", "c", "d"])
    fault_low = pd.DataFrame(rng.normal(-500.0, 1.0, size=(5, 4)), columns=["a", "b", "c", "d"])
    features = pd.concat([normal, fault_high, fault_low], ignore_index=True)
    labels = ["normal"] * 40 + ["horizontal-misalignment"] * 5 + ["imbalance"] * 5
    validation_features = pd.DataFrame(rng.normal(0, 1, size=(10, 4)), columns=["a", "b", "c", "d"])

    captured_training_batches: list[torch.Tensor] = []
    original_forward = Autoencoder.forward

    def spy_forward(self, x):
        if self.training:
            captured_training_batches.append(x.detach().clone())
        return original_forward(self, x)

    monkeypatch.setattr(Autoencoder, "forward", spy_forward)

    torch.manual_seed(42)
    model = Autoencoder(input_dim=4, hidden_dim=4, bottleneck_dim=2)
    train_autoencoder(
        model,
        features,
        labels,
        validation_features,
        config=TrainingConfig(max_epochs=5, patience=5, batch_size=8, seed=42),
    )

    assert len(captured_training_batches) > 0, "no training batches were captured"

    all_seen_rows = torch.cat(captured_training_batches, dim=0)
    normal_tensor = torch.tensor(normal.to_numpy(dtype=np.float32))

    for row in all_seen_rows:
        matches_a_normal_row = torch.any(torch.all(torch.isclose(normal_tensor, row, atol=1e-5), dim=1))
        assert bool(matches_a_normal_row), f"a non-normal row reached a training batch: {row}"

    # Extra, human-checkable margin: no seen value is anywhere near the fault
    # clusters (+-500), which would be immediately obvious if leakage occurred.
    assert torch.all(all_seen_rows.abs() < 10.0)


def test_ac3_features_labels_length_mismatch_propagates_from_select_normal_samples() -> None:
    train_features, train_labels, validation_features = _learnable_dataset(n_train=10)
    torch.manual_seed(1)
    model = Autoencoder(input_dim=6, hidden_dim=4, bottleneck_dim=2)

    with pytest.raises(IsolationForestTrainingError, match="rows"):
        train_autoencoder(
            model, train_features, train_labels[:-1], validation_features, config=TrainingConfig(max_epochs=1)
        )


# --- AC4: reproducibility ---


def test_ac4_same_seed_produces_identical_loss_curves() -> None:
    train_features, train_labels, validation_features = _learnable_dataset(seed=5)
    config = TrainingConfig(max_epochs=10, patience=10, seed=42)

    def run() -> tuple[list[float], list[float]]:
        torch.manual_seed(42)  # seeds model weight initialization too -- see module docstring
        model = Autoencoder(input_dim=6, hidden_dim=5, bottleneck_dim=2)
        history = train_autoencoder(model, train_features, train_labels, validation_features, config=config)
        return history.train_losses, history.validation_losses

    train_a, val_a = run()
    train_b, val_b = run()

    # Bit-exact on CPU -- verified empirically before writing this test.
    assert train_a == train_b
    assert val_a == val_b


def test_different_seeds_can_produce_different_curves() -> None:
    """Sanity check that seed actually matters (rules out a vacuous AC4 pass)."""
    train_features, train_labels, validation_features = _learnable_dataset(seed=5)

    torch.manual_seed(1)
    model_a = Autoencoder(input_dim=6, hidden_dim=5, bottleneck_dim=2)
    history_a = train_autoencoder(
        model_a, train_features, train_labels, validation_features, config=TrainingConfig(max_epochs=5, seed=1)
    )

    torch.manual_seed(2)
    model_b = Autoencoder(input_dim=6, hidden_dim=5, bottleneck_dim=2)
    history_b = train_autoencoder(
        model_b, train_features, train_labels, validation_features, config=TrainingConfig(max_epochs=5, seed=2)
    )

    assert history_a.train_losses != history_b.train_losses


# --- early stopping ---


def test_early_stopping_triggers_before_max_epochs_with_unreachable_min_delta() -> None:
    """A min_delta far larger than any achievable improvement forces every epoch
    after the first to count as "no improvement" -- a fully deterministic way to
    force early stopping, rather than hoping an organic plateau happens."""
    train_features, train_labels, validation_features = _learnable_dataset()
    torch.manual_seed(42)
    model = Autoencoder(input_dim=6, hidden_dim=5, bottleneck_dim=2)

    history = train_autoencoder(
        model,
        train_features,
        train_labels,
        validation_features,
        config=TrainingConfig(max_epochs=50, patience=3, min_delta=1000.0, seed=42),
    )

    assert history.stopped_early is True
    assert history.epochs_completed < 50
    # patience=3: stops 3 epochs after epoch 1 (the only one that can "improve"
    # against the initial infinite baseline).
    assert history.epochs_completed == 4
    assert history.best_epoch == 1


def test_best_checkpoint_corresponds_to_lowest_validation_loss() -> None:
    train_features, train_labels, validation_features = _learnable_dataset()
    torch.manual_seed(42)
    model = Autoencoder(input_dim=6, hidden_dim=5, bottleneck_dim=2)

    history = train_autoencoder(
        model,
        train_features,
        train_labels,
        validation_features,
        config=TrainingConfig(max_epochs=30, patience=30, seed=42),
    )

    assert history.best_validation_loss == min(history.validation_losses)
    assert history.validation_losses[history.best_epoch - 1] == pytest.approx(history.best_validation_loss)


# --- checkpoint save/reload ---


def test_checkpoint_is_saved_and_reload_produces_identical_forward_output(tmp_path: Path) -> None:
    train_features, train_labels, validation_features = _learnable_dataset()
    torch.manual_seed(42)
    model = Autoencoder(input_dim=6, hidden_dim=5, bottleneck_dim=2)

    checkpoint_path = tmp_path / "autoencoder_checkpoint.pt"
    history = train_autoencoder(
        model,
        train_features,
        train_labels,
        validation_features,
        config=TrainingConfig(max_epochs=10, patience=10, seed=42),
        checkpoint_path=checkpoint_path,
    )

    assert history.checkpoint_path == checkpoint_path
    assert checkpoint_path.exists()
    assert checkpoint_path.stat().st_size > 0

    checkpoint = load_checkpoint(checkpoint_path)
    assert checkpoint["epoch"] == history.best_epoch
    assert checkpoint["best_validation_loss"] == pytest.approx(history.best_validation_loss)
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint

    reloaded_model = Autoencoder(input_dim=6, hidden_dim=5, bottleneck_dim=2)
    reloaded_model.load_state_dict(checkpoint["model_state_dict"])
    reloaded_model.eval()
    model.eval()

    x = torch.randn(5, 6)
    with torch.no_grad():
        output_original = model(x)
        output_reloaded = reloaded_model(x)

    torch.testing.assert_close(output_original, output_reloaded)


def test_save_checkpoint_creates_parent_directories(tmp_path: Path) -> None:
    nested_path = tmp_path / "a" / "b" / "checkpoint.pt"
    result = save_checkpoint(
        nested_path, model_state_dict={}, optimizer_state_dict={}, epoch=1, best_validation_loss=0.5
    )
    assert result == nested_path
    assert nested_path.exists()


# --- feature dimension mismatch ---


def test_feature_dimension_mismatch_in_train_features_raises_explicitly() -> None:
    train_features, train_labels, validation_features = _learnable_dataset(num_features=6)
    torch.manual_seed(1)
    model = Autoencoder(input_dim=5, hidden_dim=4, bottleneck_dim=2)  # expects 5, data has 6

    with pytest.raises(AutoencoderTrainingError, match="columns"):
        train_autoencoder(model, train_features, train_labels, validation_features, config=TrainingConfig(max_epochs=1))


def test_feature_dimension_mismatch_in_validation_features_raises_explicitly() -> None:
    train_features, train_labels, _ = _learnable_dataset(num_features=6)
    wrong_validation = pd.DataFrame(np.random.default_rng(0).normal(0, 1, size=(10, 4)))
    torch.manual_seed(1)
    model = Autoencoder(input_dim=6, hidden_dim=4, bottleneck_dim=2)

    with pytest.raises(AutoencoderTrainingError, match="columns"):
        train_autoencoder(model, train_features, train_labels, wrong_validation, config=TrainingConfig(max_epochs=1))


# --- empty data ---


def test_empty_training_data_raises_explicitly() -> None:
    """An all-empty (features, labels) input has zero normal-labeled rows by
    construction, so `select_normal_samples` (reused from TASK 6.2) raises its own
    "no normal samples" error before this function's own empty-row check ever gets
    a chance to run -- that is the correct, more informative diagnosis for this
    exact input, not a bug to work around."""
    torch.manual_seed(1)
    model = Autoencoder(input_dim=4, hidden_dim=4, bottleneck_dim=2)
    empty_features = pd.DataFrame(np.empty((0, 4)), columns=["a", "b", "c", "d"])
    validation_features = pd.DataFrame(np.random.default_rng(0).normal(0, 1, size=(5, 4)))

    with pytest.raises(IsolationForestTrainingError, match="normal"):
        train_autoencoder(model, empty_features, [], validation_features, config=TrainingConfig(max_epochs=1))


def test_all_non_normal_training_labels_raises_explicitly() -> None:
    torch.manual_seed(1)
    model = Autoencoder(input_dim=4, hidden_dim=4, bottleneck_dim=2)
    features = pd.DataFrame(np.random.default_rng(0).normal(0, 1, size=(5, 4)))
    labels = ["horizontal-misalignment"] * 5
    validation_features = pd.DataFrame(np.random.default_rng(1).normal(0, 1, size=(5, 4)))

    with pytest.raises(IsolationForestTrainingError, match="normal"):
        train_autoencoder(model, features, labels, validation_features, config=TrainingConfig(max_epochs=1))


def test_empty_validation_data_raises_explicitly() -> None:
    train_features, train_labels, _ = _learnable_dataset()
    empty_validation = pd.DataFrame(np.empty((0, 6)), columns=[f"f{i}" for i in range(6)])
    torch.manual_seed(1)
    model = Autoencoder(input_dim=6, hidden_dim=4, bottleneck_dim=2)

    with pytest.raises(AutoencoderTrainingError, match="row"):
        train_autoencoder(model, train_features, train_labels, empty_validation, config=TrainingConfig(max_epochs=1))


# --- config validation ---


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"learning_rate": 0}, "learning_rate"),
        ({"learning_rate": -1e-3}, "learning_rate"),
        ({"batch_size": 0}, "batch_size"),
        ({"max_epochs": 0}, "max_epochs"),
        ({"patience": -1}, "patience"),
        ({"min_delta": -0.1}, "min_delta"),
    ],
)
def test_invalid_config_raises_explicitly(kwargs: dict, match: str) -> None:
    train_features, train_labels, validation_features = _learnable_dataset(n_train=10)
    torch.manual_seed(1)
    model = Autoencoder(input_dim=6, hidden_dim=4, bottleneck_dim=2)

    with pytest.raises(AutoencoderTrainingError, match=match):
        train_autoencoder(model, train_features, train_labels, validation_features, config=TrainingConfig(**kwargs))


# --- test set is structurally impossible to pass in ---


def test_no_test_data_parameter_exists_in_train_autoencoder_signature() -> None:
    for name in inspect.signature(train_autoencoder).parameters:
        assert "test" not in name.lower(), f"train_autoencoder must not accept a {name!r} parameter"


# --- scaling: training.py must not fit/refit a scaler itself ---


def test_training_module_does_not_call_fit_scaler_or_fit_transform() -> None:
    """Static source check: training.py must never call fit_scaler/fit_transform
    itself -- it consumes already-scaled features (TASK 5.4's own contract)."""
    source = inspect.getsource(training_module)
    assert "fit_scaler(" not in source
    assert "fit_transform(" not in source
    assert ".fit(" not in source  # no scaler .fit() call anywhere in this module


def test_training_module_does_not_import_scaling_module() -> None:
    tree = ast.parse(inspect.getsource(training_module))
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)
        elif isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)

    assert not any("app.ml.scaling" in name for name in imported_modules)


# --- input immutability ---


def test_train_autoencoder_does_not_modify_input_features_or_labels() -> None:
    train_features, train_labels, validation_features = _learnable_dataset(n_train=20, n_val=5)
    train_snapshot = train_features.copy()
    validation_snapshot = validation_features.copy()
    labels_snapshot = list(train_labels)

    torch.manual_seed(1)
    model = Autoencoder(input_dim=6, hidden_dim=4, bottleneck_dim=2)
    train_autoencoder(
        model, train_features, train_labels, validation_features, config=TrainingConfig(max_epochs=2, seed=1)
    )

    pd.testing.assert_frame_equal(train_features, train_snapshot)
    pd.testing.assert_frame_equal(validation_features, validation_snapshot)
    assert train_labels == labels_snapshot


# --- real-data sanity check ---
#
# Real MAFAULDA subset: 2 normal train recordings (matching TASK 6.2's own choice),
# 1 normal validation recording. From data/processed/split_manifest.json's existing
# lists -- not a new split, not a full dataset scan.
REAL_TRAIN_NORMAL = ["normal/12.288.csv", "normal/16.1792.csv"]
REAL_VALIDATION_NORMAL = ["normal/14.336.csv"]
REAL_WINDOW_SIZE = 1024
REAL_OVERLAP = 0.5
REAL_CHANNEL = 0
REAL_NPERSEG = 256
REAL_NOVERLAP = 128


def _write_manifest(tmp_path: Path, splits: dict[str, list[str]]) -> Path:
    manifest_path = tmp_path / "split_manifest.json"
    manifest_path.write_text(json.dumps({"splits": splits}), encoding="utf-8")
    return manifest_path


def test_real_data_training_pipeline_completes_without_leakage(tmp_path_factory: pytest.TempPathFactory) -> None:
    tmp_path = tmp_path_factory.mktemp("autoencoder_real_data")
    manifest_path = _write_manifest(
        tmp_path, {"train": REAL_TRAIN_NORMAL, "validation": REAL_VALIDATION_NORMAL, "test": []}
    )

    loaded = load_all_splits(dataset_root=REAL_DATASET_ROOT, manifest_path=manifest_path)

    def _windows_and_labels(split_name: str) -> tuple[list[Window], list[str]]:
        windows: list[Window] = []
        labels: list[str] = []
        for recording in loaded[split_name]:
            df = pd.read_csv(REAL_DATASET_ROOT / recording.relative_path, header=None)
            values = df[REAL_CHANNEL].tolist()
            recording_windows = create_windows(
                values,
                recording_id=recording.relative_path,
                split=recording.split,
                window_size=REAL_WINDOW_SIZE,
                overlap=REAL_OVERLAP,
            )
            windows.extend(recording_windows)
            labels.extend([recording.label.value] * len(recording_windows))
        return windows, labels

    train_windows, train_labels = _windows_and_labels("train")
    validation_windows, _validation_labels = _windows_and_labels("validation")

    train_matrix = extract_feature_matrix(
        train_windows, SAMPLING_RATE_HZ, nperseg=REAL_NPERSEG, noverlap=REAL_NOVERLAP
    )
    validation_matrix = extract_feature_matrix(
        validation_windows, SAMPLING_RATE_HZ, nperseg=REAL_NPERSEG, noverlap=REAL_NOVERLAP
    )

    # TASK 5.4 scaling: fit on train only, applied to both -- never refit here.
    scaler = fit_scaler(train_matrix)
    scaled_train = apply_scaler(scaler, train_matrix)
    scaled_validation = apply_scaler(scaler, validation_matrix)

    assert "normal" in train_labels
    assert all(label == "normal" for label in train_labels), "test setup error: expected an all-normal real subset"

    torch.manual_seed(42)
    model = Autoencoder(input_dim=scaled_train.shape[1], hidden_dim=10, bottleneck_dim=4)

    history = train_autoencoder(
        model,
        scaled_train,
        train_labels,
        scaled_validation,
        config=TrainingConfig(max_epochs=10, patience=10, seed=42),
    )

    assert history.epochs_completed == 10
    assert all(np.isfinite(loss) for loss in history.train_losses)
    assert all(np.isfinite(loss) for loss in history.validation_losses)
