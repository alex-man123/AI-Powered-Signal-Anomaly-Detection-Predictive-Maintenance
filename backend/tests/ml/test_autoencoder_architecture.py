from __future__ import annotations

import pytest
import torch

from app.ml.autoencoder import Autoencoder, AutoencoderError


def _expected_parameter_count(input_dim: int, hidden_dim: int, bottleneck_dim: int) -> int:
    """Independent formula (not calling the model's own parameter-counting logic):
    each nn.Linear(a, b) contributes a*b weights + b biases."""
    encoder_1 = input_dim * hidden_dim + hidden_dim
    encoder_2 = hidden_dim * bottleneck_dim + bottleneck_dim
    decoder_1 = bottleneck_dim * hidden_dim + hidden_dim
    decoder_2 = hidden_dim * input_dim + input_dim
    return encoder_1 + encoder_2 + decoder_1 + decoder_2


# --- Test A (AC1) — forward shape across multiple batch sizes ---


@pytest.mark.parametrize("batch_size", [1, 4, 32, 100])
def test_ac1_forward_output_shape_matches_input_shape(batch_size: int) -> None:
    input_dim = 15
    model = Autoencoder(input_dim=input_dim, hidden_dim=8, bottleneck_dim=3)
    x = torch.randn(batch_size, input_dim)

    output = model(x)

    assert output.shape == (batch_size, input_dim)


# --- Test B — configurability across two distinct configurations ---


def test_configuration_a_forward_shape_matches_input() -> None:
    model = Autoencoder(input_dim=10, hidden_dim=6, bottleneck_dim=3)
    x = torch.randn(5, 10)

    output = model(x)

    assert output.shape == x.shape


def test_configuration_b_forward_shape_matches_input() -> None:
    model = Autoencoder(input_dim=20, hidden_dim=12, bottleneck_dim=4)
    x = torch.randn(5, 20)

    output = model(x)

    assert output.shape == x.shape


# --- Test C/D (AC2) — parameter count, independently computed, for two configs ---


def test_ac2_parameter_count_matches_independent_formula_config_a() -> None:
    input_dim, hidden_dim, bottleneck_dim = 10, 6, 3
    model = Autoencoder(input_dim=input_dim, hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim)

    actual_params = sum(p.numel() for p in model.parameters())
    expected_params = _expected_parameter_count(input_dim, hidden_dim, bottleneck_dim)

    print(f"\nConfig A (input={input_dim}, hidden={hidden_dim}, bottleneck={bottleneck_dim}): "
          f"{actual_params} total trainable parameters")

    assert actual_params == expected_params


def test_ac2_parameter_count_matches_independent_formula_config_b() -> None:
    input_dim, hidden_dim, bottleneck_dim = 20, 12, 4
    model = Autoencoder(input_dim=input_dim, hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim)

    actual_params = sum(p.numel() for p in model.parameters())
    expected_params = _expected_parameter_count(input_dim, hidden_dim, bottleneck_dim)

    print(f"\nConfig B (input={input_dim}, hidden={hidden_dim}, bottleneck={bottleneck_dim}): "
          f"{actual_params} total trainable parameters")

    assert actual_params == expected_params


def test_parameter_count_changes_correctly_between_configurations() -> None:
    """Demonstrates dimensions are not hard-coded: two different configurations
    must produce two different (correctly formula-derived) parameter counts."""
    model_a = Autoencoder(input_dim=10, hidden_dim=6, bottleneck_dim=3)
    model_b = Autoencoder(input_dim=20, hidden_dim=12, bottleneck_dim=4)

    params_a = sum(p.numel() for p in model_a.parameters())
    params_b = sum(p.numel() for p in model_b.parameters())

    assert params_a == _expected_parameter_count(10, 6, 3)
    assert params_b == _expected_parameter_count(20, 12, 4)
    assert params_a != params_b


# --- Test E — encoder dimensions ---


def test_encoder_layer_dimensions_match_configuration() -> None:
    input_dim, hidden_dim, bottleneck_dim = 15, 8, 3
    model = Autoencoder(input_dim=input_dim, hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim)

    first_linear = model.encoder[0]
    second_linear = model.encoder[2]

    assert isinstance(first_linear, torch.nn.Linear)
    assert first_linear.in_features == input_dim
    assert first_linear.out_features == hidden_dim

    assert isinstance(second_linear, torch.nn.Linear)
    assert second_linear.in_features == hidden_dim
    assert second_linear.out_features == bottleneck_dim


# --- Test F — decoder dimensions ---


def test_decoder_layer_dimensions_match_configuration_symmetrically() -> None:
    input_dim, hidden_dim, bottleneck_dim = 15, 8, 3
    model = Autoencoder(input_dim=input_dim, hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim)

    first_linear = model.decoder[0]
    second_linear = model.decoder[2]

    assert isinstance(first_linear, torch.nn.Linear)
    assert first_linear.in_features == bottleneck_dim
    assert first_linear.out_features == hidden_dim

    assert isinstance(second_linear, torch.nn.Linear)
    assert second_linear.in_features == hidden_dim
    assert second_linear.out_features == input_dim


# --- Test G — invalid dimensions ---


@pytest.mark.parametrize("input_dim,hidden_dim,bottleneck_dim", [
    (0, 8, 3),
    (-5, 8, 3),
    (15, 0, 3),
    (15, -1, 3),
    (15, 8, 0),
    (15, 8, -2),
])
def test_non_positive_dimensions_raise_explicitly(input_dim: int, hidden_dim: int, bottleneck_dim: int) -> None:
    with pytest.raises(AutoencoderError):
        Autoencoder(input_dim=input_dim, hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim)


# --- Test H — input immutability ---


def test_forward_does_not_modify_input_tensor() -> None:
    model = Autoencoder(input_dim=10, hidden_dim=6, bottleneck_dim=3)
    x = torch.randn(4, 10)
    x_before = x.clone()

    model(x)

    assert torch.equal(x, x_before)


# --- Test I — deterministic forward ---


def test_forward_is_deterministic_for_same_model_and_input() -> None:
    model = Autoencoder(input_dim=10, hidden_dim=6, bottleneck_dim=3)
    x = torch.randn(4, 10)

    output_1 = model(x)
    output_2 = model(x)

    assert torch.equal(output_1, output_2)


# --- Test J — wrong feature dimension ---


def test_forward_with_wrong_feature_dimension_raises_explicitly() -> None:
    model = Autoencoder(input_dim=15, hidden_dim=8, bottleneck_dim=3)
    wrong_shape_input = torch.randn(8, 10)  # 10 != input_dim=15

    with pytest.raises(RuntimeError):
        model(wrong_shape_input)


# --- additional: no forbidden components sneaked in ---


def test_architecture_contains_only_linear_and_relu_layers() -> None:
    """Confirms no BatchNorm/Dropout/convolutions were introduced -- keeps the
    architecture exactly as small/explainable as this task requires."""
    model = Autoencoder(input_dim=10, hidden_dim=6, bottleneck_dim=3)

    allowed_types = (Autoencoder, torch.nn.Linear, torch.nn.ReLU, torch.nn.Sequential)
    for module in model.modules():
        assert isinstance(module, allowed_types), f"unexpected layer type: {type(module)}"


def test_final_decoder_layer_has_no_activation_after_it() -> None:
    """The decoder Sequential must end with the Linear layer itself (no bounding
    activation like Sigmoid/Tanh after it), so it can reconstruct unbounded,
    zero-centered standardized feature values (TASK 5.4's StandardScaler output)."""
    model = Autoencoder(input_dim=10, hidden_dim=6, bottleneck_dim=3)

    last_layer = model.decoder[-1]
    assert isinstance(last_layer, torch.nn.Linear)
