"""Self-contained TargetCall inference models.

This file only depends on PyTorch.  It does not import Bonito, read TOML, or
use the original Encoder/Block/TCSConv1d/Decoder wrappers.

Input layout:  [batch, 1, samples]
Output layout: [time, batch, 5] log probabilities
"""

import os

import torch
from torch import nn
from torch.nn import functional as F


ALPHABET = ["N", "A", "C", "G", "T"]

# out_channels, repeat, kernel, stride, dilation, dropout, separable, residual
DEFAULT_LAYER_SPECS = (
    (344, 1, 9, 3, 1, 0.05, False, False),
    (424, 2, 115, 1, 1, 0.05, True, True),
    (464, 7, 5, 1, 1, 0.05, True, True),
    (456, 4, 123, 1, 1, 0.05, True, True),
    (440, 9, 9, 1, 1, 0.05, True, True),
    (280, 6, 31, 1, 1, 0.05, True, True),
    (384, 1, 67, 1, 1, 0.05, True, False),
    (48, 1, 15, 1, 1, 0.05, False, False),
)


def _tiny_layer_specs(stem, wide, middle, narrow, output):
    """Return the fixed LightCall topology for a set of channel widths."""
    return (
        (stem, 1, 9, 3, 1, 0.05, False, False),
        (wide, 2, 75, 1, 1, 0.05, True, False),
        (wide, 1, 31, 1, 1, 0.05, True, False),
        (wide, 2, 31, 1, 1, 0.05, True, False),
        (middle, 1, 123, 1, 1, 0.05, True, False),
        (middle, 2, 55, 1, 1, 0.05, True, False),
        (middle, 2, 5, 1, 1, 0.05, True, False),
        (middle, 2, 3, 1, 1, 0.05, True, False),
        (middle, 2, 9, 1, 1, 0.05, True, False),
        (middle, 2, 115, 1, 1, 0.05, True, False),
        (middle, 2, 55, 1, 1, 0.05, True, False),
        (narrow, 2, 25, 1, 1, 0.05, True, False),
        (narrow, 2, 9, 1, 1, 0.05, True, False),
        (narrow, 1, 25, 1, 1, 0.05, True, False),
        (narrow, 2, 7, 1, 1, 0.05, True, False),
        (narrow, 1, 123, 1, 1, 0.05, True, False),
        (narrow, 1, 9, 1, 1, 0.05, True, False),
        (output, 1, 9, 1, 1, 0.05, False, False),
    )


TINYX0111_LAYER_SPECS = _tiny_layer_specs(48, 104, 120, 128, 48)
TINYX011_LAYER_SPECS = _tiny_layer_specs(48, 104, 80, 64, 48)
TINYX01_LAYER_SPECS = _tiny_layer_specs(48, 80, 48, 32, 48)
TINYX2_LAYER_SPECS = _tiny_layer_specs(24, 40, 24, 16, 24)
TINYX3_LAYER_SPECS = _tiny_layer_specs(12, 20, 12, 8, 12)


def _register_layer(model, name, layer, legacy_name):
    """Register a primitive layer and remember its original weight names."""
    model.add_module(name, layer)
    for key in layer.state_dict():
        model._legacy_state_keys[f"{name}.{key}"] = f"{legacy_name}.{key}"


def _add_main_step(
    model,
    layer_number,
    in_channels,
    out_channels,
    kernel,
    stride,
    dilation,
    separable,
    groups,
    legacy_layer_name,
    legacy_batch_norm_name,
):
    padding = (kernel // 2) * dilation

    if separable:
        depthwise_name = f"depthwise{layer_number}"
        pointwise_name = f"pointwise{layer_number}"
        _register_layer(
            model,
            depthwise_name,
            nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            f"{legacy_layer_name}.depthwise",
        )
        _register_layer(
            model,
            pointwise_name,
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1,
                dilation=dilation,
                bias=False,
            ),
            f"{legacy_layer_name}.pointwise",
        )
        step = (depthwise_name, pointwise_name, None)
    else:
        conv_name = f"conv{layer_number}"
        _register_layer(
            model,
            conv_name,
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            f"{legacy_layer_name}.conv",
        )
        step = (None, None, conv_name)

    batch_norm_name = f"bn{layer_number}"
    _register_layer(
        model,
        batch_norm_name,
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1),
        legacy_batch_norm_name,
    )
    return step + (batch_norm_name,)


def _add_residual(model, block_number, in_channels, out_channels):
    conv_name = f"residual_conv{block_number}"
    batch_norm_name = f"residual_bn{block_number}"
    legacy_prefix = f"encoder.encoder.{block_number}.residual"

    _register_layer(
        model,
        conv_name,
        nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
        f"{legacy_prefix}.0.conv",
    )
    _register_layer(
        model,
        batch_norm_name,
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1),
        f"{legacy_prefix}.1",
    )
    return conv_name, batch_norm_name


def _initialize_model(model, layer_specs, depthwise_group_divisor):
    model.alphabet = list(ALPHABET)
    model.stride = layer_specs[0][3]
    model.features = layer_specs[-1][0]
    model.qbias = 0.0
    model.qscale = 1.0
    model._legacy_state_keys = {}
    model._execution_plan = []

    in_channels = 1
    layer_number = 0
    for block_number, spec in enumerate(layer_specs):
        out_channels, repeat, kernel, stride, dilation, dropout, separable, residual = spec
        block_input_channels = in_channels
        steps = []

        for repeat_number in range(repeat):
            groups = in_channels // depthwise_group_divisor if separable else 1
            legacy_layer_number = repeat_number * 4
            legacy_block_name = f"encoder.encoder.{block_number}.conv"
            steps.append(
                _add_main_step(
                    model,
                    layer_number,
                    in_channels,
                    out_channels,
                    kernel,
                    stride,
                    dilation,
                    separable,
                    groups,
                    f"{legacy_block_name}.{legacy_layer_number}",
                    f"{legacy_block_name}.{legacy_layer_number + 1}",
                )
            )
            in_channels = out_channels
            layer_number += 1

        residual_layers = None
        if residual:
            residual_layers = _add_residual(
                model,
                block_number,
                block_input_channels,
                out_channels,
            )

        model._execution_plan.append((steps, residual_layers, dropout))

    _register_layer(
        model,
        "decoder",
        nn.Conv1d(model.features, len(model.alphabet), kernel_size=1, bias=True),
        "decoder.layers.0",
    )


def _forward(model, signal):
    x = signal
    for steps, residual_layers, dropout in model._execution_plan:
        block_input = x
        for step_number, step in enumerate(steps):
            depthwise_name, pointwise_name, conv_name, batch_norm_name = step
            if depthwise_name is not None:
                x = getattr(model, depthwise_name)(x)
                x = getattr(model, pointwise_name)(x)
            else:
                x = getattr(model, conv_name)(x)

            x = getattr(model, batch_norm_name)(x)
            if step_number < len(steps) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=dropout, training=model.training)

        if residual_layers is not None:
            residual_conv_name, residual_batch_norm_name = residual_layers
            residual = getattr(model, residual_conv_name)(block_input)
            residual = getattr(model, residual_batch_norm_name)(residual)
            x = x + residual

        x = F.relu(x)
        x = F.dropout(x, p=dropout, training=model.training)

    x = model.decoder(x)
    x = x.permute(2, 0, 1)
    return F.log_softmax(x, dim=-1)


class DefaultModel(nn.Module):
    def __init__(self):
        super().__init__()
        _initialize_model(self, DEFAULT_LAYER_SPECS, depthwise_group_divisor=8)

    def forward(self, signal):
        return _forward(self, signal)


class TinyX0111Model(nn.Module):
    def __init__(self):
        super().__init__()
        _initialize_model(self, TINYX0111_LAYER_SPECS, depthwise_group_divisor=1)

    def forward(self, signal):
        return _forward(self, signal)


class TinyX011Model(nn.Module):
    def __init__(self):
        super().__init__()
        _initialize_model(self, TINYX011_LAYER_SPECS, depthwise_group_divisor=1)

    def forward(self, signal):
        return _forward(self, signal)


class TinyX01Model(nn.Module):
    def __init__(self):
        super().__init__()
        _initialize_model(self, TINYX01_LAYER_SPECS, depthwise_group_divisor=1)

    def forward(self, signal):
        return _forward(self, signal)


class TinyX2Model(nn.Module):
    def __init__(self):
        super().__init__()
        _initialize_model(self, TINYX2_LAYER_SPECS, depthwise_group_divisor=1)

    def forward(self, signal):
        return _forward(self, signal)


class TinyX3Model(nn.Module):
    def __init__(self):
        super().__init__()
        _initialize_model(self, TINYX3_LAYER_SPECS, depthwise_group_divisor=1)

    def forward(self, signal):
        return _forward(self, signal)


MODEL_CLASSES = {
    "default": DefaultModel,
    "TINYX0111": TinyX0111Model,
    "TINYX011": TinyX011Model,
    "TINYX01": TinyX01Model,
    "TINYX2": TinyX2Model,
    "TINYX3": TinyX3Model,
}


def create_model(name="default"):
    """Construct a model without reading configuration or weight files."""
    try:
        model_class = MODEL_CLASSES[name]
    except KeyError as error:
        available = ", ".join(MODEL_CLASSES)
        raise ValueError(f"unknown model '{name}'; choose from: {available}") from error
    return model_class()


def _checkpoint_state(source, map_location):
    if isinstance(source, (str, bytes, os.PathLike)):
        state = torch.load(source, map_location=map_location)
    else:
        state = source

    if "model_state_dict" in state:
        state = state["model_state_dict"]
    elif "state_dict" in state:
        state = state["state_dict"]

    return {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state.items()
    }


def load_weights(model, source, map_location="cpu", strict=True):
    """Load either original TargetCall weights or already-flattened weights."""
    state = _checkpoint_state(source, map_location)
    model_keys = set(model.state_dict())

    if set(state) == model_keys:
        model.load_state_dict(state, strict=True)
        return model

    converted = {}
    missing = []
    for new_key, legacy_key in model._legacy_state_keys.items():
        if legacy_key not in state:
            missing.append(legacy_key)
        else:
            converted[new_key] = state[legacy_key]

    if missing:
        raise RuntimeError(f"missing legacy weight keys: {missing}")

    if strict:
        expected = set(model._legacy_state_keys.values())
        unexpected = sorted(set(state) - expected)
        if unexpected:
            raise RuntimeError(f"unexpected legacy weight keys: {unexpected}")

    model.load_state_dict(converted, strict=True)
    return model


def load_model(name, weights, device="cpu"):
    """Construct, load, and prepare a model for inference."""
    model = create_model(name)
    load_weights(model, weights, map_location=device)
    model.to(device)
    model.eval()
    return model


__all__ = [
    "DefaultModel",
    "TinyX0111Model",
    "TinyX011Model",
    "TinyX01Model",
    "TinyX2Model",
    "TinyX3Model",
    "MODEL_CLASSES",
    "create_model",
    "load_weights",
    "load_model",
]
