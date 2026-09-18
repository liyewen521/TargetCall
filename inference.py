"""Standalone, explicit TargetCall inference models.

Each model lives in its own module and declares every convolution and
batch-norm layer explicitly; there are no block wrappers or generated layer
specs. The package only depends on PyTorch and loads flattened weights. Use
``convert_legacy.py`` to turn the original TargetCall checkpoints into that
flattened form.
"""

import os

import torch

from default import DefaultModel
from tinyx0111 import TinyX0111Model
from tinyx011 import TinyX011Model
from tinyx01 import TinyX01Model
from tinyx2 import TinyX2Model
from tinyx3 import TinyX3Model


MODEL_CLASSES = {
    "default": DefaultModel,
    "TINYX0111": TinyX0111Model,
    "TINYX011": TinyX011Model,
    "TINYX01": TinyX01Model,
    "TINYX2": TinyX2Model,
    "TINYX3": TinyX3Model,
}

WEIGHTS_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")


def create_model(name="default"):
    """Construct a model without loading any weights."""
    try:
        model_class = MODEL_CLASSES[name]
    except KeyError as error:
        available = ", ".join(MODEL_CLASSES)
        raise ValueError(f"unknown model '{name}'; choose from: {available}") from error
    return model_class()


def flatten_state_dict(state):
    """Strip wrapper prefixes from a checkpoint's state dict."""
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    elif "state_dict" in state:
        state = state["state_dict"]
    return {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state.items()
    }


def load_weights(model, weights, map_location="cpu", strict=True):
    """Load already-flattened weights into an explicit model."""
    if isinstance(weights, (str, bytes, os.PathLike)):
        state = torch.load(weights, map_location=map_location)
    else:
        state = weights
    model.load_state_dict(flatten_state_dict(state), strict=strict)
    return model


def load_model(name="default", weights=None, device="cpu"):
    """Construct, load, and prepare a model for inference."""
    model = create_model(name)
    if weights is None:
        weights = os.path.join(WEIGHTS_DIRECTORY, name + ".pt")
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
    "WEIGHTS_DIRECTORY",
    "create_model",
    "flatten_state_dict",
    "load_weights",
    "load_model",
]
