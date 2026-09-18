"""Convert original TargetCall checkpoints into flattened model weights.

The explicit models only accept flattened state dicts whose keys match the
model. This helper maps the original nested checkpoint names onto those keys
using ``legacy_state_keys.py``.

    python -m model_inference.convert_legacy TINYX011 \
        bonito/models/TINYX011/weights_1.tar
"""

import argparse
import os

import torch

from . import MODEL_CLASSES, WEIGHTS_DIRECTORY, create_model
from .legacy_state_keys import LEGACY_STATE_KEYS


def flatten_legacy_state_dict(model_name, state):
    """Map an original TargetCall checkpoint onto the flattened key names."""
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    elif "state_dict" in state:
        state = state["state_dict"]
    state = {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state.items()
    }

    try:
        mapping = LEGACY_STATE_KEYS[model_name]
    except KeyError as error:
        available = ", ".join(LEGACY_STATE_KEYS)
        raise ValueError(
            f"unknown model '{model_name}'; choose from: {available}"
        ) from error

    flattened = {}
    for new_key, legacy_key in mapping.items():
        if legacy_key not in state:
            raise KeyError(
                "missing legacy weight '%s' for '%s'" % (legacy_key, new_key)
            )
        flattened[new_key] = state[legacy_key]
    return flattened


def convert(model_name, weights_path, output_path=None):
    """Convert one original checkpoint and return the flattened file path."""
    state = torch.load(weights_path, map_location="cpu")
    flattened = flatten_legacy_state_dict(model_name, state)

    model = create_model(model_name)
    model.load_state_dict(flattened, strict=True)

    if output_path is None:
        output_path = os.path.join(WEIGHTS_DIRECTORY, model_name + ".pt")
    output_directory = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_directory, exist_ok=True)
    torch.save(flattened, output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=sorted(MODEL_CLASSES))
    parser.add_argument("weights", help="original TargetCall weights_*.tar")
    parser.add_argument("-o", "--output", default=None, help="output .pt path")
    args = parser.parse_args()

    path = convert(args.model, args.weights, args.output)
    print("wrote", path)


if __name__ == "__main__":
    main()
