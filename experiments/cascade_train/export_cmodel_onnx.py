#!/usr/bin/env python3
"""Export a multiplier-scale ONNX for the existing RHB-Lite compiler."""

import argparse
from pathlib import Path

import onnx

from quant_reference import checked_scale, read_annotations


SCALE_ATTRIBUTES = {
    "input_scale", "input_scales", "output_scale", "output_scales",
    "weight_scale", "weight_ch_scales",
}


def compiler_model(path):
    model, _, _ = read_annotations(path)
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.name not in SCALE_ATTRIBUTES:
                continue
            if attr.type == onnx.AttributeProto.FLOAT:
                checked_scale(attr.f)
                attr.f = float(checked_scale(1.0 / attr.f))
            elif attr.type == onnx.AttributeProto.FLOATS:
                checked_scale(attr.floats)
                attr.floats[:] = checked_scale([1.0 / scale for scale in attr.floats]).tolist()
            else:
                raise ValueError(f"{node.name} has a non-float scale attribute {attr.name}")
    for prop in model.metadata_props:
        if prop.key == "quantization.scale_convention":
            prop.value = "multiplier"
    return model


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=root / "onnx_models/tinyx3_ivfs_q8.onnx")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.model.resolve() == args.out.resolve():
        parser.error("choose an output file distinct from the divisor-scale model")
    model = compiler_model(args.model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.out)
    print(f"Exported multiplier-scale ONNX to {args.out}")


if __name__ == "__main__":
    main()
