#!/usr/bin/env python3
"""Verify a compiled tinyx3 RHB-Lite case with TargetCall's integer convolution."""

import argparse
from pathlib import Path
import re
import sys

import numpy as np
from onnx import numpy_helper
import torch

from compare_layers import compare, norm
from quant_reference import conv_accumulate, read_annotations


FIELD = re.compile(
    r"\|\s*(\w+)\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*"
    r"\[\s*(-?\d+(?:,\d{3})*)\s*,\s*(?:0[xX][0-9a-fA-F]+(?:,[0-9a-fA-F]+)*|0)\s*\]"
)


def instruction_fields(path):
    return {name: int(value.replace(",", "")) for name, value in FIELD.findall(path.read_text())}


def replay_softmax(logits, fields, kernel):
    rows = fields["sm_matrix_y"] + 1
    cols = fields["sm_matrix_x"] + 1
    tile = fields["tile_mode"]
    if fields["spu_op"] != 0 or fields["sm_mode"] != 0 or tile not in (0, 1):
        raise ValueError("expected a static RHB-Lite Softmax instruction")
    if logits.shape != (1, 1, rows, cols):
        raise ValueError(f"Softmax instruction expects (1, 1, {rows}, {cols}), got {logits.shape}")

    row_alignment, col_alignment = (16, 16) if tile == 0 else (1, 256)
    padded = np.pad(logits, ((0, 0), (0, 0), (0, -rows % row_alignment), (0, -cols % col_alignment)))
    input_scale = torch.tensor(2.0 ** -fields["sm_shift1"])
    exp_scale = torch.tensor(2.0 ** -fields["sm_shift2"])
    output_scale = torch.tensor(2.0 ** -fields["sm_shift0"])
    output, _, _ = kernel(
        torch.from_numpy(np.ascontiguousarray(padded)), rows, cols, input_scale, exp_scale, output_scale,
        tile_mode=tile, sm_mode=fields["sm_mode"],
    )
    return np.asarray(output, dtype=np.int8).reshape(padded.shape)[:, :, :rows, :cols]


def replay_case(model_path, case_path, seed, input_path, softmax_kernel):
    model, nodes, initializers = read_annotations(model_path)
    layers = [(node, attrs) for node, attrs in nodes if node.op_type != "Constant"]
    body = layers[:-3]
    if (not body or
            [node.op_type for node, _ in layers[-3:]] != ["Reshape", "Transpose", "Softmax"]):
        raise ValueError("expected a tinyx3 convolution chain ending in Reshape, Transpose, Softmax")
    activation = model.graph.input[0].name
    for node, _ in layers:
        if node.input[0] != activation:
            raise ValueError(f"expected a single activation chain at {node.name}")
        activation = node.output[0]
    for index, (node, _) in enumerate(body):
        if node.op_type == "Conv":
            continue
        if node.op_type != "Relu" or index == 0 or body[index - 1][0].op_type != "Conv":
            raise ValueError(f"expected a convolution or fused ReLU at {node.name}")

    case = case_path.with_name(f"{case_path.name}_{seed}")
    codes = np.load(input_path, allow_pickle=False)
    if codes.ndim != 4 or codes.dtype != np.int8:
        raise ValueError(f"expected a four-dimensional int8 input in {input_path}")
    with np.load(case / "net_csim_input.npz", allow_pickle=False) as trace:
        if not np.array_equal(codes, trace["in_0"]):
            raise ValueError("input vector differs from the CModel trace")

    convs = [index for index, (node, _) in enumerate(body) if node.op_type == "Conv"]
    trace_files = {path.name for path in case.glob("*_csim_out.npz")}
    expected_files = {f"{index}_csim_out.npz" for index in range(len(convs) + 1)}
    if trace_files != expected_files:
        raise ValueError(f"incomplete or unexpected operator dumps in {case}")

    reference, simulated = {}, {}
    for op, index in enumerate(convs):
        node, attrs = body[index]
        fused_relu = index + 1 < len(body) and body[index + 1][0].op_type == "Relu"
        result_node = body[index + 1][0] if fused_relu else node
        fields = instruction_fields(case_path.parent / f"{op}_{op}.inst")
        if (fields["OP"] != 1 or fields["dyadic_en"] != 1
                or fields["act_mode"] != (2 if fused_relu else 0)
                or any(fields[name] for name in ("wb_width", "elmwise_en", "dyadic_lyr_en", "rmat_const_en"))
                or bool(fields["bias_en"]) != (len(node.input) > 2)):
            raise ValueError(f"instruction {op} does not match the supported convolution at {node.name}")

        acc, _ = conv_accumulate(node, attrs, initializers, codes)
        with np.load(case / f"{op}_csim_out.npz", allow_pickle=False) as trace:
            mantissa, shift = trace["mantissa"], trace["exponents"]
            if mantissa.shape != (acc.shape[1],) or shift.shape != mantissa.shape:
                raise ValueError(f"instruction {op} has invalid per-channel coefficients")
            factor = np.ldexp(mantissa.astype(np.float64), -shift).reshape(1, -1, 1, 1)
            requantized = np.rint(acc.astype(np.float64) * factor)
            codes = np.clip(requantized, 0 if fused_relu else -128, 127).astype(np.int8)
            name = norm(result_node.name)
            reference[name], simulated[name] = codes, trace["out"]

    reshape, transpose, softmax = layers[-3:]
    constants = {node.output[0]: numpy_helper.to_array(attrs["value"])
                 for node, attrs in nodes if node.op_type == "Constant"}
    shape_name = reshape[0].input[1]
    shape = constants.get(shape_name, initializers.get(shape_name))
    if shape is None:
        raise ValueError("expected a constant Reshape shape")
    shape = [codes.shape[axis] if dim == 0 else int(dim) for axis, dim in enumerate(shape)]
    logits = codes.reshape(shape).transpose(transpose[1].get("perm"))
    if softmax[1].get("axis", -1) not in (-1, 3):
        raise ValueError("expected Softmax on the last axis")

    op = len(convs)
    fields = instruction_fields(case_path.parent / f"{op}_{op}_sm_0.inst")
    result = replay_softmax(logits, fields, softmax_kernel)
    rows, cols = logits.shape[-2:]
    with np.load(case / f"{op}_csim_out.npz", allow_pickle=False) as trace:
        name = norm(transpose[0].name)
        reference[name], simulated[name] = logits, trace["input"][:, :, :rows, :cols]
        actual = trace["outputMat"][:, :, :rows, :cols]
    with np.load(case / "net_csim_output.npz", allow_pickle=False) as trace:
        if len(trace.files) != 1 or not np.array_equal(actual, trace[trace.files[0]]):
            raise ValueError("CModel Softmax dump differs from the network output")
    name = norm(softmax[0].name)
    reference[name], simulated[name] = result, actual
    return reference, simulated


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=root / "onnx_models/tinyx3_ivfs_q8.onnx")
    parser.add_argument("--case-path", type=Path, required=True, help="the --case_path passed to acsim.py")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input", type=Path, help="quantized int8 input .npy")
    parser.add_argument("--acsim-dir", type=Path, required=True, help="demo/acsim directory with pyACSim and pyEngine")
    args = parser.parse_args()
    input_path = args.input or root / f"test_vectors/tinyx3_ivfs_folded_int8/input_{args.seed:03d}.npy"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.acsim_dir.resolve()))
    try:
        from pyEngine.spu import SoftMax_CModel_torch

        torch.set_num_threads(1)
        reference, simulated = replay_case(
            args.model, args.case_path.resolve(), args.seed, input_path, SoftMax_CModel_torch,
        )
        return compare(reference, simulated)
    except (ImportError, OSError, ValueError, KeyError, OverflowError, RuntimeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
