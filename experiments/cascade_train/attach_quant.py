#!/usr/bin/env python3
"""Attach RHB-Lite divisor scales, optionally from precomputed absmax values.

For calibration from vectors, prefer attach_quant_v2.py. The legacy multiplier
and exp_scale options are intentionally unsupported: they do not describe the
ACModelSight yuhao_quant RHB-Lite interface.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx

from attach_quant_v2 import annotate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src', type=Path)
    ap.add_argument('dst', type=Path)
    ap.add_argument('--bits', type=int, choices=[8], default=8)
    ap.add_argument('--input-scale', type=float, default=1.0, help='positive divisor')
    ap.add_argument('--output-scale', type=float, default=1.0, help='positive divisor')
    ap.add_argument('--bias-bits', type=int, choices=[8, 16], default=16)
    ap.add_argument('--act-maxabs', type=Path)
    ap.add_argument('--mode', choices=['dequant'], default='dequant')
    args = ap.parse_args()
    if min(args.input_scale, args.output_scale) <= 0:
        ap.error('scales must be positive divisors')
    stats = json.loads(args.act_maxabs.read_text()) if args.act_maxabs else {}
    def scale(name, default):
        return float(np.float32(max(stats[name], 1e-8)) / np.float32(127)) if name in stats else default
    model = onnx.load(args.src)
    input_name = model.graph.input[0].name
    scales = {input_name: scale(input_name, args.input_scale)}
    for node in model.graph.node:
        if node.op_type == 'Constant':
            continue
        scales[node.output[0]] = (scale(node.output[0], args.output_scale)
                                 if node.op_type in ('Conv', 'Softmax')
                                 else scales[node.input[0]])
    model = annotate(model, scales, args.bias_bits)
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.dst)
    print('wrote divisor-scale RHB-Lite model:', args.dst)


if __name__ == '__main__':
    main()
