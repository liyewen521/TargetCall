#!/usr/bin/env python3
"""Calibrate divisor-scale RHB-Lite annotations; weights/activations int8.

The reference uses divisor scales from ACModelSight's TensorQuantizer.
End-to-end agreement with the unmodified backend and hardware is unverified.
Percentile clipping only selects static ranges; --pct 100 selects absmax.
Default calibration is vectors 000-039, leaving 040-049 for evaluation.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

from quant_reference import stripped, validate_annotations


def calibrate(model, vecdir, pct, lo=0, hi=39):
    model = stripped(model)
    have = {o.name for o in model.graph.output}
    for node in model.graph.node:
        if node.op_type == 'Constant':
            continue
        for output in node.output:
            if output not in have:
                model.graph.output.extend([onnx.ValueInfoProto(name=output)])
                have.add(output)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    sess = ort.InferenceSession(model.SerializeToString(), options, providers=['CPUExecutionProvider'])
    names = [o.name for o in sess.get_outputs()]
    input_name = sess.get_inputs()[0].name
    acc = {n: [] for n in names + [input_name]}
    files = [Path(vecdir) / ('input_%03d.npy' % i) for i in range(lo, hi + 1)]
    if not files:
        raise ValueError('calibration set is empty')
    for path in files:
        x = np.load(path)
        acc[input_name].append(x.reshape(-1))
        for name, output in zip(names, sess.run(None, {input_name: x})):
            acc[name].append(output.reshape(-1))
    stats = {}
    for name in list(acc):
        values = np.abs(np.concatenate(acc.pop(name)))
        stats[name] = (float(values.max()), float(np.percentile(values, pct)))
    print('calibrated on %d vectors (%03d-%03d), p%g' % (len(files), lo, hi, pct))
    return stats


def tensor_scales(nodes, stats, pct, input_name='input'):
    def scale(tensor, use_max=False):
        value = stats[tensor][0 if use_max or pct is None else 1]
        return float(np.float32(max(value, 1e-8)) / np.float32(127))
    scales = {input_name: scale(input_name)}
    for node in nodes:
        if node.op_type == 'Constant':
            continue
        if node.op_type == 'Conv':
            scales[node.output[0]] = scale(node.output[0])
        elif node.op_type == 'Softmax':
            scales[node.output[0]] = scale(node.output[0], use_max=True)
        else:
            scales[node.output[0]] = scales[node.input[0]]
    return scales


def add(node, name, value):
    keep = [a for a in node.attribute if a.name != name]
    del node.attribute[:]
    node.attribute.extend(keep)
    node.attribute.append(onnx.helper.make_attribute(name, value))


def annotate(model, scales, bias_bits=16):
    model = stripped(model)
    initializers = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type == 'Constant':
            continue
        add(node, 'output_bitdepth', 8)
        add(node, 'output_scale', scales[node.output[0]])
        add(node, 'output_quant_dim', [0])  # static batch=1, one scalar per tensor
        if node.op_type in ('Conv', 'Softmax'):
            add(node, 'input_bitdepth', 8)
            add(node, 'input_scale', scales[node.input[0]])
            add(node, 'input_quant_dim', [0])
        if node.op_type == 'Conv':
            w = initializers[node.input[1]].astype(np.float32)
            maximum = np.abs(w).reshape(w.shape[0], -1).max(axis=1)
            sw = np.maximum(maximum, np.float32(1e-8)) / np.float32(127)
            add(node, 'weight_bitdepth', 8)
            add(node, 'weight_quant_dim', [0])
            add(node, 'weight_ch_scales', sw.tolist())
            if len(node.input) > 2:
                add(node, 'bias_bitdepth', bias_bits)
        elif node.op_type == 'Softmax':
            add(node, 'mode', 0)
    metadata = {p.key: p.value for p in model.metadata_props}
    metadata.update({'quantization.scale_convention': 'divisor',
                     'quantization.backend': 'rhblite',
                     'quantization.bias_bitdepth': str(bias_bits)})
    onnx.helper.set_model_props(model, metadata)
    validate_annotations([(n, {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute})
                          for n in model.graph.node])
    return model


def main():
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=Path, default=root / 'onnx_models/tinyx3_ivfs.onnx')
    ap.add_argument('--vectors', type=Path, default=root / 'test_vectors/tinyx3_ivfs_folded')
    ap.add_argument('--out', type=Path, default=root / 'onnx_models/tinyx3_ivfs_q8.onnx')
    ap.add_argument('--pct', type=float, default=99.5)
    ap.add_argument('--calib-from', type=int, default=0)
    ap.add_argument('--calib-to', type=int, default=39)
    ap.add_argument('--bias-bits', type=int, choices=[8, 16], default=16)
    ap.add_argument('--stats-out', type=Path, help='write calibration maxabs values as JSON')
    args = ap.parse_args()
    if not 0 < args.pct <= 100 or args.calib_from > args.calib_to:
        ap.error('invalid percentile or calibration interval')
    src = onnx.load(args.model)
    stats = calibrate(src, args.vectors, args.pct, args.calib_from, args.calib_to)
    if args.stats_out:
        args.stats_out.write_text(json.dumps({k: v[0] for k, v in stats.items()}, indent=2, sort_keys=True) + '\n')
    scales = tensor_scales(src.graph.node, stats, args.pct, src.graph.input[0].name)
    result = annotate(src, scales, args.bias_bits)
    metadata = {p.key: p.value for p in result.metadata_props}
    metadata.update({'quantization.calibration_vectors': '%03d-%03d' % (args.calib_from, args.calib_to),
                     'quantization.percentile': str(args.pct)})
    onnx.helper.set_model_props(result, metadata)
    assert [i.SerializeToString() for i in src.graph.initializer] == [i.SerializeToString() for i in result.graph.initializer]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(result, args.out)
    print('wrote %s; scale invariants passed; initializers unchanged' % args.out)


if __name__ == '__main__':
    main()
