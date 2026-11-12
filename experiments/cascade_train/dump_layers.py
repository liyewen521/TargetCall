#!/usr/bin/env python3
"""Dump the same RHB-Lite numerical pipeline used by check_quant.py.

--mode ideal is diagnostic float-activation quantization, not a pipeline
reference. Softmax and requantization follow TargetCall's software reference;
hardware LUT/multiplier implementations require separate verification.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from quant_reference import in_scale, out_scale, quantize, read_annotations, run_integer, stripped


def runtime_session(model, out_dir):
    model = stripped(model)
    have = {o.name for o in model.graph.output}
    for node in model.graph.node:
        if node.op_type == 'Constant':
            continue
        for output in node.output:
            if output not in have:
                model.graph.output.extend([onnx.ValueInfoProto(name=output)])
                have.add(output)
    path = out_dir / '_runtime.onnx'
    onnx.save(model, path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])


def qname(name):
    return name.replace('/', '_').lstrip('_')


def main():
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=Path, default=root / 'onnx_models/tinyx3_ivfs_q8.onnx')
    ap.add_argument('--vector', type=Path, help='float input .npy before quantization (not int8 codes)')
    ap.add_argument('--idx', type=int, default=0)
    ap.add_argument('--out', type=Path, default=root / 'test_vectors/tinyx3_ivfs_folded_int8/layer_ref')
    ap.add_argument('--nodes', type=int, default=10)
    ap.add_argument('--mode', choices=['int', 'ideal'], default='int')
    args = ap.parse_args()
    torch.set_num_threads(1)
    path = args.vector or root / 'test_vectors/tinyx3_ivfs_folded' / ('input_%03d.npy' % args.idx)
    args.out.mkdir(parents=True, exist_ok=True)
    model, nodes, initializers = read_annotations(args.model)
    first = next(d for n, d in nodes if n.op_type != 'Constant')
    x = np.load(path).astype(np.float32)
    xq = quantize(x, in_scale(first), first['input_bitdepth'])
    np.save(args.out / '000_input.i8.npy', xq)
    sess = runtime_session(model, args.out)
    floats = {o.name: v for o, v in zip(sess.get_outputs(), sess.run(None, {model.graph.input[0].name: x}))}
    _, codes, accumulators = run_integer(nodes, initializers, xq)
    if args.mode == 'ideal':
        codes = {n.name: quantize(floats[n.output[0]], out_scale(d), d['output_bitdepth'])
                 for n, d in nodes if n.op_type != 'Constant'}
        accumulators = {}
    if accumulators:
        name = next(iter(accumulators))
        np.save(args.out / (qname(name) + '.acc.i32.npy'), accumulators[name])
    rows = ['idx\top\tnode\tn\tshape\tin_bits\tout_bits\tbias_bits\tinput_scale\toutput_scale\tfile_i8\tfile_f32']
    for i, (node, attrs) in enumerate((n, d) for n, d in nodes if n.op_type != 'Constant'):
        base = '%03d_%s' % (i, qname(node.name))
        if i < args.nodes:
            np.save(args.out / (base + '.i8.npy'), codes[node.name])
            np.save(args.out / (base + '.f32.npy'), floats[node.output[0]].reshape(-1).astype(np.float32))
        rows.append('\t'.join(map(str, [i, node.op_type, node.name, codes[node.name].size,
            ','.join(map(str, codes[node.name].shape)), attrs.get('input_bitdepth', ''),
            attrs['output_bitdepth'], attrs.get('bias_bitdepth', ''), attrs.get('input_scale', ''),
            attrs['output_scale'], base + '.i8.npy' if i < args.nodes else '-',
            base + '.f32.npy' if i < args.nodes else '-'])))
    np.savez_compressed(args.out / 'all_layers.i8.npz', **{qname(k): v for k, v in codes.items()})
    (args.out / 'manifest.tsv').write_text('\n'.join(rows) + '\n')
    (args.out / 'provenance.json').write_text(json.dumps({
        'model': args.model.name, 'model_sha256': hashlib.sha256(args.model.read_bytes()).hexdigest(),
        'input': path.name, 'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'mode': args.mode, 'scale_convention': 'divisor', 'node_count': len(codes),
    }, indent=2) + '\n')
    print('%d tensors -> %s (mode=%s)' % (len(codes), args.out, args.mode))
    print('input divisor:', in_scale(first))


if __name__ == '__main__':
    main()
