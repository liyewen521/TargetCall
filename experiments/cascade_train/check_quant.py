#!/usr/bin/env python3
"""Evaluate the shared RHB-Lite numerical reference against float golden outputs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from quant_reference import in_scale, quantize, read_annotations, run_integer


def load(path):
    _, nodes, initializers = read_annotations(path)
    return nodes, initializers


def emulate(nodes, initializers, x):
    return run_integer(nodes, initializers, x)[0]


def decode(q):
    labels = np.argmax(np.asarray(q).reshape(-1, 5), 1)
    out, previous = [], None
    for value in labels:
        if value != previous and value != 0:
            out.append(int(value))
        previous = value
    return out


def edit_identity(s1, s2):
    n, m = len(s1), len(s2)
    if max(n, m) == 0:
        return 1.0
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (s1[i - 1] != s2[j - 1]))
        prev = cur
    return 1.0 - prev[m] / max(n, m)


def main():
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=Path, default=root / 'onnx_models/tinyx3_ivfs_q8.onnx')
    ap.add_argument('--vectors', type=Path, default=root / 'test_vectors/tinyx3_ivfs_folded')
    ap.add_argument('--from', dest='lo', type=int, default=40)
    ap.add_argument('--to', dest='hi', type=int, default=49)
    ap.add_argument('--json', type=Path, help='write reproducible evaluation results')
    a = ap.parse_args()
    if a.lo > a.hi:
        ap.error('--from must not exceed --to')
    torch.set_num_threads(1)
    model, nodes, initializers = read_annotations(a.model)
    first = next(d for n, d in nodes if n.op_type != 'Constant')
    rows = []
    for i in range(a.lo, a.hi + 1):
        x = np.load(a.vectors / ('input_%03d.npy' % i))
        out = emulate(nodes, initializers, quantize(x, in_scale(first), first['input_bitdepth']))
        ref = np.load(a.vectors / ('output_%03d.npy' % i))
        agreement = float(np.mean(out.reshape(-1, 5).argmax(1) == ref.reshape(-1, 5).argmax(1)))
        identity = edit_identity(decode(out), decode(ref))
        rows.append(dict(index=i, frame_agreement=agreement, decoded_identity=identity))
        print('vector %03d: argmax %.2f%%  identity %.2f%%' % (i, 100 * agreement, 100 * identity))
    result = {
        'model': a.model.name,
        'model_sha256': hashlib.sha256(a.model.read_bytes()).hexdigest(),
        'quantization': {p.key: p.value for p in model.metadata_props},
        'reference': 'float golden outputs (not ground-truth bases)',
        'mean_frame_agreement': float(np.mean([r['frame_agreement'] for r in rows])),
        'mean_decoded_identity': float(np.mean([r['decoded_identity'] for r in rows])),
        'vectors': rows,
    }
    print('mean over %d vectors: argmax %.2f%%  identity %.2f%%' %
          (len(rows), 100 * result['mean_frame_agreement'], 100 * result['mean_decoded_identity']))
    if a.json:
        a.json.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
