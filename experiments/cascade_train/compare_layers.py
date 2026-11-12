#!/usr/bin/env python3
"""Compare complete integer layer dumps; fail on missing layers or mismatches.

Exit status: 0 = pass, 1 = numerical mismatch, 2 = malformed/incomplete dump.
Default comparison is exact. --allow-partial and --flatten explicitly relax
coverage and layout checks for diagnostic dumps, never implicitly for CI.
"""
import argparse
from pathlib import Path
import sys

import numpy as np


def norm(key):
    return key.replace('/', '_').lstrip('_')


def load(path):
    path = Path(path)
    if path.is_dir():
        path = path / 'all_layers.i8.npz'
    result = {}
    with np.load(path, allow_pickle=False) as archive:
        for key in archive.files:
            name = norm(key)
            if name in result:
                raise ValueError('duplicate normalized node name: ' + name)
            value = archive[key]
            if value.size == 0 or not np.issubdtype(value.dtype, np.integer):
                raise ValueError('node %s must contain nonempty integer codes' % name)
            result[name] = value
    if not result:
        raise ValueError('empty layer dump')
    return result


def compare(ref, sim, threshold=0.0, allow_partial=False, flatten=False):
    missing = sorted(set(ref) - set(sim))
    extra = sorted(set(sim) - set(ref))
    if missing:
        print('missing %d reference nodes: %s' % (len(missing), ', '.join(missing)))
    if extra:
        print('unexpected nodes: ' + ', '.join(extra))
    malformed = bool(extra or (missing and not allow_partial))
    common = [key for key in ref if key in sim]
    if not common:
        print('no common nodes')
        return 2
    print('%-40s %7s %9s %7s %7s  %s' % ('node', 'n', 'mismatch', 'max|d|', '>1', 'note'))
    first = None
    for key in common:
        r, s = ref[key], sim[key]
        if r.shape != s.shape and not (flatten and r.size == s.size):
            print('%s: SHAPE %s vs %s' % (key, r.shape, s.shape))
            malformed = True
            continue
        d = s.reshape(-1).astype(np.int64) - r.reshape(-1).astype(np.int64)
        mismatch = float(np.mean(d != 0))
        note = ''
        if r.ndim == 4 and r.shape[1] > 1 and mismatch:
            channels = d.reshape(r.shape).transpose(1, 0, 2, 3).reshape(r.shape[1], -1)
            if np.all(channels == channels[:, :1]):
                note = 'constant per-channel offset -> check bias: ' + str(channels[:, 0].tolist())
        print('%-40s %7d %8.2f%% %7d %7d  %s' %
              (key, r.size, 100 * mismatch, np.abs(d).max(), np.sum(np.abs(d) > 1), note))
        if mismatch > threshold and first is None:
            first = key
    if malformed:
        print('FAIL: incomplete or structurally incompatible layer dump')
        return 2
    if first is not None:
        print('FAIL: first divergent node: ' + first)
        return 1
    print('PASS: %d/%d nodes, mismatch threshold %.2f%%' % (len(common), len(ref), 100 * threshold))
    return 0


def main():
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', default=root / 'test_vectors/tinyx3_ivfs_folded_int8/layer_ref/all_layers.i8.npz')
    ap.add_argument('--sim', required=True)
    ap.add_argument('--thr', type=float, default=0.0)
    ap.add_argument('--allow-partial', action='store_true')
    ap.add_argument('--flatten', action='store_true')
    args = ap.parse_args()
    if not 0 <= args.thr <= 1:
        ap.error('--thr must be between 0 and 1')
    try:
        return compare(load(args.ref), load(args.sim), args.thr, args.allow_partial, args.flatten)
    except (OSError, ValueError) as error:
        print('FAIL:', error)
        return 2


if __name__ == '__main__':
    sys.exit(main())
