#!/usr/bin/env python3
"""Compare a simulator / compiler layer dump against the reference dump.

    python compare_layers.py --sim sim_layers.npz [--ref all_layers.i8.npz]

Both files are npz with one array per node.  Keys may use '/', '_' or a
leading '/' — they are normalised before matching, so 'conv0_Conv',
'/conv0/Conv' and 'conv0/Conv' are the same key.

For every common node it prints the mismatch ratio, the largest absolute
difference and the number of elements off by more than 1, and it flags a
constant per-channel offset, which is the signature of a dropped bias.
"""
import argparse
import os
import sys

import numpy as np

CHANNEL_CANDIDATES = (12, 20, 5)


def norm(key):
    return key.replace('/', '_').lstrip('_')


def load(path):
    if os.path.isdir(path):
        path = os.path.join(path, 'all_layers.i8.npz')
    z = np.load(path, allow_pickle=True)
    return {norm(k): np.asarray(z[k]) for k in z.files}


def constant_per_channel(d):
    """Return (channels, offsets) when d is a constant offset per channel."""
    for c in CHANNEL_CANDIDATES:
        if d.size % c:
            continue
        rows = d.reshape(c, -1)
        if all(len(np.unique(r)) == 1 for r in rows) and np.any(rows[:, 0]):
            return c, rows[:, 0]
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', default=None,
                    help='reference npz (default: the layer_ref directory of this repo)')
    ap.add_argument('--sim', required=True, help='simulator dump, npz keyed by node name')
    ap.add_argument('--thr', type=float, default=0.05,
                    help='mismatch ratio that counts as divergence (default 0.05)')
    a = ap.parse_args()

    if a.ref is None:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        a.ref = os.path.join(root, 'test_vectors', 'tinyx3_ivfs_folded_int8',
                             'layer_ref', 'all_layers.i8.npz')

    ref, sim = load(a.ref), load(a.sim)
    common = [k for k in ref if k in sim]
    if not common:
        print('no common keys')
        print('  ref:', list(ref)[:3], '...')
        print('  sim:', list(sim)[:3], '...')
        return 2

    print('%-40s %7s %9s %7s %6s  %s' % ('node', 'n', 'mismatch', 'max|d|', '>1', 'note'))
    first = None
    for k in common:
        r, s = ref[k].reshape(-1), sim[k].reshape(-1)
        if r.size != s.size:
            print('%-40s  SIZE %d vs %d' % (k, r.size, s.size))
            continue
        d = s.astype(np.int32) - r.astype(np.int32)
        mis = float(np.mean(d != 0))
        note = ''
        if mis:
            c, off = constant_per_channel(d)
            if c:
                note = 'CONSTANT PER CHANNEL (%d) -> bias? ' % c + ' '.join('%+d' % v for v in off)
        print('%-40s %7d %8.2f%% %7d %6d  %s'
              % (k, r.size, mis * 100, int(np.abs(d).max()), int(np.sum(np.abs(d) > 1)), note))
        if first is None and mis > a.thr:
            first = k

    print()
    if first:
        print('first divergent node: %s  -> inspect its input and the op just before it' % first)
    else:
        print('no node exceeds %.1f%% mismatch' % (a.thr * 100))
    return 0


if __name__ == '__main__':
    sys.exit(main())

