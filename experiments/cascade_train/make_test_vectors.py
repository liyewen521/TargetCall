#!/usr/bin/env python3
"""Rebuild test_vectors/<name>_int8 from an annotated ONNX and the float vectors.

The int8 boundary vectors are just the float stimuli quantised with the two
boundary multipliers carried by the model, so they become stale whenever the
annotation changes (input scale) or the model is retrained (both).

    python make_test_vectors.py --model ../../onnx_models/tinyx3_ivfs_q8.onnx

Writes input_*.npy, output_*.npy, scales.tsv, layer_scales.tsv and reports how
many files actually changed.
"""
import argparse
import glob
import os

import numpy as np
import onnx

QMAX = 127


def q(x, s):
    return np.clip(np.rint(np.asarray(x, np.float64) * s), -QMAX - 1, QMAX).astype(np.int8)


def node_attrs(n):
    d = {}
    for a in n.attribute:
        if a.name.endswith("_scales"):
            d[a.name] = np.asarray(list(a.floats), np.float64)
        elif a.name.endswith("_scale"):
            d[a.name] = float(a.f)
        elif a.name.endswith("_bitdepth"):
            d[a.name] = int(a.i)
    return d


def fmt(v):
    if v is None:
        return "nan"
    a = np.asarray(v, np.float64)
    if a.size == 1:
        return "%.9f" % float(a.reshape(-1)[0])
    return "[" + ",".join("%.9f" % x for x in a) + "]"


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    ap.add_argument("--model", default=os.path.join(root, "onnx_models", "tinyx3_ivfs_q8.onnx"))
    ap.add_argument("--src", default=os.path.join(root, "test_vectors", "tinyx3_ivfs_folded"))
    ap.add_argument("--dst", default=os.path.join(root, "test_vectors", "tinyx3_ivfs_folded_int8"))
    a = ap.parse_args()

    m = onnx.load(a.model)
    nodes = [(n, node_attrs(n)) for n in m.graph.node if n.op_type != "Constant"]

    first = nodes[0][1]
    s_in = first.get("input_scales", first.get("input_scale"))
    last = [d for n, d in nodes if n.op_type == "Softmax"][0]
    s_out = last["output_scale"]
    print("input  scale = %s" % fmt(s_in))
    print("output scale = %s" % fmt(s_out))

    changed, clip_tot, n_tot = 0, 0, 0
    for f in sorted(glob.glob(os.path.join(a.src, "input_*.npy"))):
        tag = os.path.basename(f)[6:9]
        x = np.load(f)
        qi = q(x, s_in)
        clip_tot += int(np.sum(np.abs(np.rint(np.asarray(x, np.float64) * s_in)) > QMAX))
        n_tot += qi.size
        dst = os.path.join(a.dst, "input_%s.npy" % tag)
        old = np.load(dst) if os.path.exists(dst) else None
        if old is None or not np.array_equal(old, qi):
            np.save(dst, qi)
            changed += 1

        y = np.load(os.path.join(a.src, "output_%s.npy" % tag))
        qo = q(y, s_out)
        dst = os.path.join(a.dst, "output_%s.npy" % tag)
        old = np.load(dst) if os.path.exists(dst) else None
        if old is None or not np.array_equal(old, qo):
            np.save(dst, qo)
            changed += 1

    with open(os.path.join(a.dst, "scales.tsv"), "w") as fh:
        fh.write("tensor\tshape\tdtype\tscale(multiplier)\tdivisor(1/scale)\tbits\n")
        x0 = np.load(sorted(glob.glob(os.path.join(a.src, "input_*.npy")))[0])
        y0 = np.load(sorted(glob.glob(os.path.join(a.src, "output_*.npy")))[0])
        for name, shape, s in (("input", x0.shape, s_in), ("output", y0.shape, s_out)):
            sv = float(np.asarray(s, np.float64).reshape(-1)[0])
            fh.write("%s\t%s\tint8\t%.9f\t%.9f\t8\n"
                     % (name, ",".join(str(d) for d in shape), sv, 1.0 / sv))

    with open(os.path.join(a.dst, "layer_scales.tsv"), "w") as fh:
        fh.write("op\tname\tinput_bitdepth\toutput_bitdepth\tinput_scale\toutput_scale\n")
        for n, d in nodes:
            fh.write("%s\t%s\t%s\t%s\t%s\t%s\n"
                     % (n.op_type, n.name, d.get("input_bitdepth", ""), d.get("output_bitdepth", ""),
                        fmt(d.get("input_scales", d.get("input_scale"))),
                        fmt(d.get("output_scales", d.get("output_scale")))))

    print("wrote %s (%d files changed)" % (a.dst, changed))
    print("input codes clipped: %d / %d (%.4f%%)" % (clip_tot, n_tot, 100.0 * clip_tot / n_tot))


if __name__ == "__main__":
    main()

