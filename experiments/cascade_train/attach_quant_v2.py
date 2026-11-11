#!/usr/bin/env python3
"""Annotate tinyx3_ivfs.onnx with the quantisation the compiler actually supports.

Rules (verified against the reference file ctc_test.tinyx3.onnx):
  * activation scales are PER-TENSOR only - per-channel is not implemented for
    Conv input/output (ac_model_sight/quantizer/utils/act_quant.py raises
    NotImplementedError for it), so every activation scale is a scalar;
  * ONE scale per tensor: a Relu / Reshape / Transpose output carries exactly
    the scale of its input tensor, and a Conv input_scale is the scale of the
    tensor it consumes (in the reference: 29/29 Relu and 120/120 Conv agree);
  * weights are per-channel int8 (`weight_ch_scales = 127 / max|w|_c`);
  * the calibrated tensors (network input, Conv outputs, Softmax in/out) are
    clipped at `--pct` instead of absmax, which was the one remaining lever
    once per-channel is off the table;
  * decoder conv / Reshape / Transpose / Softmax stay 8-bit.

    python attach_quant_v2.py --pct 99.9
"""
import argparse
import glob
import os

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

QUANT_SUFFIX = ("_scale", "_scales", "_bitdepth", "_quant_dim", "_quant", "mode")


def stripped(model):
    m = onnx.ModelProto()
    m.CopyFrom(model)
    for n in m.graph.node:
        keep = [a for a in n.attribute if not a.name.endswith(QUANT_SUFFIX)]
        del n.attribute[:]
        n.attribute.extend(keep)
    return m


def calibrate(model, vecdir, pct):
    """max|.| and the pct percentile of every tensor over the calibration set."""
    m = stripped(model)
    have = {o.name for o in m.graph.output}
    for n in m.graph.node:
        for o in n.output:
            if o and o not in have:
                m.graph.output.extend([onnx.ValueInfoProto(name=o)])
                have.add(o)
    onnx.save(m, "/tmp/_calib_v2.onnx")
    sess = ort.InferenceSession("/tmp/_calib_v2.onnx", providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]

    acc = {n: [] for n in names}
    acc["input"] = []
    files = sorted(glob.glob(os.path.join(vecdir, "input_*.npy")))
    for f in files:
        x = np.load(f)
        acc["input"].append(np.asarray(x).reshape(-1))
        for n, o in zip(names, sess.run(None, {"input": x})):
            acc[n].append(np.asarray(o).reshape(-1))

    stats = {}
    for k in list(acc.keys()):
        ab = np.abs(np.concatenate(acc.pop(k)))
        stats[k] = (float(ab.max()), float(np.percentile(ab, pct)))
        del ab
    print("calibrated on %d vectors, clip = p%g" % (len(files), pct))
    return stats


def tensor_scales(nodes, stats, pct):
    """One multiplier per tensor. Only calibrated tensors use the percentile."""
    def ref(t):
        amax, p = stats[t]
        return p if pct is not None else amax

    sc = {"input": 127.0 / max(ref("input"), 1e-12)}
    for n in nodes:
        if n.op_type == "Constant":
            continue
        t = n.output[0]
        if n.op_type == "Conv":
            sc[t] = 127.0 / max(ref(t), 1e-12)
        elif n.op_type == "Softmax":
            sc[t] = 127.0 / max(stats[t][0], 1e-12)     # probabilities: keep the peak
        else:
            sc[t] = sc[n.input[0]]                      # Relu / Reshape / Transpose
    return sc


def add(node, name, value):
    for a in list(node.attribute):
        if a.name == name:
            node.attribute.remove(a)
    node.attribute.append(onnx.helper.make_attribute(name, value))


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    ap.add_argument("--model", default=os.path.join(root, "onnx_models", "tinyx3_ivfs.onnx"))
    ap.add_argument("--vectors", default=os.path.join(root, "test_vectors", "tinyx3_ivfs_folded"))
    ap.add_argument("--out", default=os.path.join(root, "onnx_models", "tinyx3_ivfs_q8.onnx"))
    ap.add_argument("--pct", type=float, default=99.5)
    a = ap.parse_args()

    src = onnx.load(a.model)
    model = stripped(src)
    stats = calibrate(model, a.vectors, a.pct)
    nodes = [n for n in model.graph.node if n.op_type != "Constant"]
    sc = tensor_scales(nodes, stats, a.pct)
    wi = {i.name: i for i in model.graph.initializer}

    for n in nodes:
        if n.op_type == "Conv":
            add(n, "input_bitdepth", 8)
            add(n, "input_scale", float(sc[n.input[0]]))
            add(n, "weight_bitdepth", 8)
            w = numpy_helper.to_array(wi[n.input[1]]).astype(np.float64)
            wm = np.abs(w).reshape(w.shape[0], -1).max(axis=1)
            add(n, "weight_ch_scales", [float(x) for x in 127.0 / np.maximum(wm, 1e-12)])
            add(n, "output_bitdepth", 8)
            add(n, "output_scale", float(sc[n.output[0]]))
        elif n.op_type == "Softmax":
            add(n, "input_bitdepth", 8)
            add(n, "input_scale", float(sc[n.input[0]]))
            add(n, "output_bitdepth", 8)
            add(n, "output_scale", float(sc[n.output[0]]))
            add(n, "exp_scale", float(sc[n.output[0]] / sc[n.input[0]]))
            add(n, "mode", 0)
        else:
            add(n, "output_bitdepth", 8)
            add(n, "output_scale", float(sc[n.output[0]]))

    onnx.save(model, a.out)
    print("wrote %s (%d nodes)" % (a.out, len(model.graph.node)))

    # ---- invariants the reference file obeys
    prod = {}
    for n in model.graph.node:
        for o in n.output:
            prod[o] = n
    bad = []
    for n in model.graph.node:
        if n.op_type in ("Relu", "Reshape", "Transpose"):
            up = prod.get(n.input[0])
            if up is None:
                continue
            if _attr(up, "output_scale") != _attr(n, "output_scale"):
                bad.append((n.name, "elementwise scale differs from producer"))
        if n.op_type == "Conv":
            up = prod.get(n.input[0])
            if up is not None and _attr(up, "output_scale") != _attr(n, "input_scale"):
                bad.append((n.name, "conv input_scale differs from producer"))
    print("invariant violations:", len(bad))
    for b in bad[:5]:
        print("  ", b)

    src_i = {i.name: i.SerializeToString() for i in src.graph.initializer}
    dst_i = {i.name: i.SerializeToString() for i in model.graph.initializer}
    print("initializers unchanged:", src_i == dst_i)


def _attr(n, name):
    for a in n.attribute:
        if a.name == name:
            return onnx.helper.get_attribute_value(a)
    return None


if __name__ == "__main__":
    main()

