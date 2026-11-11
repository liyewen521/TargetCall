#!/usr/bin/env python3
"""Dump per-node int8 activations of a quantisation-annotated ONNX model.

Two modes:

  int    (default) run a true integer pipeline: quantise the input, convolve,
         requantise and clamp at every node. These are the codes the hardware is
         supposed to carry, so a correct compiler should match them layer by
         layer and any divergence is a real bug.
  ideal  quantise the float onnxruntime activation of every node instead. This
         is the old behaviour; it is one quantisation step "too good" at every
         layer and diverges from any real int8 pipeline by ~20% at the output.

    python dump_layers.py --idx 0 --out ../../test_vectors/tinyx3_ivfs_folded_int8/layer_ref

Writes:
    000_input.i8.npy            int8 codes of the network input
    NNN_<node>.i8.npy/.f32.npy  the first --nodes nodes, int8 and float
    <node>.acc.i32.npy          int32 accumulator of the first Conv
    all_layers.i8.npz           int8 codes of every node, key = node name
    manifest.tsv                node -> bit depths -> scales -> file names
"""
import argparse
import os

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from onnx import numpy_helper

QUANT_SUFFIX = ("_scale", "_scales", "_bitdepth", "_quant_dim", "_quant", "mode")


def read_annotations(path):
    m = onnx.load(path)
    ini = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
    nodes = []
    for n in m.graph.node:
        d = {}
        for a in n.attribute:
            if a.name.endswith("_scales") or a.name == "weight_ch_scales":
                d[a.name] = np.asarray(list(a.floats), np.float64)
            elif a.name.endswith("_scale"):
                d[a.name] = float(a.f)
            elif a.name.endswith("_bitdepth"):
                d[a.name] = int(a.i)
            elif len(a.ints):
                d[a.name] = list(a.ints)
            elif a.type == onnx.AttributeProto.INT:
                d[a.name] = [a.i]
        nodes.append((n, d))
    return m, nodes, ini


def runtime_session(path, out_dir):
    m = onnx.load(path)
    for n in m.graph.node:
        keep = [a for a in n.attribute if not a.name.endswith(QUANT_SUFFIX)]
        del n.attribute[:]
        n.attribute.extend(keep)
    have = {o.name for o in m.graph.output}
    for n in m.graph.node:
        for o in n.output:
            if o and o not in have:
                m.graph.output.extend([onnx.ValueInfoProto(name=o)])
                have.add(o)
    p = os.path.join(out_dir, "_runtime.onnx")
    onnx.save(m, p)
    return ort.InferenceSession(p, providers=["CPUExecutionProvider"])


def bc(s, ndim):
    s = np.asarray(s, np.float64)
    if s.size == 1:
        return float(s.reshape(-1)[0])
    shape = [1] * ndim
    shape[1] = s.size
    return s.reshape(shape)


def out_scale(d):
    return d.get("output_scales", d.get("output_scale"))


def in_scale(d):
    return d.get("input_scales", d.get("input_scale"))


def run_integer(nodes, ini, x, s0):
    val = {"input": x}
    tsc = {"input": s0}          # scale of every tensor, set by its producer
    codes = {}
    for n, d in nodes:
        op = n.op_type
        if op == "Constant":
            for a in n.attribute:
                if a.name in ("t", "value"):
                    val[n.output[0]] = numpy_helper.to_array(a.t)
            continue
        if op == "Conv":
            qmax = (1 << d["output_bitdepth"]) - 1
            s_in = in_scale(d) if "input_scale" in d or "input_scales" in d else tsc[n.input[0]]
            s_out = out_scale(d)
            s_w = d["weight_ch_scales"]
            w = ini[n.input[1]].astype(np.float64)
            s4 = s_w.reshape(-1, 1, 1, 1)
            wd = np.rint(w * s4) / s4
            pads = d.get("pads", [0, 0, 0, 0])
            st = d.get("strides", [1, 1])
            xin = val[n.input[0]].astype(np.float64) / bc(s_in, 4)
            y = F.conv2d(torch.from_numpy(xin), torch.from_numpy(wd), None,
                         padding=(int(pads[0]), int(pads[1])),
                         stride=(int(st[0]), int(st[1])),
                         groups=int(d.get("group", [1])[0])).numpy()
            if len(n.input) > 2:
                y = y + ini[n.input[2]].astype(np.float64).reshape(1, -1, 1, 1)
            y = np.clip(np.rint(y * bc(s_out, 4)), -qmax - 1, qmax)
        elif op == "Relu":
            qmax = (1 << d["output_bitdepth"]) - 1
            y = np.maximum(val[n.input[0]].astype(np.float64) / bc(tsc[n.input[0]], 4), 0.0)
            y = np.clip(np.rint(y * bc(out_scale(d), 4)), -qmax - 1, qmax)
        elif op == "Reshape":
            v = val[n.input[0]]
            shape = np.asarray(val[n.input[1]]).astype(np.int64).tolist()
            shape = [v.shape[i] if s == 0 else s for i, s in enumerate(shape)]
            y = v.reshape(shape)
            tsc[n.output[0]] = tsc[n.input[0]]
        elif op == "Transpose":
            y = np.transpose(val[n.input[0]], d.get("perm", list(range(val[n.input[0]].ndim))))
            tsc[n.output[0]] = tsc[n.input[0]]
        elif op == "Softmax":
            qmax = (1 << d["output_bitdepth"]) - 1
            z = val[n.input[0]].astype(np.float64) / bc(d["input_scale"], 4)
            p = torch.softmax(torch.from_numpy(z), dim=-1).numpy()
            y = np.clip(np.rint(p * bc(d["output_scale"], 4)), -qmax - 1, qmax)
        else:
            raise SystemExit("unhandled op %s in node %s" % (op, n.name))
        if n.output[0] not in tsc:
            tsc[n.output[0]] = out_scale(d)
        val[n.output[0]] = y
        codes[n.name] = np.asarray(y.astype(np.int8))
    return codes


def run_ideal(nodes, sess, x):
    val = {o.name: np.asarray(v) for o, v in zip(sess.get_outputs(), sess.run(None, {"input": x}))}
    codes = {}
    for n, d in nodes:
        if n.op_type == "Constant" or not n.output:
            continue
        y = val.get(n.output[0])
        s = out_scale(d)
        if y is None or s is None:
            continue
        q = (1 << d.get("output_bitdepth", 8)) - 1
        codes[n.name] = np.clip(np.rint(y.astype(np.float64) * bc(s, y.ndim)),
                                -q - 1, q).astype(np.int8)
    return codes


def qname(name):
    return name.replace("/", "_").lstrip("_")


def conv0_accumulator(model, nodes, ini, xq):
    node, d = nodes[0]
    if node.op_type != "Conv" or node.input[0] != "input":
        return None
    w = ini[node.input[1]]
    b = ini[node.input[2]] if len(node.input) > 2 else None
    s_in = np.asarray(in_scale(d), np.float64).reshape(-1)
    s_out = np.asarray(out_scale(d), np.float64).reshape(-1)
    s_w = np.asarray(d["weight_ch_scales"], np.float64)
    sin = float(s_in[0])
    kh, kw = w.shape[2], w.shape[3]
    wq = np.clip(np.rint(w[:, 0].astype(np.float64) * s_w[:, None, None]), -128, 127).astype(np.int32)
    xp = np.pad(xq[0, 0].astype(np.int32), ((0, 0), (kh // 2, kh // 2)))
    acc = np.zeros((wq.shape[0], xp.shape[1] - kw + 1), np.int32)
    for k in range(kh):
        for l in range(kw):
            acc += wq[:, k, l][:, None] * xp[k, l:l + acc.shape[1]][None, :]
    if b is not None:
        acc += np.rint(b.astype(np.float64) * sin * s_w).astype(np.int32)[:, None]
    m = s_out / (sin * s_w)
    yq = np.clip(np.rint(acc.astype(np.float64) * m.reshape(-1, 1)), -128, 127).astype(np.int8)
    return acc, yq, s_in, s_out, s_w, b, m


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    ap.add_argument("--model", default=os.path.join(root, "onnx_models", "tinyx3_ivfs_q8.onnx"))
    ap.add_argument("--vector", default=None, help="float32 input .npy")
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(root, "test_vectors",
                                                  "tinyx3_ivfs_folded_int8", "layer_ref"))
    ap.add_argument("--nodes", type=int, default=10)
    ap.add_argument("--mode", choices=("int", "ideal"), default="int")
    a = ap.parse_args()

    vec = a.vector or os.path.join(root, "test_vectors", "tinyx3_ivfs_folded",
                                   "input_%03d.npy" % a.idx)
    os.makedirs(a.out, exist_ok=True)

    model, nodes, ini = read_annotations(a.model)
    sess = runtime_session(a.model, a.out)
    x = np.load(vec).astype(np.float32)
    s0 = in_scale(nodes[0][1])
    xq = np.clip(np.rint(x.astype(np.float64) * bc(s0, 4)), -128, 127).astype(np.int8)
    np.save(os.path.join(a.out, "000_input.i8.npy"), xq)

    codes = run_integer(nodes, ini, xq, s0) if a.mode == "int" else run_ideal(nodes, sess, x)
    f32 = {o.name: np.asarray(v) for o, v in zip(sess.get_outputs(), sess.run(None, {"input": x}))}

    acc = conv0_accumulator(model, nodes, ini, xq)
    if acc is not None:
        np.save(os.path.join(a.out, "%s.acc.i32.npy" % qname(nodes[0][0].name)), acc[0])

    rows = []
    for i, (n, d) in enumerate([(n, d) for n, d in nodes if n.op_type != "Constant"]):
        if n.name not in codes:
            continue
        y = f32.get(n.output[0])
        base = "%03d_%s" % (i, qname(n.name))
        if i < a.nodes and y is not None:
            np.save(os.path.join(a.out, base + ".f32.npy"), y.reshape(-1).astype(np.float32))
            np.save(os.path.join(a.out, base + ".i8.npy"), codes[n.name])
        rows.append([i, n.op_type, n.name, codes[n.name].size,
                     d.get("input_bitdepth", ""), d.get("output_bitdepth", ""),
                     base + ".i8.npy" if i < a.nodes else "", base + ".f32.npy" if i < a.nodes else ""])

    np.savez_compressed(os.path.join(a.out, "all_layers.i8.npz"),
                        **{qname(k): v for k, v in codes.items()})
    with open(os.path.join(a.out, "manifest.tsv"), "w") as f:
        f.write("idx\top\tnode\tn\tin_bits\tout_bits\tfile_i8\tfile_f32\n")
        for r in rows:
            f.write("\t".join(str(v) for v in r) + "\n")

    print("%d nodes, %d tensors -> %s (mode=%s)" % (len(rows), len(codes), a.out, a.mode))
    print("input scale=%s" % np.asarray(s0).reshape(-1)[:4])
    if acc is not None:
        print("conv0 acc range [%d, %d]" % (acc[0].min(), acc[0].max()))


if __name__ == "__main__":
    main()

