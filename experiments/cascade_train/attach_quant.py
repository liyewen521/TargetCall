#!/usr/bin/env python3
"""Attach per-node quantization attributes to an ONNX model.

Attributes per op type:

    Conv                     input_bitdepth, input_scale, output_bitdepth,
                             output_scale, weight_bitdepth, weight_ch_scales
    Relu / Reshape / Transpose     output_bitdepth, output_scale
    Softmax                  input_bitdepth, input_scale, output_bitdepth,
                             output_scale, exp_scale

`weight_ch_scales` is per output-channel (length = out_channels) and uses the
symmetric int8 multiplier convention  scale = QMAX / max|w_channel|.

Non-Conv nodes are scale pass-throughs: their output_scale equals the scale of
the tensor feeding them, so every node's input_scale matches the previous
node's output_scale (the invariant the hardware viewer/reference export
`ctc_test.tinyx3.onnx` follows).  Softmax also carries `exp_scale`, which is
written as the ratio output_scale / input_scale (an int when it divides evenly,
otherwise a float) so that output_scale == input_scale * exp_scale always
holds, whichever of the two a consumer reads.

Activation scales default to 1.0 (no calibration).  Pass --act-maxabs with a
JSON file mapping tensor name -> max|value| measured over a calibration set to
give every Conv its own output_scale = max|value| / 127: the graph input scale
is read from the "input" entry and the Softmax output scale from its own output
tensor, so the int8 test vectors and the model agree on both boundaries.

Default QMAX for weights is 2**(bits-1)-1 (127 for int8), the convention used
by the first export of this model.  Pass --qmax 128 to match the reference
export, which uses 2**(bits-1).
"""

import argparse
import json

import numpy as np
import onnx
from onnx import numpy_helper


CONV_ATTRS = ["input_bitdepth", "input_scale", "output_bitdepth", "output_scale",
              "weight_bitdepth", "weight_ch_scales"]

PASSTHROUGH_OPS = ("Relu", "Reshape", "Transpose")

ACT_QMAX = 127.0


def get_attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            return onnx.helper.get_attribute_value(a)
    return default


def set_attr(node, name, value):
    keep = [a for a in node.attribute if a.name != name]
    del node.attribute[:]
    node.attribute.extend(keep)
    node.attribute.append(onnx.helper.make_attribute(name, value))


def load_act_scales(path):
    """tensor name -> symmetric int8 scale, from a max|value| calibration file."""
    if not path:
        return {}
    maxabs = json.load(open(path))
    return {t: (float(v) / ACT_QMAX if v > 0 else 1.0) for t, v in maxabs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--input-scale", type=float, default=1.0)
    ap.add_argument("--output-scale", type=float, default=1.0)
    ap.add_argument("--mode", choices=["mult", "dequant"], default="mult",
                    help="mult: QMAX/max|w|; dequant: max|w|/QMAX")
    ap.add_argument("--qmax", type=float, default=127.0,
                    help="symmetric int8 full scale (127 = 2**(b-1)-1, "
                         "128 = 2**(b-1) as in ctc_test.tinyx3.onnx)")
    ap.add_argument("--exp-scale", type=int, default=2,
                    help="Softmax output gain used when no calibrated scale is "
                         "available; exp_scale itself is written as "
                         "output_scale / input_scale")
    ap.add_argument("--act-maxabs", default="",
                    help="JSON {tensor: max|value|} used for activation scales")
    args = ap.parse_args()

    m = onnx.load(args.src)
    g = m.graph
    inits = {i.name: i for i in g.initializer}

    act_scale = load_act_scales(args.act_maxabs)

    # tensor name -> quantization scale, filled in topological order
    scale_of = {}
    if "input" in act_scale:
        scale_of["input"] = act_scale["input"]
    counts = {"conv": 0, "pass": 0, "softmax": 0, "skipped": 0, "calibrated": 0}

    def upstream_scale(node):
        for t in node.input:
            if t in scale_of:
                return scale_of[t]
        return args.input_scale

    for node in g.node:
        if node.op_type == "Conv":
            w = numpy_helper.to_array(inits[node.input[1]])
            out_ch = w.shape[0]
            maxabs = np.abs(w.reshape(out_ch, -1)).max(axis=1)
            if args.mode == "mult":
                scales = np.where(maxabs > 0, args.qmax / np.maximum(maxabs, 1e-12), 1.0)
            else:
                scales = np.where(maxabs > 0, maxabs / args.qmax, 1.0)

            in_s = upstream_scale(node)
            out_s = act_scale.get(node.output[0],
                                  get_attr(node, "output_scale", args.output_scale))
            if node.output[0] in act_scale:
                counts["calibrated"] += 1

            keep = [a for a in node.attribute if a.name not in CONV_ATTRS]
            del node.attribute[:]
            node.attribute.extend(keep)
            node.attribute.extend([
                onnx.helper.make_attribute("input_bitdepth", args.bits),
                onnx.helper.make_attribute("input_scale", float(in_s)),
                onnx.helper.make_attribute("output_bitdepth", args.bits),
                onnx.helper.make_attribute("output_scale", float(out_s)),
                onnx.helper.make_attribute("weight_bitdepth", args.bits),
                onnx.helper.make_attribute("weight_ch_scales", [float(x) for x in scales]),
            ])
            scale_of[node.output[0]] = float(out_s)
            counts["conv"] += 1

        elif node.op_type in PASSTHROUGH_OPS:
            in_s = upstream_scale(node)
            set_attr(node, "output_bitdepth", args.bits)
            set_attr(node, "output_scale", float(in_s))
            scale_of[node.output[0]] = float(in_s)
            counts["pass"] += 1

        elif node.op_type == "Softmax":
            in_s = upstream_scale(node)
            out_s = act_scale.get(node.output[0], float(in_s) * float(args.exp_scale))
            if node.output[0] in act_scale:
                counts["calibrated"] += 1
            ratio = float(out_s) / float(in_s) if in_s else float(args.exp_scale)
            exp_scale = int(round(ratio)) if abs(ratio - round(ratio)) < 1e-9 else ratio
            set_attr(node, "input_bitdepth", args.bits)
            set_attr(node, "input_scale", float(in_s))
            set_attr(node, "output_bitdepth", args.bits)
            set_attr(node, "output_scale", float(out_s))
            set_attr(node, "exp_scale", exp_scale)
            scale_of[node.output[0]] = float(out_s)
            counts["softmax"] += 1

        else:
            counts["skipped"] += 1

    onnx.save(m, args.dst)
    print("[attach] %d Conv, %d pass-through, %d Softmax, %d skipped, "
          "%d activation scales applied -> %s "
          "(mode=%s, bits=%d, qmax=%g, exp_scale=%d)"
          % (counts["conv"], counts["pass"], counts["softmax"], counts["skipped"],
             counts["calibrated"], args.dst, args.mode, args.bits, args.qmax,
             args.exp_scale))


if __name__ == "__main__":
    main()
