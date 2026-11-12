"""Shared RHB-Lite reference: divisor scales, symmetric codes, quantized bias.

Convolutions accumulate integer codes exactly (float64 is exact for int32
accumulators). Requantization uses float32 scales/operations, as ACModelSight's
float32 RHB-Lite wrappers do. This is a numerical reference, not a simulation
of an undocumented hardware multiplier/shift or Softmax lookup table.
"""
import numpy as np
import onnx
import torch
import torch.nn.functional as F
from onnx import numpy_helper

QUANT_SUFFIX = ("_scale", "_scales", "_bitdepth", "_quant_dim", "_quant", "mode")


def qmax(bits):
    if bits not in (4, 8, 16):
        raise ValueError("unsupported signed bitdepth: %s" % bits)
    return (1 << (bits - 1)) - 1


def code_dtype(bits):
    return np.int8 if bits <= 8 else np.int16 if bits <= 16 else np.int32


def checked_scale(value):
    scale = np.asarray(value, dtype=np.float32)
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("scales must be finite positive divisors")
    return scale


def scalar_scale(value):
    scale = checked_scale(value)
    if scale.size != 1:
        raise ValueError("RHB-Lite activations require per-tensor scales")
    return scale.reshape(())[()]


def quantize(x, scale, bits=8):
    """Round to nearest even, saturate symmetrically, then convert to integers."""
    x = np.asarray(x, dtype=np.float32)
    if not np.all(np.isfinite(x)):
        raise ValueError("cannot quantize non-finite values")
    limit = qmax(bits)
    codes = np.clip(np.rint(x / checked_scale(scale)), -limit, limit)
    return codes.astype(code_dtype(bits))


def out_scale(attrs):
    return scalar_scale(attrs.get("output_scale", attrs.get("output_scales")))


def in_scale(attrs):
    return scalar_scale(attrs.get("input_scale", attrs.get("input_scales")))


def stripped(model):
    result = onnx.ModelProto()
    result.CopyFrom(model)
    for node in result.graph.node:
        keep = [a for a in node.attribute if not a.name.endswith(QUANT_SUFFIX)]
        del node.attribute[:]
        node.attribute.extend(keep)
    return result


def read_annotations(path):
    model = onnx.load(path)
    meta = {p.key: p.value for p in model.metadata_props}
    if meta.get("quantization.scale_convention") != "divisor":
        raise ValueError("model needs divisor-scale annotations; regenerate with attach_quant_v2.py")
    initializers = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    nodes = [(n, {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute})
             for n in model.graph.node]
    validate_annotations(nodes)
    return model, nodes, initializers


def validate_annotations(nodes):
    scales = {}
    for node, attrs in nodes:
        if node.op_type == "Constant":
            continue
        qmax(attrs["output_bitdepth"])
        so = out_scale(attrs)
        if node.op_type in ("Conv", "Softmax"):
            si = in_scale(attrs)
            qmax(attrs["input_bitdepth"])
            if node.input[0] in scales and si != scales[node.input[0]]:
                raise ValueError("%s: input scale differs from producer" % node.name)
        if node.op_type in ("Reshape", "Transpose") and so != scales[node.input[0]]:
            raise ValueError("%s: layout operation changes scale" % node.name)
        if node.op_type == "Conv":
            qmax(attrs["weight_bitdepth"])
            checked_scale(attrs.get("weight_ch_scales", attrs.get("weight_scale")))
            if len(node.input) > 2:
                qmax(attrs["bias_bitdepth"])
        scales[node.output[0]] = so


def conv_integer(node, attrs, initializers, x):
    """Return output codes and the int32 accumulator, including saturated bias."""
    si, so = in_scale(attrs), out_scale(attrs)
    w = initializers[node.input[1]]
    sw = checked_scale(attrs.get("weight_ch_scales", attrs.get("weight_scale"))).reshape(-1)
    if sw.size not in (1, w.shape[0]):
        raise ValueError("%s: wrong number of weight scales" % node.name)
    wq = quantize(w, sw.reshape(-1, 1, 1, 1), attrs["weight_bitdepth"])
    acc_scale = si * sw
    bias = None
    if len(node.input) > 2:
        bias = quantize(initializers[node.input[2]], acc_scale, attrs["bias_bitdepth"])
    pads = attrs.get("pads", [0, 0, 0, 0])
    xt = torch.from_numpy(np.asarray(x, dtype=np.float64))
    xt = F.pad(xt, (pads[1], pads[3], pads[0], pads[2]))
    acc = F.conv2d(
        xt, torch.from_numpy(wq.astype(np.float64)),
        None if bias is None else torch.from_numpy(bias.astype(np.float64)),
        stride=attrs.get("strides", [1, 1]),
        dilation=attrs.get("dilations", [1, 1]), groups=attrs.get("group", 1),
    ).numpy()
    bounds = np.iinfo(np.int32)
    if np.any(acc < bounds.min) or np.any(acc > bounds.max):
        raise OverflowError("%s: int32 accumulator overflow" % node.name)
    acc = acc.astype(np.int32)
    real = acc.astype(np.float32) * acc_scale.reshape(1, -1, 1, 1)
    return quantize(real, so, attrs["output_bitdepth"]), acc


def run_integer(nodes, initializers, x, input_name=None):
    first, attrs = next((n, d) for n, d in nodes if n.op_type != "Constant")
    input_name = input_name or first.input[0]
    limit = qmax(attrs["input_bitdepth"])
    x = np.asarray(x)
    if not np.all(np.isfinite(x)) or np.any(x != np.rint(x)) or np.any(np.abs(x.astype(np.float64)) > limit):
        raise ValueError("input codes are outside the symmetric integer range")
    values = dict(initializers)
    values[input_name] = x.astype(code_dtype(attrs["input_bitdepth"]))
    scales = {input_name: in_scale(attrs)}
    codes, accumulators = {}, {}
    for node, attrs in nodes:
        op = node.op_type
        if op == "Constant":
            values[node.output[0]] = numpy_helper.to_array(attrs["value"])
            continue
        xi = values[node.input[0]]
        if op == "Conv":
            y, accumulators[node.name] = conv_integer(node, attrs, initializers, xi)
        elif op == "Relu":
            real = np.maximum(xi, 0).astype(np.float32) * scales[node.input[0]]
            y = quantize(real, out_scale(attrs), attrs["output_bitdepth"])
        elif op == "Reshape":
            shape = np.asarray(values[node.input[1]]).astype(np.int64).tolist()
            shape = [xi.shape[i] if dim == 0 else dim for i, dim in enumerate(shape)]
            y = xi.reshape(shape)
        elif op == "Transpose":
            y = np.transpose(xi, attrs.get("perm"))
        elif op == "Softmax":
            logits = xi.astype(np.float32) * in_scale(attrs)
            prob = torch.softmax(torch.from_numpy(logits), dim=attrs.get("axis", -1)).numpy()
            y = quantize(prob, out_scale(attrs), attrs["output_bitdepth"])
        else:
            raise ValueError("unsupported op %s at %s" % (op, node.name))
        values[node.output[0]] = y
        scales[node.output[0]] = out_scale(attrs)
        codes[node.name] = y
    return values[nodes[-1][0].output[0]], codes, accumulators
