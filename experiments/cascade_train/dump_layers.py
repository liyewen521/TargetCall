#!/usr/bin/env python3
"""Dump per-node activations of a quantisation-annotated ONNX model.

The model carries the scale attributes produced by attach_quant.py
(input_scale / output_scale / weight_ch_scales / exp_scale ...).  Those
attributes are unknown to onnxruntime and onnx.checker, so this script strips
them from a copy of the graph before running, then quantises each tensor with
the scale of the node that produced it.

    python dump_layers.py --idx 0 --out /tmp/layerref/input_000

Writes:
    000_input.i8.npy            int8 codes of the network input
    NNN_<node>.i8.npy/.f32.npy  the first --nodes nodes, int8 and float
    <node>.acc.i32.npy          int32 accumulator of the first Conv (if it
                                consumes the graph input directly)
    all_layers.i8.npz           int8 codes of every node, key = node name
    manifest.tsv                node -> scales -> file names
"""
import argparse
import os

import numpy as np
import onnx
import onnxruntime as ort

SCALE_ATTRS = ('input_scale', 'output_scale', 'exp_scale',
               'input_bitdepth', 'output_bitdepth', 'weight_bitdepth')


def read_annotations(path):
    """Return (model, {node name: {attr: value}})."""
    m = onnx.load(path)
    attrs = {}
    for n in m.graph.node:
        d = {}
        for a in n.attribute:
            if a.name in SCALE_ATTRS:
                d[a.name] = a.f if a.type == onnx.AttributeProto.FLOAT else a.i
            elif a.name == 'weight_ch_scales':
                d[a.name] = np.asarray(list(a.floats), dtype=np.float64)
        if d:
            attrs[n.name] = d
    return m, attrs


def runtime_session(path, out_dir):
    """Strip the custom attributes and expose every node output."""
    m = onnx.load(path)
    for n in m.graph.node:
        keep = [a for a in n.attribute if a.name not in SCALE_ATTRS
                and a.name != 'weight_ch_scales']
        del n.attribute[:]
        n.attribute.extend(keep)
    have = {o.name for o in m.graph.output}
    for n in m.graph.node:
        for o in n.output:
            if o and o not in have:
                m.graph.output.extend([onnx.ValueInfoProto(name=o)])
                have.add(o)
    p = os.path.join(out_dir, '_runtime.onnx')
    onnx.save(m, p)
    return ort.InferenceSession(p, providers=['CPUExecutionProvider'])


def qname(name):
    return name.replace('/', '_').lstrip('_')


def quantise(y, scale):
    return np.clip(np.rint(y.astype(np.float64) * scale), -128, 127).astype(np.int8)


def conv0_accumulator(model, attrs, xq):
    """int32 accumulator (bias included) of the first Conv, which reads the input."""
    node = model.graph.node[0]
    if node.op_type != 'Conv' or node.input[0] != 'input':
        return None
    ini = {i.name: onnx.numpy_helper.to_array(i) for i in model.graph.initializer}
    w = ini[node.input[1]]
    b = ini[node.input[2]] if len(node.input) > 2 else None
    a = attrs[node.name]
    s_in, s_out = a['input_scale'], a['output_scale']
    s_w = a['weight_ch_scales']

    kh, kw = w.shape[2], w.shape[3]
    wq = np.clip(np.rint(w[:, 0].astype(np.float64) * s_w[:, None, None]), -128, 127).astype(np.int32)
    xp = np.pad(xq[0, 0].astype(np.int32), ((0, 0), (kh // 2, kh // 2)))
    acc = np.zeros((wq.shape[0], xp.shape[1] - kw + 1), np.int32)
    for k in range(kh):
        for l in range(kw):
            acc += wq[:, k, l][:, None] * xp[k, l:l + acc.shape[1]][None, :]
    if b is not None:
        acc += np.rint(b.astype(np.float64) * s_in * s_w).astype(np.int32)[:, None]

    m = s_out / (s_in * s_w)                      # requant multiplier per channel
    yq = np.clip(np.rint(acc.astype(np.float64) * m[:, None]), -128, 127).astype(np.int8)
    return acc, yq, s_in, s_out, s_w, b, m


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    ap.add_argument('--model', default=os.path.join(root, 'onnx_models', 'tinyx3_ivfs.onnx'))
    ap.add_argument('--vector', default=None, help='float32 input .npy')
    ap.add_argument('--idx', type=int, default=0, help='index into test_vectors/tinyx3_ivfs_folded')
    ap.add_argument('--out', default='/tmp/layerref')
    ap.add_argument('--nodes', type=int, default=10, help='how many leading nodes get their own .npy')
    a = ap.parse_args()

    vec = a.vector or os.path.join(root, 'test_vectors', 'tinyx3_ivfs_folded',
                                   'input_%03d.npy' % a.idx)
    os.makedirs(a.out, exist_ok=True)

    model, attrs = read_annotations(a.model)
    sess = runtime_session(a.model, a.out)
    x = np.load(vec).astype(np.float32)
    val = {o.name: np.asarray(v) for o, v in zip(sess.get_outputs(), sess.run(None, {'input': x}))}

    nodes = [n for n in model.graph.node if n.op_type != 'Constant']
    s_in = attrs[nodes[0].name]['input_scale']
    xq = quantise(x, s_in)
    np.save(os.path.join(a.out, '000_input.i8.npy'), xq)

    acc = conv0_accumulator(model, attrs, xq)
    if acc is not None:
        np.save(os.path.join(a.out, '%s.acc.i32.npy' % qname(nodes[0].name)), acc[0])
        ref = quantise(val[nodes[0].output[0]], acc[3]).reshape(acc[1].shape)
        agree = float(np.mean(acc[1] == ref))
    else:
        agree = float('nan')

    allq, rows = {}, []
    for i, n in enumerate(nodes):
        y = val.get(n.output[0])
        if y is None:
            continue
        at = attrs.get(n.name, {})
        s = at.get('output_scale')
        if s:
            q = quantise(y, s)
            allq[qname(n.name)] = q
        if i < a.nodes:
            base = '%03d_%s' % (i, qname(n.name))
            np.save(os.path.join(a.out, base + '.f32.npy'), y.reshape(-1).astype(np.float32))
            if s:
                np.save(os.path.join(a.out, base + '.i8.npy'), q)
            rows.append([i, n.op_type, n.name, y.size, at.get('input_scale', ''),
                         s, base + '.i8.npy' if s else '', base + '.f32.npy'])

    np.savez_compressed(os.path.join(a.out, 'all_layers.i8.npz'), **allq)
    with open(os.path.join(a.out, 'manifest.tsv'), 'w') as f:
        f.write('idx\top\tnode\tn\tscale_in\tscale_out\tfile_i8\tfile_f32\n')
        for r in rows:
            f.write('\t'.join(str(v) for v in r) + '\n')

    print('%d nodes, %d tensors -> %s' % (len(nodes), len(allq), a.out))
    if acc is not None:
        print('conv0: s_in=%.9g s_out=%.9g' % (acc[2], acc[3]))
        print('  per-channel weight scale:', ' '.join('%.4f' % v for v in acc[4]))
        print('  requant multiplier      :', ' '.join('%.6g' % v for v in acc[6]))
        print('  bias in acc units       :', ' '.join('%+d' % v for v in
              np.rint(acc[5] * acc[2] * acc[4]).astype(int)))
        print('  drop-bias shift         :', ' '.join('%+d' % v for v in
              np.rint(acc[5] * acc[3]).astype(int)))
        print('  acc range [%d, %d]' % (acc[0].min(), acc[0].max()))
        print('  int8 conv vs float+quantise: %.2f%% identical' % (agree * 100))


if __name__ == '__main__':
    main()

