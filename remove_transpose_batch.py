#!/usr/bin/env python3
"""Batch-remove trailing Transpose before Softmax -> batch-first output [B,C,1,L]."""
import numpy as np
import onnx
import onnxruntime as ort

OUT_DIR = '/Projects/ACCESS_IVFS/onnx_models'
MODELS = ['bonito_default', 'TINYX0111', 'TINYX011', 'TINYX01', 'TINYX2', 'TINYX3']


def remove_transpose(src, dst):
    m = onnx.load(src)
    g = m.graph

    trans = [n for n in g.node if n.op_type == 'Transpose']
    softmax = [n for n in g.node if n.op_type == 'Softmax']
    assert len(trans) == 1 and len(softmax) == 1, f"Transpose={len(trans)} Softmax={len(softmax)}"
    tn, sn = trans[0], softmax[0]

    # rewire Softmax input to Transpose's input
    assert sn.input[0] == tn.output[0]
    sn.input[0] = tn.input[0]

    # axis -1 -> 1
    for attr in sn.attribute:
        if attr.name == 'axis':
            attr.i = 1

    g.node.remove(tn)

    # output shape [L,B,1,C] -> [B,C,1,L] (inverse perm of [3,0,2,1])
    for out in g.output:
        dims = list(out.type.tensor_type.shape.dim)
        reordered = [dims[i] for i in [1, 3, 2, 0]]
        del out.type.tensor_type.shape.dim[:]
        out.type.tensor_type.shape.dim.extend(reordered)

    onnx.save(m, dst)


for name in MODELS:
    src = f'{OUT_DIR}/{name}_bchw_softmax.onnx'
    dst = f'{OUT_DIR}/{name}_bchw_softmax_notranspose.onnx'
    remove_transpose(src, dst)

    # verify
    s_old = ort.InferenceSession(src, providers=['CPUExecutionProvider'])
    s_new = ort.InferenceSession(dst, providers=['CPUExecutionProvider'])
    x = np.random.randn(1, 1, 1, 4000).astype(np.float32)
    y_old = s_old.run(None, {s_old.get_inputs()[0].name: x})[0]
    y_new = s_new.run(None, {s_new.get_inputs()[0].name: x})[0]
    diff = np.abs(y_new - y_old.transpose(1, 3, 2, 0)).max()
    ok = np.allclose(y_new, y_old.transpose(1, 3, 2, 0), atol=1e-5)
    print(f'{name:16s}  {tuple(y_old.shape)} -> {tuple(y_new.shape)}  max|Δ|={diff:.2e}  {"OK" if ok else "DIFF"}')

print('\n全部完成')
