#!/usr/bin/env python3
"""
Convert *_deploy.onnx output to channel-last NDWC layout [B,D,W,C] = [B,1,L,C].

Reason: the chip's softmax layouts (ndcw/ndwc/ndhw/ndcc) all have D (spatial)
in the 2nd dim, while current output [B,C,1,L] has C in the 2nd dim -> mismatch.

Fix: insert Transpose(perm=[0,2,3,1]) between Conv and Softmax, so
  Conv output [B,C,1,L] -> Transpose -> [B,1,L,C] (NDWC) -> Softmax(axis=-1)

softmax still normalizes over C (now the last dim). Results identical up to layout.
"""
import numpy as np
import onnx
from onnx import helper
import onnxruntime as ort

OUT_DIR = '/Projects/ACCESS_IVFS/onnx_models'
MODELS = ['bonito_default', 'TINYX0111', 'TINYX011', 'TINYX01', 'TINYX2', 'TINYX3']


def to_ndwc(src, dst):
    m = onnx.load(src)
    g = m.graph

    softmax = [n for n in g.node if n.op_type == 'Softmax']
    assert len(softmax) == 1
    sn = softmax[0]

    # current: Softmax(axis=1) with input = Conv output [B,C,1,L]
    conv_out = sn.input[0]
    new_tensor = conv_out + '_ndwc'

    # insert Transpose(perm=[0,2,3,1]): [B,C,1,L] -> [B,1,L,C]
    tp = helper.make_node('Transpose', inputs=[conv_out], outputs=[new_tensor],
                          perm=[0, 2, 3, 1], name='layout_to_ndwc')
    g.node.append(tp)

    # rewire Softmax to Transpose output, axis back to -1
    sn.input[0] = new_tensor
    for attr in sn.attribute:
        if attr.name == 'axis':
            attr.i = -1

    # output shape [B,C,1,L] -> [B,1,L,C]
    for out in g.output:
        dims = list(out.type.tensor_type.shape.dim)
        reordered = [dims[0], dims[2], dims[3], dims[1]]
        del out.type.tensor_type.shape.dim[:]
        out.type.tensor_type.shape.dim.extend(reordered)

    onnx.save(m, dst)


for name in MODELS:
    src = f'{OUT_DIR}/{name}_deploy.onnx'
    dst = f'{OUT_DIR}/{name}_deploy_ndwc.onnx'
    to_ndwc(src, dst)

    s_old = ort.InferenceSession(src, providers=['CPUExecutionProvider'])
    s_new = ort.InferenceSession(dst, providers=['CPUExecutionProvider'])
    x = np.random.randn(1, 1, 1, 4000).astype(np.float32)
    y_old = s_old.run(None, {s_old.get_inputs()[0].name: x})[0]   # [B,C,1,L]
    y_new = s_new.run(None, {s_new.get_inputs()[0].name: x})[0]   # [B,1,L,C]

    y_old_t = y_old.transpose(0, 2, 3, 1)  # [B,C,1,L] -> [B,1,L,C]
    diff = np.abs(y_new - y_old_t).max()
    ok = np.allclose(y_new, y_old_t, atol=1e-5)
    print(f'{name:16s}  {tuple(y_old.shape)} -> {tuple(y_new.shape)}  max|Δ|={diff:.2e}  {"OK" if ok else "DIFF"}')

print('\n全部完成')
