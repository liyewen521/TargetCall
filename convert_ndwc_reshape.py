#!/usr/bin/env python3
"""
Convert output to NDWC layout using Reshape+Transpose (not bare Transpose),
so the chip's layout inference can actually see the change.

Current:  Conv -> Softmax(axis=1), output [N,C,1,L] (NCHW, H=1)
New:      Conv -> Reshape[N,C,L] -> Transpose[N,L,C] -> Reshape[N,1,L,C] -> Softmax(axis=-1)

Steps:
  1. Reshape [N,C,1,L] -> [N,C,L]   (merge H(=1) into W)
  2. Transpose [N,C,L] -> [N,L,C]   (move C to last)
  3. Reshape [N,L,C] -> [N,1,L,C]   (add D=1 back -> NDWC)
  4. Softmax axis=-1
"""
import numpy as np
import onnx
from onnx import helper, numpy_helper
import onnxruntime as ort

OUT_DIR = '/Projects/ACCESS_IVFS/onnx_models'
MODELS = ['bonito_default', 'TINYX0111', 'TINYX011', 'TINYX01', 'TINYX2', 'TINYX3']


def to_ndwc_reshape(src, dst):
    m = onnx.load(src)
    g = m.graph

    softmax = [n for n in g.node if n.op_type == 'Softmax']
    assert len(softmax) == 1
    sn = softmax[0]
    conv_out = sn.input[0]

    # read output shape [N, C, 1, L]
    out_dims = [d.dim_value for d in g.output[0].type.tensor_type.shape.dim]
    N, C, H, L = out_dims
    assert H == 1, f"expected H=1, got {H}"

    t1 = conv_out + '_merged3d'   # [N, C, L]
    t2 = conv_out + '_clast'      # [N, L, C]
    t3 = conv_out + '_ndwc'       # [N, 1, L, C]

    # Reshape1: [N,C,1,L] -> [N,C,L]
    sh1 = numpy_helper.from_array(np.array([N, C, L], dtype=np.int64), name=conv_out + '_shape3d')
    g.initializer.append(sh1)
    g.node.append(helper.make_node('Reshape', inputs=[conv_out, sh1.name], outputs=[t1], name='reshape_merge_hw'))

    # Transpose: [N,C,L] -> [N,L,C]
    g.node.append(helper.make_node('Transpose', inputs=[t1], outputs=[t2], perm=[0, 2, 1], name='transpose_clast'))

    # Reshape2: [N,L,C] -> [N,1,L,C]
    sh2 = numpy_helper.from_array(np.array([N, 1, L, C], dtype=np.int64), name=conv_out + '_shapendwc')
    g.initializer.append(sh2)
    g.node.append(helper.make_node('Reshape', inputs=[t2, sh2.name], outputs=[t3], name='reshape_ndwc'))

    # rewire Softmax to t3, axis=-1
    sn.input[0] = t3
    for attr in sn.attribute:
        if attr.name == 'axis':
            attr.i = -1

    # output shape [N,C,1,L] -> [N,1,L,C]
    for out in g.output:
        dims = list(out.type.tensor_type.shape.dim)
        reordered = [dims[0], dims[2], dims[3], dims[1]]
        del out.type.tensor_type.shape.dim[:]
        out.type.tensor_type.shape.dim.extend(reordered)

    onnx.save(m, dst)


for name in MODELS:
    src = f'{OUT_DIR}/{name}_deploy.onnx'
    dst = f'{OUT_DIR}/{name}_deploy_ndwc2.onnx'
    to_ndwc_reshape(src, dst)

    s_old = ort.InferenceSession(src, providers=['CPUExecutionProvider'])
    s_new = ort.InferenceSession(dst, providers=['CPUExecutionProvider'])
    x = np.random.randn(1, 1, 1, 4000).astype(np.float32)
    y_old = s_old.run(None, {s_old.get_inputs()[0].name: x})[0]   # [N,C,1,L]
    y_new = s_new.run(None, {s_new.get_inputs()[0].name: x})[0]   # [N,1,L,C]

    y_old_t = y_old.transpose(0, 2, 3, 1)  # -> [N,1,L,C]
    diff = np.abs(y_new - y_old_t).max()
    ok = np.allclose(y_new, y_old_t, atol=1e-5)
    print(f'{name:16s}  {tuple(y_old.shape)} -> {tuple(y_new.shape)}  max|Δ|={diff:.2e}  {"OK" if ok else "DIFF"}')

print('\n全部完成')
