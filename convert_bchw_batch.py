#!/usr/bin/env python3
"""Batch-convert all 6 ONNX models from 1D (BCL) to 2D (BCHW), verify argmax equivalence."""
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorShapeProto

OUT_DIR = '/Projects/ACCESS_IVFS/onnx_models'
MODELS = ['bonito_default', 'TINYX0111', 'TINYX011', 'TINYX01', 'TINYX2', 'TINYX3']


def fixed_dim(value):
    d = TensorShapeProto.Dimension()
    d.dim_value = value
    return d


def convert_1d_to_bchw(src, dst):
    m = onnx.load(src)
    g = m.graph

    # 1. Conv weights [M, C/g, K] -> [M, C/g, 1, K]
    for init in g.initializer:
        if len(init.dims) == 3:
            init.dims[:] = [init.dims[0], init.dims[1], 1, init.dims[2]]

    # 2. Conv attributes and Transpose perm
    for node in g.node:
        if node.op_type == 'Conv':
            for attr in node.attribute:
                if attr.name in ('strides', 'dilations'):
                    attr.ints[:] = [1, attr.ints[0]]
                elif attr.name == 'kernel_shape':
                    attr.ints[:] = [1, attr.ints[0]]
                elif attr.name == 'pads':
                    attr.ints[:] = [0, attr.ints[0], 0, attr.ints[1]]
        elif node.op_type == 'Transpose':
            for attr in node.attribute:
                if attr.name == 'perm':
                    old = list(attr.ints)
                    new = [3 if p == 2 else p for p in old]
                    idx = new.index(1)  # C 的位置
                    new.insert(idx, 2)  # H 插在 C 之前
                    attr.ints[:] = new

    # 3. input/output shape: insert H=1 at position 2
    for inp in g.input:
        inp.type.tensor_type.shape.dim.insert(2, fixed_dim(1))
    for out in g.output:
        out.type.tensor_type.shape.dim.insert(2, fixed_dim(1))

    onnx.save(m, dst)


for name in MODELS:
    src = f'{OUT_DIR}/{name}.onnx'
    dst = f'{OUT_DIR}/{name}_bchw.onnx'
    print(f'Converting {name} ...')
    convert_1d_to_bchw(src, dst)

    # verify argmax equivalence
    s1 = ort.InferenceSession(src, providers=['CPUExecutionProvider'])
    s2 = ort.InferenceSession(dst, providers=['CPUExecutionProvider'])
    x = np.random.randn(1, 1, 4000).astype(np.float32)
    y1 = s1.run(None, {s1.get_inputs()[0].name: x})[0]
    y2 = s2.run(None, {s2.get_inputs()[0].name: x.reshape(1, 1, 1, 4000)})[0]
    y2s = y2.reshape(y1.shape)

    mismatch = (y1.argmax(-1) != y2s.argmax(-1)).sum()
    total = y1.argmax(-1).size
    abs_diff = np.abs(y1 - y2s).max()
    status = 'OK' if mismatch == 0 else f'{mismatch} MISMATCH'
    print(f'  -> {name}_bchw.onnx  argmax={mismatch}/{total}  max|Δ|={abs_diff:.2e}  [{status}]')

print('\n全部完成')
