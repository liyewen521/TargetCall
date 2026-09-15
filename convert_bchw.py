#!/usr/bin/env python3
"""
Convert a 1D-conv ONNX model (BCL: [B,C,L]) to 2D BCHW layout ([B,C,1,L]).
Pure shape transform: 1D Conv (kernel [M, C/g, K]) -> 2D Conv (kernel [M, C/g, 1, K]).
Weights are reused unchanged; no retraining.
"""
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorShapeProto

src = sys.argv[1] if len(sys.argv) > 1 else '/Projects/ACCESS_IVFS/onnx_models/TINYX011.onnx'
dst = sys.argv[2] if len(sys.argv) > 2 else src.replace('.onnx', '_bchw.onnx')

m = onnx.load(src)
g = m.graph

def fix_dims(dims):
    """1D kernel dims [K] -> 2D [1, K]"""
    return [1, dims[0]]

converted = 0
for init in g.initializer:
    # Conv weights have 3 dims in 1D: [M, C/g, K]
    if len(init.dims) == 3:
        init.dims[:] = [init.dims[0], init.dims[1], 1, init.dims[2]]
        converted += 1

for node in g.node:
    if node.op_type == 'Conv':
        for attr in node.attribute:
            if attr.name in ('strides', 'dilations'):
                # [S] -> [1, S]
                attr.ints[:] = [1, attr.ints[0]]
            elif attr.name == 'kernel_shape':
                # [K] -> [1, K]
                attr.ints[:] = [1, attr.ints[0]]
            elif attr.name == 'pads':
                # [W_beg, W_end] -> [H_beg, W_beg, H_end, W_end] = [0, W_beg, 0, W_end]
                attr.ints[:] = [0, attr.ints[0], 0, attr.ints[1]]
    elif node.op_type == 'Transpose':
        for attr in node.attribute:
            if attr.name == 'perm':
                # 3D perm [a,b,c] (B/C/L) -> 4D: B->0, C->1, L->3, H(=2) 插在 C 之前
                # 这样 log_softmax(dim=-1) 仍作用在碱基维度 C 上
                old = list(attr.ints)
                new = [3 if p == 2 else p for p in old]
                idx = new.index(1)  # C 的位置
                new.insert(idx, 2)  # H 插在 C 之前
                attr.ints[:] = new

# input/output shape: [B,C,L] -> [B,C,1,L]; [T,B,5] -> [T,B,1,5]
def fixed_dim(value):
    d = TensorShapeProto.Dimension()
    d.dim_value = value
    return d

for inp in g.input:
    inp.type.tensor_type.shape.dim.insert(2, fixed_dim(1))
for out in g.output:
    out.type.tensor_type.shape.dim.insert(2, fixed_dim(1))

onnx.save(m, dst)
print(f"converted {converted} Conv weight tensors 1D -> 2D")
print(f"saved: {dst}")

# verify
import onnxruntime as ort
s1 = ort.InferenceSession(src, providers=['CPUExecutionProvider'])
s2 = ort.InferenceSession(dst, providers=['CPUExecutionProvider'])
x = np.random.randn(1, 1, 4000).astype(np.float32)
y1 = s1.run(None, {s1.get_inputs()[0].name: x})[0]
y2 = s2.run(None, {s2.get_inputs()[0].name: x.reshape(1, 1, 1, 4000)})[0]
print(f"\n1D output shape: {y1.shape}")
print(f"2D output shape: {y2.shape}  (2D 版本第2维是 H=1)")
print(f"max abs diff: {np.max(np.abs(y1 - y2.reshape(y1.shape))):.3e}")
print("=> 输出完全一致" if np.allclose(y1, y2.reshape(y1.shape), atol=1e-5) else "=> 有差异！")
