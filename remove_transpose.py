#!/usr/bin/env python3
"""
Remove the trailing Transpose before Softmax so the model outputs batch-first
layout [B, C, 1, L] (NCDW-style) instead of time-major [L, B, 1, C].

Original:  Conv -> Transpose(perm=[3,0,2,1]) -> Softmax(axis=-1)   (time-major)
New:       Conv -> Softmax(axis=1)                                  (batch-first)

softmax still normalizes over the class dim (C), so results are identical
up to layout. Verify equivalence by re-transposing the outputs.
"""
import numpy as np
import onnx
import onnxruntime as ort

SRC = '/Projects/ACCESS_IVFS/onnx_models/TINYX011_bchw_softmax.onnx'
DST = '/Projects/ACCESS_IVFS/onnx_models/TINYX011_bchw_softmax_notranspose.onnx'

m = onnx.load(SRC)
g = m.graph

# find Transpose and Softmax nodes
trans_nodes = [n for n in g.node if n.op_type == 'Transpose']
softmax_nodes = [n for n in g.node if n.op_type == 'Softmax']
assert len(trans_nodes) == 1 and len(softmax_nodes) == 1
tn = trans_nodes[0]
sn = softmax_nodes[0]

print(f"Transpose: input={list(tn.input)} output={list(tn.output)}")
print(f"Softmax:   input={list(sn.input)} output={list(sn.output)}")

# 1. rewire Softmax input: Transpose.output -> Transpose.input
assert sn.input[0] == tn.output[0]
sn.input[0] = tn.input[0]

# 2. change Softmax axis: -1 -> 1 (class dim moved to position 1)
for attr in sn.attribute:
    if attr.name == 'axis':
        attr.i = 1
print(f"Softmax axis now: 1")

# 3. remove Transpose node
g.node.remove(tn)

# 4. update output shape: [L,B,1,C] -> [B,C,1,L]
# original output dims: [L, B, 1, C]  (time-major)
# new output dims:      [B, C, 1, L]  (batch-first)
for out in g.output:
    dims = list(out.type.tensor_type.shape.dim)
    # perm [3,0,2,1] inverse = [1,3,2,0] reorders [L,B,1,C] -> [B,C,1,L]
    new_order = [1, 3, 2, 0]
    reordered = [dims[i] for i in new_order]
    del out.type.tensor_type.shape.dim[:]
    out.type.tensor_type.shape.dim.extend(reordered)

onnx.save(m, DST)
print(f"saved: {DST}")

# --- verify equivalence ---
s_old = ort.InferenceSession(SRC, providers=['CPUExecutionProvider'])
s_new = ort.InferenceSession(DST, providers=['CPUExecutionProvider'])
x = np.random.randn(1, 1, 1, 4000).astype(np.float32)
y_old = s_old.run(None, {s_old.get_inputs()[0].name: x})[0]   # [L,B,1,C]
y_new = s_new.run(None, {s_new.get_inputs()[0].name: x})[0]   # [B,C,1,L]

print(f"\nold output shape: {y_old.shape}")
print(f"new output shape: {y_new.shape}")

# y_new[B,C,1,L] should equal y_old[L,B,1,C].transpose(1,3,2,0)
y_old_retransposed = y_old.transpose(1, 3, 2, 0)
max_diff = np.abs(y_new - y_old_retransposed).max()
print(f"max |new - old_retransposed| = {max_diff:.2e}")
print("等价（仅 layout 不同）" if np.allclose(y_new, y_old_retransposed, atol=1e-5) else "有差异！")
