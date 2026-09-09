#!/usr/bin/env python3
"""Run an ONNX model with dummy signal input and show the output."""
import sys
import numpy as np
import onnxruntime as ort

path = sys.argv[1] if len(sys.argv) > 1 else '/Projects/ACCESS_IVFS/onnx_models/TINYX011.onnx'
length = int(sys.argv[2]) if len(sys.argv) > 2 else 4000

sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
inp = sess.get_inputs()[0]
out = sess.get_outputs()[0]
print(f"Model: {path}")
print(f"  input:  {inp.name} {inp.shape} ({inp.type})")
print(f"  output: {out.name} {out.shape} ({out.type})")

# dummy signal: batch=1, channels=1, length=length
x = np.random.randn(1, 1, length).astype(np.float32)
y = sess.run(None, {inp.name: x})[0]

print(f"\n输入 shape: {x.shape}")
print(f"输出 shape: {y.shape}  (time, batch, 5 碱基)")
print(f"输出数值范围: [{y.min():.3f}, {y.max():.3f}]  (log-prob, 应为负值)")
print(f"输出第 0 个时间步的 5 个碱基 log-prob (N,A,C,G,T):")
print(f"  {y[0,0,:]}")
print(f"\n输出总时间步 = {y.shape[0]} (输入 {length} / stride 3 再取整)")
