#!/usr/bin/env python3
"""Run an ONNX model with dummy signal input and show the output.

Auto-adapts to the model's declared input rank:
  - 1D model  (BCL):   input (1, 1, length)
  - BCHW model (2D):   input (1, 1, 1, length)
"""
import sys
import numpy as np
import onnxruntime as ort

path = sys.argv[1] if len(sys.argv) > 1 else '/Projects/ACCESS_IVFS/onnx_models/TINYX011.onnx'
length = int(sys.argv[2]) if len(sys.argv) > 2 else 4000

sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
inp = sess.get_inputs()[0]
out = sess.get_outputs()[0]

# 从模型声明读取输入 shape，动态维度(None/0)用 length 替换
declared = list(inp.shape)
shape = [length if (d is None or d == 0) else d for d in declared]

print(f"Model: {path}")
print(f"  input:  {inp.name} {declared} ({inp.type})")
print(f"  output: {out.name} {out.shape} ({out.type})")

x = np.random.randn(*shape).astype(np.float32)
y = sess.run(None, {inp.name: x})[0]

print(f"\n输入 shape: {x.shape}")
print(f"输出 shape: {y.shape}")
print(f"输出数值范围: [{y.min():.3f}, {y.max():.3f}]  (log-prob, 应为负值)")
print(f"输出第 0 个时间步的碱基 log-prob:")
print(f"  {y.reshape(-1, y.shape[-1])[0]}")  # 展平到 (..., 5) 取第一个时间步
