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

# read input shape from the model; replace dynamic dims (None/0) with length
declared = list(inp.shape)
shape = [length if (d is None or d == 0) else d for d in declared]

print(f"Model: {path}")
print(f"  input:  {inp.name} {declared} ({inp.type})")
print(f"  output: {out.name} {out.shape} ({out.type})")

x = np.random.randn(*shape).astype(np.float32)
y = sess.run(None, {inp.name: x})[0]

print(f"\ninput shape: {x.shape}")
print(f"output shape: {y.shape}")
print(f"output value range: [{y.min():.3f}, {y.max():.3f}]  (log-prob, should be negative)")
print(f"base log-prob at time step 0:")
print(f"  {y.reshape(-1, y.shape[-1])[0]}")  # flatten to (..., 5) and take the first time step
