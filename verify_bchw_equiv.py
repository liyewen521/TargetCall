import numpy as np
import onnxruntime as ort

s1 = ort.InferenceSession('/Projects/ACCESS_IVFS/onnx_models/TINYX011.onnx', providers=['CPUExecutionProvider'])
s2 = ort.InferenceSession('/Projects/ACCESS_IVFS/onnx_models/TINYX011_bchw.onnx', providers=['CPUExecutionProvider'])

x = np.random.randn(1, 1, 4000).astype(np.float32)
y1 = s1.run(None, {s1.get_inputs()[0].name: x})[0]                    # (1334,1,5)
y2 = s2.run(None, {s2.get_inputs()[0].name: x.reshape(1,1,1,4000)})[0]  # (1334,1,1,5)

y2s = y2.reshape(y1.shape)

abs_diff = np.abs(y1 - y2s)
rel_diff = abs_diff / (np.abs(y1) + 1e-6)

print(f"log-prob absolute diff: max={abs_diff.max():.3e}  mean={abs_diff.mean():.3e}")
print(f"log-prob relative diff: max={rel_diff.max():.3e}  mean={rel_diff.mean():.3e}")
print(f"log-prob value range: [{y1.min():.3f}, {y1.max():.3f}]")

# whether the final base call (argmax) matches
arg1 = y1.argmax(-1)
arg2 = y2s.argmax(-1)
mismatch = (arg1 != arg2).sum()
total = arg1.size
print(f"\nargmax base calls: {total} positions, {mismatch} differ ({mismatch/total*100:.4f}%)")
print("=> base calls fully consistent" if mismatch == 0 else f"=> {mismatch} differ")

# whether probability distributions (softmax) match
p1 = np.exp(y1) / np.exp(y1).sum(-1, keepdims=True)
p2 = np.exp(y2s) / np.exp(y2s).sum(-1, keepdims=True)
print(f"softmax probability diff: max={np.abs(p1-p2).max():.3e}")
