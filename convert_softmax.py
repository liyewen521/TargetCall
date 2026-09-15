#!/usr/bin/env python3
"""
Replace the final LogSoftmax with plain Softmax (for hardware that lacks LogSoftmax).
Pure graph transform: no retraining. argmax is unchanged (log is monotonic).

Verify:
  1. argmax(log_softmax) == argmax(softmax)
  2. softmax(x) == exp(log_softmax(x))   (mathematical equivalence)
"""
import numpy as np
import onnx
import onnxruntime as ort

OUT_DIR = '/Projects/ACCESS_IVFS/onnx_models'
MODELS = ['bonito_default', 'TINYX0111', 'TINYX011', 'TINYX01', 'TINYX2', 'TINYX3']


def replace_logsoftmax(src, dst):
    m = onnx.load(src)
    replaced = 0
    for node in m.graph.node:
        if node.op_type == 'LogSoftmax':
            node.op_type = 'Softmax'
            replaced += 1
    onnx.save(m, dst)
    return replaced


for name in MODELS:
    src = f'{OUT_DIR}/{name}_bchw.onnx'
    dst = f'{OUT_DIR}/{name}_bchw_softmax.onnx'
    n = replace_logsoftmax(src, dst)
    print(f'{name}: replaced {n} LogSoftmax -> Softmax, saved {name}_bchw_softmax.onnx')

    # verify
    s_log = ort.InferenceSession(src, providers=['CPUExecutionProvider'])
    s_sm = ort.InferenceSession(dst, providers=['CPUExecutionProvider'])
    x = np.random.randn(1, 1, 1, 4000).astype(np.float32)
    y_log = s_log.run(None, {s_log.get_inputs()[0].name: x})[0]   # log-prob
    y_sm = s_sm.run(None, {s_sm.get_inputs()[0].name: x})[0]      # prob

    argmax_same = (y_log.argmax(-1) == y_sm.argmax(-1)).all()
    # softmax == exp(log_softmax)
    max_dev = np.abs(y_sm - np.exp(y_log)).max()
    print(f'  argmax 一致: {argmax_same}   |softmax-exp(logsoftmax)|={max_dev:.2e}')

print('\n全部完成')
