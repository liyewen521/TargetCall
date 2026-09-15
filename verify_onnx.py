import onnx
import os
from pathlib import Path

for p in sorted(Path('/Projects/ACCESS_IVFS/onnx_models').glob('*.onnx')):
    m = onnx.load(str(p))
    onnx.checker.check_model(m)
    in_shape = [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]
    out_shape = [d.dim_value for d in m.graph.output[0].type.tensor_type.shape.dim]
    n_ops = len(m.graph.node)
    size = os.path.getsize(p) / 1024 / 1024
    print(f'{p.name:24s} {size:6.2f} MB  ops={n_ops:3d}  in={in_shape}  out={out_shape}')

print()
print('全部通过 onnx.checker 校验')
