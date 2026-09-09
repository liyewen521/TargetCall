import onnx, sys
from collections import Counter

path = sys.argv[1] if len(sys.argv) > 1 else '/Projects/ACCESS_IVFS/onnx_models/TINYX3.onnx'
m = onnx.load(path)
onnx.checker.check_model(m)
print(f"Model: {path}")
print(f"Opset: {m.opset_import[0].version}")
print()
print('Inputs:')
for i in m.graph.input:
    shape = [d.dim_value for d in i.type.tensor_type.shape.dim]
    print(f'  {i.name}: {shape}')
print('Outputs:')
for o in m.graph.output:
    shape = [d.dim_value for d in o.type.tensor_type.shape.dim]
    print(f'  {o.name}: {shape}')
print(f'\nTotal ops: {len(m.graph.node)}')
op_types = Counter(n.op_type for n in m.graph.node)
print('Op breakdown:')
for op, cnt in op_types.most_common():
    print(f'  {op}: {cnt}')
