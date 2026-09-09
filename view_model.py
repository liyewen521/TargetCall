#!/usr/bin/env python3
"""Print detailed layer-by-layer structure of an ONNX model."""
import sys
import onnx
from pathlib import Path

path = sys.argv[1] if len(sys.argv) > 1 else '/Projects/ACCESS_IVFS/onnx_models/TINYX011.onnx'

m = onnx.load(path)
g = m.graph

# tensor shape lookup
shapes = {}
for init in g.initializer:
    shapes[init.name] = [d for d in init.dims]
for inp in g.input:
    shapes[inp.name] = [d.dim_value for d in inp.type.tensor_type.shape.dim]
for vi in g.value_info:
    shapes[vi.name] = [d.dim_value for d in vi.type.tensor_type.shape.dim]
for out in g.output:
    shapes[out.name] = [d.dim_value for d in out.type.tensor_type.shape.dim]

def fmt(name):
    s = shapes.get(name, '?')
    return f"{name}[{','.join(str(x) if x else '?' for x in s)}]" if s != '?' else name

print(f"Model: {path}")
print(f"  inputs:  {[fmt(i.name) for i in g.input]}")
print(f"  outputs: {[fmt(o.name) for o in g.output]}")
print(f"  weights: {len(g.initializer)} tensors")
print()

print("逐层结构 (node -> input -> output):")
for i, node in enumerate(g.node):
    attrs = []
    for a in node.attribute:
        if a.name in ('group', 'strides', 'pads', 'dilations'):
            val = onnx.helper.get_attribute_value(a)
            attrs.append(f"{a.name}={val}")
    attr_str = ' '.join(attrs)
    print(f"  [{i:3d}] {node.op_type:12s} {attr_str:20s} "
          f"{' '.join(fmt(x) for x in node.input)} -> {' '.join(fmt(x) for x in node.output)}")
