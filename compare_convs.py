import onnx, sys, os
from collections import defaultdict

def get_conv_info(model_path, label):
    m = onnx.load(model_path)
    convs = []
    for node in m.graph.node:
        if node.op_type != 'Conv':
            continue
        # Find weight initializer
        w_name = node.input[1]
        w = None
        for init in m.graph.initializer:
            if init.name == w_name:
                w = init
                break
        if w is None:
            continue
        out_ch = w.dims[0]
        in_ch = w.dims[1]
        kernel = w.dims[2]

        # Get group from attributes
        group = 1
        for attr in node.attribute:
            if attr.name == 'group':
                group = attr.i
                break

        convs.append({
            'out_ch': out_ch,
            'in_ch': in_ch,
            'kernel': kernel,
            'group': group,
        })
    return convs

MODELS = {
    'bonito_default': '/Projects/ACCESS_IVFS/onnx_models/bonito_default.onnx',
    'TINYX0111':      '/Projects/ACCESS_IVFS/onnx_models/TINYX0111.onnx',
    'TINYX011':       '/Projects/ACCESS_IVFS/onnx_models/TINYX011.onnx',
    'TINYX01':        '/Projects/ACCESS_IVFS/onnx_models/TINYX01.onnx',
    'TINYX2':         '/Projects/ACCESS_IVFS/onnx_models/TINYX2.onnx',
    'TINYX3':         '/Projects/ACCESS_IVFS/onnx_models/TINYX3.onnx',
}

all_data = {}
for name, path in MODELS.items():
    all_data[name] = get_conv_info(path, name)

# Print header
print(f"{'Model':<16} {'Type':<11} {'Layer#':>6} {'InCh':>5} {'OutCh':>5} {'K':>4} {'Grp':>5}")
print("-" * 58)

for name, convs in all_data.items():
    for i, c in enumerate(convs):
        if c['group'] > 1:
            typ = "grouped"
        elif c['kernel'] == 1:
            typ = "pointwise"
        else:
            typ = "standard"
        print(f"{name:<16} {typ:<11} {i:>5}  {c['in_ch']:>4}  {c['out_ch']:>4}  {c['kernel']:>3}  {c['group']:>4}")

# Summary: min/max channels per type
print("\n" + "="*58)
print("CHANNEL RANGE PER CONV TYPE")
print(f"{'Model':<16} {'Pointwise in→out':<20} {'Grouped in→out':<20} {'Standard in→out':<20}")
print("-"*76)
for name, convs in all_data.items():
    pw  = [(c['in_ch'], c['out_ch']) for c in convs if c['kernel'] == 1 and c['group'] == 1]
    grp = [(c['in_ch'], c['out_ch']) for c in convs if c['group'] > 1]
    std = [(c['in_ch'], c['out_ch']) for c in convs if c['kernel'] > 1 and c['group'] == 1]

    pw_str  = f"{min(min(pw))}→{max(max(pw))}" if pw else "—"
    grp_str = f"{min(min(grp))}→{max(max(grp))}" if grp else "—"
    std_str = f"{min(min(std))}→{max(max(std))}" if std else "—"
    print(f"{name:<16} {pw_str:<20} {grp_str:<20} {std_str:<20}")

print("\n" + "="*58)
print("BONITO vs LIGHTCALL RATIO (TINYX3 extreme)")
bd = all_data['bonito_default']
lx = all_data['TINYX3']
for label, b_convs, l_convs in [
    ("pointwise (k=1)", [c for c in bd if c['kernel']==1 and c['group']==1],
                        [c for c in lx if c['kernel']==1 and c['group']==1]),
    ("grouped   (g>1)", [c for c in bd if c['group']>1],
                        [c for c in lx if c['group']>1]),
    ("standard  (k>1)", [c for c in bd if c['kernel']>1 and c['group']==1],
                        [c for c in lx if c['kernel']>1 and c['group']==1]),
]:
    b_max = max(c['out_ch'] for c in b_convs) if b_convs else 0
    l_max = max(c['out_ch'] for c in l_convs) if l_convs else 0
    print(f"{label}: Bonito max={b_max}, LightCall(TINYX3) max={l_max}, ratio={b_max/l_max:.1f}x" if l_max else f"{label}: LightCall has none")
