import re
with open('TargetCall/bonito/models/default/config.toml') as f:
    src = f.read()
blocks = re.findall(r'\[\[block\]\](.*?)(?=\[\[block\]\]|\[model\]|\[labels\]|\[input\]|\[encoder\])', src, re.DOTALL)
print('Bonito Default (9739K params):')
prev = 1
for i, b in enumerate(blocks):
    fch = re.search(r'filters\s*=\s*(\d+)', b)
    rep = re.search(r'repeat\s*=\s*(\d+)', b)
    ker = re.search(r'kernel\s*=\s*\[(.+?)\]', b)
    st  = re.search(r'stride\s*=\s*\[(.+?)\]', b)
    res = re.search(r'residual\s*=\s*(true|false)', b)
    sep = re.search(r'separable\s*=\s*(true|false)', b)
    fc = fch.group(1) if fch else '?'
    print(f'  Block {i}: {prev:>3d} -> {fc:>3s}  k={ker.group(1) if ker else "?":>4s}  s={st.group(1) if st else "?":>2s}  r={rep.group(1) if rep else "?"}  res={res.group(1) if res else "?":>5s}  sep={sep.group(1) if sep else "?":>5s}')
    prev = int(fc) if fch else prev
print(f'  Decoder: Conv1d({prev} -> 5) -> log_softmax')
print(f'  Activation: ReLU')
print(f'  Residual skip connections: ENABLED')
