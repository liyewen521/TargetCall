import re, os, glob

base = "/Projects/ACCESS_IVFS/TargetCall/bonito/ctc"
files = sorted(glob.glob(f"{base}/modeltinynoskipx*.py"))

for fpath in files:
    fname = os.path.basename(fpath)
    print(f"\n{'='*60}")
    print(f"  {fname}")
    print(f"{'='*60}")
    with open(fpath) as fh:
        src = fh.read()

    # Find all Block/Block_repeat assignments
    pattern = r'(c\d+_b|b\d+_b)\s*=\s*(Block|Block_repeat)\(\s*(\d+),\s*(\d+),\s*activation'
    matches = list(re.finditer(pattern, src))

    # For each match, find the end of the call
    for m in matches:
        varname = m.group(1)
        btype = m.group(2)
        in_ch = m.group(3)
        out_ch = m.group(4)
        start = m.start()
        # Find the matching closing paren by counting
        depth = 0
        i = m.start()
        started = False
        while i < len(src):
            if src[i] == '(':
                depth += 1
                started = True
            elif src[i] == ')':
                depth -= 1
                if started and depth == 0:
                    break
            i += 1
        block_text = src[start:i+1]

        repeat = re.search(r'repeat\s*=\s*(\d+)', block_text)
        kernel = re.search(r'kernel_size\s*=\s*(\d+)', block_text)
        stride = re.search(r'stride\s*=\s*(\d+)', block_text)
        residual = re.search(r'residual\s*=\s*(True|False)', block_text)
        separable = re.search(r'separable\s*=\s*(True|False)', block_text)

        k = kernel.group(1) if kernel else "?"
        s = stride.group(1) if stride else "?"
        r = repeat.group(1) if repeat else "?"
        res = residual.group(1) if residual else "?"
        sep = separable.group(1) if separable else "?"

        print(f"  {varname:8s} {btype:13s} in={in_ch:>3s}  out={out_ch:>3s}  k={k:>3s}  s={s}  r={r}  res={res:5s}  sep={sep:5s}")

    # Also print the decoder features
    feat = re.search(r'self\.features\s*=\s*(\d+)', src)
    if feat:
        print(f"  {'(decoder)':8s} {'Conv1d':13s} in={feat.group(1):>3s}  out=5     k=1")
