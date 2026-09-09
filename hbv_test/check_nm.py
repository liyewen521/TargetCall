from pathlib import Path
import re
from collections import Counter

BASE = Path('/Projects/ACCESS_IVFS/hbv_test')

gt = set()
sam_records = {}
for line in (BASE/'gt.sam').read_text().splitlines():
    if line.startswith('@'):
        continue
    p = line.split('\t')
    if len(p) < 12:
        continue
    if p[2] != '*':
        gt.add(p[0])
        sam_records.setdefault(p[0], []).append(p)

hf = set(l.strip() for l in (BASE/'results'/'readids.h32.txt').read_text().splitlines() if l.strip())
fn = gt - hf
tp = gt - fn  # accepted by both

def best_nm(rid):
    best = None
    for p in sam_records.get(rid, []):
        nm = 0
        for tag in p[11:]:
            if tag.startswith('NM:i:'):
                nm = int(tag[5:])
        alen = sum(int(n) for n in re.findall(r'(\d+)M', p[5]))
        if best is None or alen > best[0]:
            best = (alen, nm)
    return best or (0, 0)

def mismatch_rate(rid):
    alen, nm = best_nm(rid)
    return nm / alen if alen else 0

fn_rates = [mismatch_rate(r) for r in fn]
tp_rates = [mismatch_rate(r) for r in tp]

def stats(name, rates):
    rates = sorted(rates)
    n = len(rates)
    med = rates[n//2] if n else 0
    print(f"{name}: n={n}  median mismatch rate={med:.3f}  "
          f"mean={sum(rates)/n:.3f}  max={max(rates):.3f}" if n else f"{name}: empty")

print("错配率 (NM / aligned length) 对比:")
stats("FN (hash filter 漏掉)", fn_rates)
stats("TP (两者都接受)  ", tp_rates)

# Bucket distribution of FN mismatch rates
buckets = Counter()
for r in fn_rates:
    b = '>=15%' if r >= 0.15 else ('10-15%' if r >= 0.10 else ('5-10%' if r >= 0.05 else '<5%'))
    buckets[b] += 1
print("\nFN 错配率分布:")
for b in ['<5%', '5-10%', '10-15%', '>=15%']:
    print(f"  {b}: {buckets.get(b, 0)}")
