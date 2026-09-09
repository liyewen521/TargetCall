import csv
from pathlib import Path
from collections import Counter

BASE = Path('/Projects/ACCESS_IVFS/hbv_test')

gt = set()
for line in (BASE/'gt.sam').read_text().splitlines():
    if line.startswith('@'):
        continue
    p = line.split('\t')
    if len(p) > 2 and p[2] != '*':
        gt.add(p[0])

hf = set(l.strip() for l in (BASE/'results'/'readids.h32.txt').read_text().splitlines() if l.strip())
fn = gt - hf

rows = {}
with open(BASE/'results'/'decisions.h32.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'):
        rows[r['read_id']] = r

c = Counter()
for rid in fn:
    r = rows.get(rid)
    if r:
        c[(r['coherent_hits'], r['initial_decision'], r['final_decision'])] += 1
    else:
        c[('MISSING',)] += 1

print('FN 的 (coherent_hits, initial_decision, final_decision) 分布:')
for k, v in sorted(c.items(), key=lambda x: -x[1]):
    print(f'  {k}: {v}')
