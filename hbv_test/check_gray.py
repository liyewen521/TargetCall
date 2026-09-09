import csv
from pathlib import Path
from collections import Counter

BASE = Path('/Projects/ACCESS_IVFS/hbv_test')

for seed in [31, 25, 21, 17, 15]:
    path = BASE / 'seed_scan' / f'decisions.k{seed}.tsv'
    c = Counter()
    with open(path) as f:
        for r in csv.DictReader(f, delimiter='\t'):
            c[r['initial_decision']] += 1
    total = sum(c.values())
    print(f"seed={seed:>2}: total={total}  "
          f"accept={c.get('accept',0)}  gray={c.get('gray',0)}  reject={c.get('reject',0)}  "
          f"(gray 走 minimap2 兜底)")
