import re
from pathlib import Path
from collections import Counter

BASE = Path('/Projects/ACCESS_IVFS/hbv_test')
SAM = BASE / 'gt.sam'

gt = set()
sam_records = {}
for line in SAM.read_text().splitlines():
    if line.startswith('@'):
        continue
    p = line.split('\t')
    if len(p) < 12:
        continue
    if p[2] != '*':
        gt.add(p[0])
        sam_records.setdefault(p[0], []).append(p)

# 0-hit reads (131 FN + 5 off-target)
import csv
zero_hit = set()
with open(BASE/'seed_scan'/'decisions.k31.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'):
        if r['initial_decision'] == 'reject':
            zero_hit.add(r['read_id'])

fn = zero_hit & gt  # 131 true HBV that were 0-hit

def best_align(rid):
    best = None
    for p in sam_records.get(rid, []):
        alen = sum(int(n) for n in re.findall(r'(\d+)M', p[5]))
        if best is None or alen > best[0]:
            nm = 0
            for tag in p[11:]:
                if tag.startswith('NM:i:'):
                    nm = int(tag[5:])
            best = (alen, nm, p[5])
    return best or (0, 0, '')

print("=== 131 FN 的差异构成：substitution vs indel ===")
n_sub_dominant = 0
n_indel_dominant = 0
indel_counts = []
for rid in fn:
    alen, nm, cigar = best_align(rid)
    # count indel events (D + I operations) and their total bases
    del_ops = re.findall(r'(\d+)D', cigar)
    ins_ops = re.findall(r'(\d+)I', cigar)
    n_indel_bases = sum(int(n) for n in del_ops) + sum(int(n) for n in ins_ops)
    n_indel_events = len(del_ops) + len(ins_ops)
    # NM = substitutions + indels (edit distance); approx substitutions
    n_sub = max(0, nm - n_indel_bases)
    indel_counts.append(n_indel_events)
    if n_indel_events >= n_sub * 0.5:
        n_indel_dominant += 1
    else:
        n_sub_dominant += 1

print(f"  substitution 主导: {n_sub_dominant}")
print(f"  indel 主导:       {n_indel_dominant}")

# distribution of indel events per read
dist = Counter()
for c in indel_counts:
    b = '0' if c == 0 else ('1-2' if c <= 2 else ('3-5' if c <= 5 else ('6-10' if c <= 10 else '>10')))
    dist[b] += 1
print("\n  每个 FN 的 indel 事件数分布:")
for b in ['0', '1-2', '3-5', '6-10', '>10']:
    print(f"    {b:>5}: {dist.get(b, 0)}")

print("\n  示例 CIGAR（前 5 个 FN）:")
for i, rid in enumerate(list(fn)[:5]):
    alen, nm, cigar = best_align(rid)
    print(f"    {rid}: alen={alen} NM={nm} cigar={cigar[:80]}")
