#!/usr/bin/env python3
"""Analyze the 131 false negatives: why does hash filter reject reads minimap2 aligns?"""
from pathlib import Path
import sys

sys.path.insert(0, '/Projects/ACCESS_IVFS')
os = __import__('os')
os.chdir('/Projects/ACCESS_IVFS')

BASE = Path('/Projects/ACCESS_IVFS/hbv_test')
SAM = BASE / 'gt.sam'

# Ground truth accepted
gt_accepted = set()
sam_records = {}  # readid -> list of alignment lines
for line in SAM.read_text().splitlines():
    if line.startswith('@'):
        continue
    p = line.split('\t')
    if len(p) < 11:
        continue
    rid = p[0]
    if p[2] != '*':
        gt_accepted.add(rid)
        sam_records.setdefault(rid, []).append(p)

# Hash filter accepted at 32-bit
hf_accepted = set(
    l.strip() for l in (BASE/'results'/'readids.h32.txt').read_text().splitlines() if l.strip()
)

fn = gt_accepted - hf_accepted
print(f"ground truth accepted: {len(gt_accepted)}")
print(f"hash filter accepted:  {len(hf_accepted)}")
print(f"false negatives:       {len(fn)}\n")

# Analyze FN alignment quality
import collections
mapq_dist = collections.Counter()
alen_dist = collections.Counter()
for rid in fn:
    best = None
    for p in sam_records.get(rid, []):
        flag = int(p[1])
        mapq = int(p[4])
        # cigar-based aligned length
        cigar = p[5]
        import re
        alen = sum(int(n) for n in re.findall(r'(\d+)M', cigar))
        if best is None or alen > best[0]:
            best = (alen, mapq, flag, cigar)
    if best:
        alen, mapq, flag, cigar = best
        mapq_dist[mapq] += 1
        bucket = '>=100' if alen >= 100 else ('50-99' if alen >= 50 else ('30-49' if alen>=30 else '<30'))
        alen_dist[bucket] += 1

print("FN reads: MAPQ distribution (best alignment)")
for q in sorted(mapq_dist):
    print(f"  MAPQ={q}: {mapq_dist[q]}")

print("\nFN reads: aligned-length distribution (best alignment)")
for b in ['>=100','50-99','30-49','<30']:
    print(f"  {b:>6}: {alen_dist.get(b,0)}")

# How many FN have multiple alignments / supplementary?
mult = sum(1 for rid in fn if len(sam_records.get(rid, [])) > 1)
print(f"\nFN with >1 alignment record: {mult}")

# Show a few examples
print("\nExample FN alignments (read_id, flag, mapq, aligned_len, cigar):")
count = 0
for rid in fn:
    if count >= 8:
        break
    for p in sam_records.get(rid, []):
        cigar = p[5]
        alen = sum(int(n) for n in re.findall(r'(\d+)M', cigar))
        print(f"  {rid}  flag={p[1]} mapq={p[4]} alen={alen} cigar={cigar} pos={p[3]}")
    count += 1
