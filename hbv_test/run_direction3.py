#!/usr/bin/env python3
"""
Direction-3 simulation: CAM fuzzy match as fallback for 31-mer 0-hit reads.
Compute per-seed minimum Hamming distance once, then scan d/threshold.
"""
import sys, csv
from pathlib import Path

sys.path.insert(0, '/Projects/ACCESS_IVFS')

BASE = Path('/Projects/ACCESS_IVFS/hbv_test')
REF = BASE / 'hbv.fasta'
READS = BASE / 'data' / 'combined.fastq'
SAM = BASE / 'gt.sam'
DEC31 = BASE / 'seed_scan' / 'decisions.k31.tsv'

# ground truth
gt = set()
for line in SAM.read_text().splitlines():
    if line.startswith('@'):
        continue
    p = line.split('\t')
    if len(p) > 2 and p[2] != '*':
        gt.add(p[0])

ref_seq = ''.join(l.strip() for l in REF.read_text().splitlines()
                  if not l.startswith('>')).upper()

B2B = {'A': 0b1000, 'C': 0b0100, 'G': 0b0010, 'T': 0b0001}

def encode(seq):
    code = 0
    for b in seq:
        code = (code << 4) | B2B.get(b, 0)
    return code

def hamming(a, b):
    return (a ^ b).bit_count() // 2

K = 31
ref_kmers = []
for i in range(0, len(ref_seq) - K + 1):
    seed = ref_seq[i:i+K]
    if 'N' in seed:
        continue
    ref_kmers.append(encode(seed))
print(f"reference {K}-mers: {len(ref_kmers)}")

# read sequences
read_seqs = {}
with READS.open() as f:
    while True:
        hdr = f.readline().strip()
        if not hdr:
            break
        seq = f.readline().strip()
        f.readline()
        f.readline()
        read_seqs[hdr[1:].split()[0]] = seq.upper()

# 0-hit reads
zero_hit = []
with DEC31.open() as f:
    for r in csv.DictReader(f, delimiter='\t'):
        if r['initial_decision'] == 'reject':
            zero_hit.append(r['read_id'])

print(f"0-hit reads: {len(zero_hit)} "
      f"(true HBV={sum(1 for r in zero_hit if r in gt)}, "
      f"off-target={sum(1 for r in zero_hit if r not in gt)})")

# per-seed minimum Hamming distance to any reference k-mer
min_dists = {}
for rid in zero_hit:
    seq = read_seqs[rid]
    dists = []
    for i in range(0, len(seq) - K + 1, K):
        seed = seq[i:i+K]
        if 'N' in seed:
            continue
        code = encode(seed)
        md = K + 1
        for ref in ref_kmers:
            d = hamming(code, ref)
            if d < md:
                md = d
                if md == 0:
                    break
        dists.append(md)
    min_dists[rid] = dists

print("\n=== CAM fallback: recover FN with Hamming <= d, accept if hits >= thr ===")
print(f"{'d':>2} {'thr':>3} | {'recovered':>9} {'FP':>3} {'FN':>3} | result")
best = None
for d in [0, 1, 2, 3, 4, 5, 6, 7, 9]:
    for thr in [1, 2, 3, 5]:
        rec = fp = fn = 0
        for rid in zero_hit:
            hits = sum(1 for md in min_dists[rid] if md <= d)
            accept = hits >= thr
            is_hbv = rid in gt
            if accept:
                if is_hbv:
                    rec += 1
                else:
                    fp += 1
            elif is_hbv:
                fn += 1
        tag = ""
        if fp == 0 and fn == 0:
            tag = "  <<< LOSSLESS (FP=0, FN=0)"
            if best is None:
                best = (d, thr)
        print(f"{d:>2} {thr:>3} | {rec:>9} {fp:>3} {fn:>3} |{tag}")

if best:
    print(f"\nBest lossless config: d={best[0]}, threshold={best[1]}")
else:
    print("\nNo (d, thr) achieves both FP=0 and FN=0 in this scan.")
