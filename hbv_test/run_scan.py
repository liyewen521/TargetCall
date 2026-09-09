#!/usr/bin/env python3
"""
HBV real-data lossless check: scan hash-bit widths and compare against
minimap2 ground truth (the traditional TargetCall decision).
"""
import sys, os
from pathlib import Path

sys.path.insert(0, '/Projects/ACCESS_IVFS')
os.chdir('/Projects/ACCESS_IVFS')

from genstore_targetcall import hybrid_filter as hf

BASE = Path('/Projects/ACCESS_IVFS/hbv_test')
REF = BASE / 'hbv.fasta'
READS = BASE / 'data' / 'combined.fastq'
SAM = BASE / 'gt.sam'
OUT = BASE / 'results'
OUT.mkdir(exist_ok=True)

# Ground truth: minimap2 aligned reads (RNAME != '*'), matching extract_filtered.py
gt_accepted = set()
for line in SAM.read_text().splitlines():
    if line.startswith('@'):
        continue
    parts = line.split('\t')
    if len(parts) > 2 and parts[2] != '*':
        gt_accepted.add(parts[0])

print(f"Ground truth (minimap2 aligned): {len(gt_accepted)} accepted of {1572} reads\n")

HASH_BITS = [48, 40, 32, 28, 24, 20, 16]

for bits in HASH_BITS:
    index_path = OUT / f"hbv.k31.h{bits}.gsi"
    index = hf.build_index(REF, index_path, seed_length=31, hash_bits=bits)

    hashes = index['hashes']
    n_kmers = index['indexed_kmers']
    n_unique = len(hashes)
    max_occ = max(len(v) for v in hashes.values())

    decisions, accepted = hf.filter_reads(
        index,
        READS,
        OUT / f"readids.h{bits}.txt",
        REF,
        seed_stride=31,
        accept_hits=3,
        reject_hits=0,
        position_tolerance=64,
        max_hash_occurrences=1000,
        minimap2='/Projects/ACCESS_IVFS/.tools/minimap2-2.24/minimap2',
        threads=8,
        decisions_path=OUT / f"decisions.h{bits}.tsv",
        gray_sam_path=OUT / f"gray.h{bits}.sam",
    )

    accepted_ids = set(d.read_id for d in decisions if d.final_decision == 'accept')

    fp = accepted_ids - gt_accepted
    fn = gt_accepted - accepted_ids
    match = (accepted_ids == gt_accepted)

    print(f"hash_bits={bits:>2}: k-mers={n_kmers:>6} unique={n_unique:>6} max_occ={max_occ:>3} | "
          f"accepted={len(accepted_ids):>4} "
          f"FP={len(fp):>3} FN={len(fn):>3} "
          f"{'== GT (lossless)' if match else '!= GT (DIFFERS)'}")
    if fp and len(fp) <= 5:
        print(f"          FP: {sorted(fp)}")
    if fn and len(fn) <= 5:
        print(f"          FN: {sorted(fn)}")
