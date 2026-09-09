#!/usr/bin/env python3
"""
Scan seed length (fixed 32-bit hash) to see if shorter seeds recover the
131 false negatives while keeping FP=0.
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
OUT = BASE / 'seed_scan'
OUT.mkdir(exist_ok=True)

# Ground truth
gt_accepted = set()
for line in SAM.read_text().splitlines():
    if line.startswith('@'):
        continue
    p = line.split('\t')
    if len(p) > 2 and p[2] != '*':
        gt_accepted.add(p[0])

print(f"Ground truth: {len(gt_accepted)} accepted\n")

SEED_LENGTHS = [31, 25, 21, 17, 15]

for seed in SEED_LENGTHS:
    index_path = OUT / f"hbv.k{seed}.h32.gsi"
    index = hf.build_index(REF, index_path, seed_length=seed, hash_bits=32)

    decisions, accepted = hf.filter_reads(
        index,
        READS,
        OUT / f"readids.k{seed}.txt",
        REF,
        seed_stride=seed,
        accept_hits=3,
        reject_hits=0,
        position_tolerance=64,
        max_hash_occurrences=1000,
        minimap2='/Projects/ACCESS_IVFS/.tools/minimap2-2.24/minimap2',
        threads=8,
        decisions_path=OUT / f"decisions.k{seed}.tsv",
        gray_sam_path=OUT / f"gray.k{seed}.sam",
    )

    accepted_ids = set(d.read_id for d in decisions if d.final_decision == 'accept')
    fp = accepted_ids - gt_accepted
    fn = gt_accepted - accepted_ids

    print(f"seed_length={seed:>2}: accepted={len(accepted_ids):>4} "
          f"FP={len(fp):>3} FN={len(fn):>3}")
