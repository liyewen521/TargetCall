#!/usr/bin/env python3
"""
Compare hash-bit widths against ground-truth TargetCall decisions.

For each hash_bits in [48, 40, 32, 28, 24, 20, 16]:
  1. build a Monkeypox seed-hash index at that width
  2. filter the 23 real basecalled reads
  3. record accepted read IDs + collision statistics
Then compare each against ground truth (traditional TargetCall readids.txt).
"""
import sys, os
from pathlib import Path

sys.path.insert(0, '/Projects/ACCESS_IVFS')
os.chdir('/Projects/ACCESS_IVFS')

from genstore_targetcall import hybrid_filter as hf

REF = Path('/Projects/ACCESS_IVFS/TargetCall/sample_data/Monkeypox_virus.fasta')
READS = Path('/Projects/ACCESS_IVFS/targetcall-reproduction/output.fasta')
GT = Path('/Projects/ACCESS_IVFS/targetcall-reproduction/readids.txt')
OUTDIR = Path('/Projects/ACCESS_IVFS/hashbits_experiment')
OUTDIR.mkdir(exist_ok=True)

# Ground truth accepted read IDs
gt_accepted = set(line.strip() for line in GT.read_text().splitlines() if line.strip())
print(f"Ground truth (traditional TargetCall): {len(gt_accepted)} accepted reads\n")

HASH_BITS = [48, 40, 32, 28, 24, 20, 16]

for bits in HASH_BITS:
    index_path = OUTDIR / f"monkeypox.k31.h{bits}.gsi"

    # Build index at this width (in memory, no disk write needed for comparison)
    index = hf.build_index(REF, index_path, seed_length=31, hash_bits=bits)

    # Collision statistics
    hashes = index['hashes']
    n_kmers = index['indexed_kmers']
    n_unique = len(hashes)
    collisions = n_kmers - n_unique
    max_occ = max(len(v) for v in hashes.values())

    # Filter reads
    decisions, accepted = hf.filter_reads(
        index,
        READS,
        OUTDIR / f"readids.h{bits}.txt",
        REF,
        seed_stride=31,
        accept_hits=3,
        reject_hits=0,
        position_tolerance=64,
        max_hash_occurrences=1000,
        minimap2='/Projects/ACCESS_IVFS/.tools/minimap2-2.24/minimap2',
        threads=8,
        decisions_path=OUTDIR / f"decisions.h{bits}.tsv",
        gray_sam_path=OUTDIR / f"gray.h{bits}.sam",
    )

    accepted_ids = set(decision.read_id for decision in decisions
                       if decision.final_decision == 'accept')

    # Compare with ground truth
    fp = accepted_ids - gt_accepted   # false positives (we accept, GT rejects)
    fn = gt_accepted - accepted_ids   # false negatives (we reject, GT accepts)
    match = (accepted_ids == gt_accepted)

    print(f"hash_bits={bits:>2}: "
          f"k-mers={n_kmers:>7} unique={n_unique:>7} "
          f"collisions={collisions:>6} max_occ={max_occ:>3} | "
          f"accepted={len(accepted_ids):>2} "
          f"FP={len(fp)} FN={len(fn)} "
          f"{'== GT (lossless)' if match else '!= GT (DIFFERS)'}")
    if fp:
        print(f"          false positives: {sorted(fp)}")
    if fn:
        print(f"          false negatives: {sorted(fn)}")
