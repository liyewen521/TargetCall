# GenStore–TargetCall hybrid experiment

This directory contains a software prototype that replaces most of
TargetCall's whole-read minimap2 work with a GenStore-style seed hash lookup.
It does not modify either upstream Git submodule.

## Pipeline

1. Use TargetCall's lightweight model to produce noisy reads.
2. Split each read into non-overlapping canonical seeds (31 bp by default).
3. Query a 48-bit, dual-strand reference hash index.
4. Aggregate hits by read and require approximately collinear reference
   positions.
5. Accept reads with at least three coherent seeds, reject reads with no
   coherent seed, and send the 1–2 hit gray zone to minimap2.
6. Write `readids.txt` directly. A SAM file is not created for reads decided
   by the hash filter.

The hash function reproduces GenStore's x86 MD5/canonical-k-mer/truncation
scheme. Unlike the original GenStore utilities, this adapter retains read IDs
and reference positions so it can make one decision per variable-length read.

## Build an index

From the repository root:

```bash
python -m genstore_targetcall.hybrid_filter index \
  TargetCall/sample_data/Monkeypox_virus.fasta \
  targetcall-reproduction/Monkeypox_virus.k31.gsi
```

## Filter existing TargetCall output

```bash
python -m genstore_targetcall.hybrid_filter filter \
  targetcall-reproduction/Monkeypox_virus.k31.gsi \
  targetcall-reproduction/output.fasta \
  targetcall-reproduction/readids.genstore.txt \
  --minimap2 .tools/minimap2-2.24/minimap2 \
  --decisions targetcall-reproduction/genstore-decisions.tsv \
  --gray-sam targetcall-reproduction/genstore-gray.sam
```

The input reader accepts standard FASTA, standard FASTQ, and TargetCall's
two-line `@`-header output format.

## Run the full hybrid TargetCall flow

This replaces TargetCall's `minimap2 -> output.sam -> extract_filtered.py`
stage with the GenStore-style filter:

```bash
python -m genstore_targetcall.hybrid_filter pipeline \
  TargetCall/sample_data/fast5 \
  TargetCall/sample_data/Monkeypox_virus.fasta \
  TINYX011 \
  targetcall-genstore-run \
  --index targetcall-reproduction/Monkeypox_virus.k31.gsi \
  --minimap2 .tools/minimap2-2.24/minimap2
```

Outputs:

- `output.fasta`: noisy low-accuracy basecalls.
- `readids.txt`: accepted read IDs, written directly by the hash filter.
- `genstore-decisions.tsv`: per-read seed counts and final decisions.
- `output.gray.sam`: minimap2 alignments only for gray-zone reads.

`output.fastq` is deleted after conversion unless `--keep-fastq` is passed.

## Current scope

This is a correctness and threshold-calibration prototype. Its compressed
Python index is suitable for the supplied viral reference, but not yet for a
multi-gigabase human reference. A production-scale version should reuse
GenStore's compact sorted binary index and move per-read aggregation to C++.
