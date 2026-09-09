# TargetCall sample reproduction

Reproduced on 2026-08-10 from TargetCall commit
`195b6c17af0bb8bf6326496ff0f4e23ae9485867`.

## Scope

This is the sample workflow from the repository README, not the full paper
benchmark. It uses:

- input: `TargetCall/sample_data/fast5/`
- reference: `TargetCall/sample_data/Monkeypox_virus.fasta`
- model: `TINYX011`
- GPU: one NVIDIA H20 (`CUDA_VISIBLE_DEVICES=0`)

## Environment

- Python 3.8.20
- Bonito / TargetCall 0.5.0
- Minimap2 2.24-r1122
- PyTorch 2.1.2+cu121

The persistent virtual environment is stored at
`/Projects/ACCESS_IVFS/.venv-targetcall`. Activate it with:

```bash
source /Projects/ACCESS_IVFS/.venv-targetcall/bin/activate
export PATH="/Projects/ACCESS_IVFS/.tools/minimap2-2.24:$PATH"
```

The repository pins PyTorch 1.10.0+cu113, but that build only contains GPU
kernels through `sm_86`. The available NVIDIA H20 is `sm_90`, so the pinned
build fails with `no kernel image is available for execution on the device`.
PyTorch was therefore upgraded to 2.1.2+cu121 while keeping the TargetCall
source, model weights, and remaining runtime flow unchanged.

## Results

- basecalled records: 23
- SAM alignment records: 24 (one read has a secondary alignment)
- unique SAM reads: 23
- unique mapped/accepted reads: 8
- Bonito-reported throughput: approximately 1.5e6 samples/second

`output.fastq` is an intermediate file. The repository's
`src/fastq_to_fasta.py` converts it into two-line records in `output.fasta`
and then deletes it.

## Output hashes

```text
22ae45f3df6fe5b10f92d76bba4a3665d2d94b589894bb9408b86b1f5e59c6b2  output.fasta
0d2f4e2b94dc3af015bdf60471888c6998532e59322baa5b3903ed7b9a9968e8  output.sam
25b1b9e0980a02a37d68535072cab95d2a56f2ba1ef71cfef1bf7118520ef3ad  readids.txt
1afe3e5826a3150f0b7da7ac33c3daea4d3a5149d9e07c77475a33f53337d8b1  output_summary.tsv
```

The full execution transcript is in `reproduce.log`.
