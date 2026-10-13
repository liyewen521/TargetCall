# minimap2 for ARM (aarch64)

Cross-compiled **ARM 64-bit (aarch64)** static executable of minimap2, for
running TargetCall's alignment step on an ARM host CPU.

- File: `minimap2`
- Version: `2.24-r1122`
- Architecture: `aarch64` (ARMv8-A, NEON)
- Linking: fully **static** (no dependency on the host's glibc or zlib)
- sha256: `76efc7d0d892969f0858b4a97e5c4b14bac20918ca6bd33ed630d1f11649096a`
- Size: ~0.9 MB

## Quick start (on the ARM host)

```bash
chmod +x minimap2
./minimap2 --version          # -> 2.24-r1122
```

## Inputs and outputs

| | Description |
| --- | --- |
| Input 1 | reference sequence, FASTA (optionally gzip-compressed) |
| Input 2 | reads to align, FASTA or FASTQ (optionally gzip-compressed) |
| Output | PAF by default; SAM with `-a`; SAM/PAF written to stdout unless `-o` is given |

## Example (included)

A tiny self-contained example is bundled in this folder so you can run it
immediately:

| File | Role |
| --- | --- |
| `reference.fasta` | example reference (4000 bp) |
| `reads.fasta` | 3 example reads (substrings of the reference) |
| `example.sam` | SAM produced by `minimap2 -a -x map-ont reference.fasta reads.fasta` |

Reproduce / run:

```bash
./run_example.sh
# -> writes output.sam

# or explicitly
./minimap2 -a -x map-ont reference.fasta reads.fasta > output.sam
```

On a non-ARM machine, run the binary through QEMU:

```bash
qemu-aarch64-static ./minimap2 -a -x map-ont reference.fasta reads.fasta > output.sam
```

Expected alignments (from `example.sam`):

```
read_1  0  example_reference  101   60  1500M
read_2  0  example_reference  1801  60  1600M
read_3  0  example_reference  2501  60  1500M
```

## Usage

Align ONT reads to a reference and write SAM:

```bash
./minimap2 -a -x map-ont reference.fasta reads.fasta > output.sam
```

This is exactly what TargetCall runs (`reference` first, then the reads):

```bash
./minimap2 -a -x map-ont <reference.fasta> <output.fasta> > output.sam
```

Other useful pieces:

- Presets: `-x map-ont` (ONT), `-x map-pb` (PacBio CLR), `-x map-hifi`,
  `-x sr` (short reads), `-x asm5|asm10|asm20` (assembly).
- Threads: `-t 16`.
- Secondary alignments: `-N 5`.
- Mini-batch size: `-K 1G`.
- Full manual: `./minimap2 --help`.

## Using it in TargetCall

`targetcall.py` prefers the bundled `tools/minimap2` (x86-64). On an ARM host
use this binary instead — put it on your `PATH`, or replace `tools/minimap2`
with it, or run it directly as shown above.

There is a small wrapper for convenience:

```bash
./run_example.sh <reference.fasta> <reads.fasta> [output.sam]
```

## Verified

- `qemu-aarch64-static ./minimap2 --version` -> `2.24-r1122`
- A real alignment against `sample_data/` produced a valid SAM under emulation.

## How it was built

See `build.sh`. It builds a static aarch64 zlib, then minimap2 with the ARM NEON
kernels using an `aarch64-linux-gnu` cross toolchain:

```bash
CC=aarch64-linux-gnu-gcc AR=aarch64-linux-gnu-ar \
  make arm_neon=1 aarch64=1 \
  CPPFLAGS="-DHAVE_KALLOC -I<zlib-static>/include" \
  LIBS="-lm <zlib-static>/lib/libz.a -lpthread" \
  CFLAGS="-O2 -static -Wc++-compat -D_FILE_OFFSET_BITS=64 -fsigned-char"

aarch64-linux-gnu-strip minimap2
```
