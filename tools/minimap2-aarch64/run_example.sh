#!/usr/bin/env bash
#
# Run the ARM minimap2 on a reference and a set of reads, writing SAM.
#
#   ./run_example.sh <reference.fasta> <reads.fasta|reads.fastq> [output.sam]
#
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
MINIMAP2="$HERE/minimap2"

if [ "$#" -lt 2 ]; then
    echo "usage: $0 <reference.fasta> <reads.fasta|reads.fastq> [output.sam]" >&2
    exit 2
fi

REFERENCE="$1"
READS="$2"
OUTPUT="${3:-output.sam}"

"$MINIMAP2" -a -x map-ont -t 16 "$REFERENCE" "$READS" > "$OUTPUT"
echo "wrote $OUTPUT"
