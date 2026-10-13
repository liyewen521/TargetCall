#!/usr/bin/env bash
#
# Run the ARM minimap2 on a reference and a set of reads, writing SAM.
#
#   ./run_example.sh [reference.fasta] [reads.fasta] [output.sam]
#
# With no arguments it uses the bundled example files (reference.fasta,
# reads.fasta) and writes output.sam.
#
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
MINIMAP2="$HERE/minimap2"

REFERENCE="${1:-$HERE/reference.fasta}"
READS="${2:-$HERE/reads.fasta}"
OUTPUT="${3:-$HERE/output.sam}"

"$MINIMAP2" -a -x map-ont -t 16 "$REFERENCE" "$READS" > "$OUTPUT"
echo "wrote $OUTPUT"
