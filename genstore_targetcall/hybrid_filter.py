#!/usr/bin/env python3
"""Filter TargetCall basecalls with a GenStore-style seed hash index.

The original GenStore software counts matching hashes globally.  This adapter
keeps the same canonical, truncated-hash idea but retains reference positions
so matches can be aggregated per read and checked for approximate collinearity.
Reads with an ambiguous score can optionally be checked by minimap2.
"""

import argparse
import csv
import gzip
import hashlib
import os
import pickle
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


INDEX_VERSION = 1
DNA = frozenset("ACGT")
COMPLEMENT = str.maketrans("ACGT", "TGCA")
MODEL_CONFIG = {
    "default": (512, "default"),
    "TINYX1": (1600, "tinynoskipx1"),
    "TINYX0111": (1600, "tinynoskipx0111"),
    "TINYX011": (3200, "tinynoskipx011"),
    "TINYX01": (6400, "tinynoskipx01"),
    "TINYX2": (12800, "tinynoskipx2"),
    "TINYX3": (25600, "tinynoskipx3"),
    "TINYX4": (51200, "tinynoskipx4"),
}


@dataclass
class SequenceRecord:
    name: str
    sequence: str


@dataclass
class ReadDecision:
    read_id: str
    length: int
    seeds: int
    matched_seeds: int
    coherent_hits: int
    contig: str
    strand: str
    anchor: Optional[int]
    initial_decision: str
    final_decision: str


def reverse_complement(sequence: str) -> str:
    return sequence.translate(COMPLEMENT)[::-1]


def _nonempty_lines(path: Path) -> Iterator[str]:
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield line


def read_sequences(path: Path) -> Iterator[SequenceRecord]:
    """Read FASTA, FASTQ, or TargetCall's two-line @-header format."""
    lines = _nonempty_lines(path)
    try:
        line = next(lines)
    except StopIteration:
        return

    if line.startswith(">"):
        name = line[1:].split()[0]
        parts: List[str] = []
        for line in lines:
            if line.startswith(">"):
                yield SequenceRecord(name, "".join(parts).upper())
                name = line[1:].split()[0]
                parts = []
            else:
                parts.append(line)
        yield SequenceRecord(name, "".join(parts).upper())
        return

    if not line.startswith("@"):
        raise ValueError(f"{path}: expected a FASTA/FASTQ header, got {line[:40]!r}")

    pending = line
    while pending:
        if not pending.startswith("@"):
            raise ValueError(f"{path}: expected an @ header, got {pending[:40]!r}")
        name = pending[1:].split()[0]
        try:
            sequence = next(lines).upper()
        except StopIteration as exc:
            raise ValueError(f"{path}: missing sequence for {name}") from exc
        try:
            third = next(lines)
        except StopIteration:
            third = ""
        if third.startswith("+"):
            try:
                next(lines)  # quality
            except StopIteration as exc:
                raise ValueError(f"{path}: missing quality for {name}") from exc
            try:
                pending = next(lines)
            except StopIteration:
                pending = ""
        else:
            pending = third
        yield SequenceRecord(name, sequence)


def _canonical_code(sequence: str) -> Tuple[int, bool]:
    forward = 0
    reverse = 0
    shift = 2 * (len(sequence) - 1)
    encoding = {"A": 0, "C": 1, "G": 2, "T": 3}
    for base in sequence:
        value = encoding[base]
        forward = (forward << 2) | value
        reverse = (reverse >> 2) | ((3 ^ value) << shift)
    return (forward, False) if forward <= reverse else (reverse, True)


def genstore_hash(sequence: str, hash_bits: int) -> Tuple[int, bool]:
    """Reproduce GenStore's canonical MD5-based truncated hash on x86."""
    code, reverse = _canonical_code(sequence)
    byte_count = (code.bit_length() + 7) // 8
    encoded = code.to_bytes(byte_count, "big")
    digest = hashlib.md5(encoded).digest()
    hash64 = int.from_bytes(digest[8:16], "little")
    minimap_hash = ((hash64 << 8) & ((1 << 64) - 1)) | len(sequence)
    return minimap_hash >> (64 - hash_bits), reverse


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_index(reference_path: Path, index_path: Path, seed_length: int, hash_bits: int) -> dict:
    if not 1 <= hash_bits <= 56:
        raise ValueError("hash_bits must be between 1 and 56")
    if not 1 <= seed_length <= 255:
        raise ValueError("seed_length must be between 1 and 255")

    contigs = list(read_sequences(reference_path))
    if not contigs:
        raise ValueError(f"{reference_path}: reference contains no sequences")

    hashes: Dict[int, List[Tuple[int, int, bool]]] = defaultdict(list)
    indexed_kmers = 0
    for contig_id, record in enumerate(contigs):
        sequence = record.sequence
        for position in range(0, len(sequence) - seed_length + 1):
            seed = sequence[position : position + seed_length]
            if not DNA.issuperset(seed):
                continue
            hash_value, reverse = genstore_hash(seed, hash_bits)
            hashes[hash_value].append((contig_id, position, reverse))
            indexed_kmers += 1

    payload = {
        "version": INDEX_VERSION,
        "seed_length": seed_length,
        "hash_bits": hash_bits,
        "reference_path": str(reference_path.resolve()),
        "reference_sha256": _file_sha256(reference_path),
        "contigs": [(record.name, len(record.sequence)) for record in contigs],
        "indexed_kmers": indexed_kmers,
        "hashes": dict(hashes),
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = index_path.with_suffix(index_path.suffix + ".tmp")
    with gzip.open(temporary, "wb", compresslevel=3) as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, index_path)
    return payload


def load_index(index_path: Path) -> dict:
    with gzip.open(index_path, "rb") as handle:
        payload = pickle.load(handle)
    if payload.get("version") != INDEX_VERSION:
        raise ValueError(
            f"unsupported index version {payload.get('version')}; expected {INDEX_VERSION}"
        )
    return payload


def _best_anchor_cluster(
    anchors: Dict[Tuple[int, bool], List[Tuple[int, int]]], tolerance: int
) -> Tuple[int, int, bool, Optional[int]]:
    best = (0, -1, False, None)
    for (contig_id, same_strand), values in anchors.items():
        values.sort()
        left = 0
        query_positions: Counter = Counter()
        for right, (anchor, query_position) in enumerate(values):
            query_positions[query_position] += 1
            while anchor - values[left][0] > tolerance:
                old_query_position = values[left][1]
                query_positions[old_query_position] -= 1
                if query_positions[old_query_position] == 0:
                    del query_positions[old_query_position]
                left += 1
            score = len(query_positions)
            if score > best[0]:
                representative = values[(left + right) // 2][0]
                best = (score, contig_id, same_strand, representative)
    return best


def score_read(
    record: SequenceRecord,
    index: dict,
    seed_stride: int,
    position_tolerance: int,
    max_hash_occurrences: int,
) -> Tuple[int, int, int, int, bool, Optional[int]]:
    seed_length = index["seed_length"]
    hash_bits = index["hash_bits"]
    anchors: Dict[Tuple[int, bool], List[Tuple[int, int]]] = defaultdict(list)
    seeds = 0
    matched_query_positions = set()
    for query_position in range(0, len(record.sequence) - seed_length + 1, seed_stride):
        seed = record.sequence[query_position : query_position + seed_length]
        if not DNA.issuperset(seed):
            continue
        seeds += 1
        hash_value, query_reverse = genstore_hash(seed, hash_bits)
        occurrences = index["hashes"].get(hash_value, ())
        if len(occurrences) > max_hash_occurrences:
            continue
        if occurrences:
            matched_query_positions.add(query_position)
        for contig_id, reference_position, reference_reverse in occurrences:
            same_strand = query_reverse == reference_reverse
            anchor = (
                reference_position - query_position
                if same_strand
                else reference_position + query_position
            )
            anchors[(contig_id, same_strand)].append((anchor, query_position))
    coherent, contig_id, same_strand, anchor = _best_anchor_cluster(
        anchors, position_tolerance
    )
    return seeds, len(matched_query_positions), coherent, contig_id, same_strand, anchor


def _write_fasta(records: Iterable[SequenceRecord], path: Path) -> None:
    with path.open("w") as handle:
        for record in records:
            handle.write(f">{record.name}\n{record.sequence}\n")


def _minimap2_mapped_reads(
    records: Sequence[SequenceRecord],
    reference_path: Path,
    minimap2: str,
    threads: int,
    sam_output: Optional[Path],
) -> set:
    if not records:
        if sam_output:
            sam_output.write_text("")
        return set()
    with tempfile.TemporaryDirectory(prefix="genstore-targetcall-") as directory:
        query_path = Path(directory) / "gray.fasta"
        temporary_sam = Path(directory) / "gray.sam"
        _write_fasta(records, query_path)
        with temporary_sam.open("w") as sam_handle:
            subprocess.run(
                [
                    minimap2,
                    "-a",
                    "-x",
                    "map-ont",
                    "-t",
                    str(threads),
                    str(reference_path),
                    str(query_path),
                ],
                stdout=sam_handle,
                check=True,
            )
        if sam_output:
            sam_output.write_bytes(temporary_sam.read_bytes())
        mapped = set()
        with temporary_sam.open() as sam_handle:
            for line in sam_handle:
                if line.startswith("@"):
                    continue
                fields = line.split("\t", 4)
                if len(fields) >= 3 and fields[2] != "*":
                    mapped.add(fields[0])
        return mapped


def filter_reads(
    index: dict,
    reads_path: Path,
    output_path: Path,
    reference_path: Path,
    seed_stride: int,
    accept_hits: int,
    reject_hits: int,
    position_tolerance: int,
    max_hash_occurrences: int,
    minimap2: str,
    threads: int,
    decisions_path: Optional[Path],
    gray_sam_path: Optional[Path],
) -> Tuple[List[ReadDecision], List[str]]:
    if reject_hits >= accept_hits:
        raise ValueError("reject_hits must be smaller than accept_hits")
    if seed_stride < 1:
        raise ValueError("seed_stride must be positive")
    records = list(read_sequences(reads_path))
    decisions: List[ReadDecision] = []
    gray_records: List[SequenceRecord] = []

    for record in records:
        seeds, matched, coherent, contig_id, same_strand, anchor = score_read(
            record,
            index,
            seed_stride,
            position_tolerance,
            max_hash_occurrences,
        )
        if coherent >= accept_hits:
            initial = final = "accept"
        elif coherent <= reject_hits:
            initial = final = "reject"
        else:
            initial = "gray"
            final = "pending"
            gray_records.append(record)
        contig = index["contigs"][contig_id][0] if contig_id >= 0 else ""
        decisions.append(
            ReadDecision(
                record.name,
                len(record.sequence),
                seeds,
                matched,
                coherent,
                contig,
                "+" if contig and same_strand else "-" if contig else "",
                anchor,
                initial,
                final,
            )
        )

    mapped_gray = _minimap2_mapped_reads(
        gray_records, reference_path, minimap2, threads, gray_sam_path
    )
    for decision in decisions:
        if decision.initial_decision == "gray":
            if decision.read_id in mapped_gray:
                decision.final_decision = "accept"
            else:
                decision.final_decision = "reject"

    accepted = [
        decision.read_id for decision in decisions if decision.final_decision == "accept"
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(f"{read_id}\n" for read_id in accepted))
    if decisions_path:
        decisions_path.parent.mkdir(parents=True, exist_ok=True)
        with decisions_path.open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(ReadDecision.__dataclass_fields__.keys())
            for decision in decisions:
                writer.writerow(vars(decision).values())
    return decisions, accepted


def _targetcall_root(default_root: Optional[Path]) -> Path:
    if default_root:
        return default_root
    return Path(__file__).resolve().parents[1] / "TargetCall"


def run_hybrid_pipeline(
    reads_dir: Path,
    reference_path: Path,
    model_name: str,
    output_dir: Path,
    index_path: Optional[Path],
    targetcall_root: Optional[Path],
    bonito: str,
    keep_fastq: bool,
    rebuild_index: bool,
    seed_length: int,
    hash_bits: int,
    seed_stride: Optional[int],
    accept_hits: int,
    reject_hits: int,
    position_tolerance: int,
    max_hash_occurrences: int,
    minimap2: str,
    threads: int,
) -> Tuple[List[ReadDecision], List[str]]:
    if model_name not in MODEL_CONFIG:
        valid = ", ".join(sorted(MODEL_CONFIG))
        raise ValueError(f"unknown model {model_name!r}; expected one of: {valid}")

    root = _targetcall_root(targetcall_root)
    model_path = root / "bonito" / "models" / model_name
    if not model_path.exists():
        raise ValueError(f"model directory does not exist: {model_path}")
    if not reads_dir.exists():
        raise ValueError(f"reads directory does not exist: {reads_dir}")
    if not reference_path.exists():
        raise ValueError(f"reference does not exist: {reference_path}")

    batch_size, model_type = MODEL_CONFIG[model_name]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_fastq = output_dir / "output.fastq"
    output_fasta = output_dir / "output.fasta"
    output_readids = output_dir / "readids.txt"
    decisions_path = output_dir / "genstore-decisions.tsv"
    gray_sam_path = output_dir / "output.gray.sam"
    resolved_index = index_path or output_dir / f"{reference_path.stem}.k{seed_length}.gsi"

    with output_fastq.open("w") as fastq_handle:
        subprocess.run(
            [
                bonito,
                "basecaller",
                str(model_path),
                str(reads_dir),
                "--batchsize",
                str(batch_size),
                "--modeltype",
                model_type,
            ],
            stdout=fastq_handle,
            check=True,
        )

    _write_fasta(read_sequences(output_fastq), output_fasta)
    if not keep_fastq:
        output_fastq.unlink()

    if rebuild_index or not resolved_index.exists():
        index = build_index(reference_path, resolved_index, seed_length, hash_bits)
    else:
        index = load_index(resolved_index)
        if index["seed_length"] != seed_length or index["hash_bits"] != hash_bits:
            raise ValueError(
                "existing index uses different seed/hash settings; pass --rebuild-index "
                "or choose another --index"
            )
    if _file_sha256(reference_path) != index["reference_sha256"]:
        raise ValueError("reference does not match the indexed reference")

    return filter_reads(
        index,
        output_fasta,
        output_readids,
        reference_path,
        seed_stride or index["seed_length"],
        accept_hits,
        reject_hits,
        position_tolerance,
        max_hash_occurrences,
        minimap2,
        threads,
        decisions_path,
        gray_sam_path,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="build a reference seed hash index")
    index_parser.add_argument("reference", type=Path)
    index_parser.add_argument("index", type=Path)
    index_parser.add_argument("--seed-length", type=int, default=31)
    index_parser.add_argument("--hash-bits", type=int, default=48)

    filter_parser = subparsers.add_parser("filter", help="filter noisy basecalled reads")
    filter_parser.add_argument("index", type=Path)
    filter_parser.add_argument("reads", type=Path)
    filter_parser.add_argument("output", type=Path)
    filter_parser.add_argument("--reference", type=Path)
    filter_parser.add_argument("--seed-stride", type=int)
    filter_parser.add_argument("--accept-hits", type=int, default=3)
    filter_parser.add_argument("--reject-hits", type=int, default=0)
    filter_parser.add_argument("--position-tolerance", type=int, default=64)
    filter_parser.add_argument("--max-hash-occurrences", type=int, default=1000)
    filter_parser.add_argument("--minimap2", default="minimap2")
    filter_parser.add_argument("--threads", type=int, default=16)
    filter_parser.add_argument("--decisions", type=Path)
    filter_parser.add_argument("--gray-sam", type=Path)

    pipeline_parser = subparsers.add_parser(
        "pipeline", help="run TargetCall basecalling followed by the GenStore filter"
    )
    pipeline_parser.add_argument("reads_dir", type=Path)
    pipeline_parser.add_argument("reference", type=Path)
    pipeline_parser.add_argument("model")
    pipeline_parser.add_argument("output_dir", type=Path)
    pipeline_parser.add_argument("--index", type=Path)
    pipeline_parser.add_argument("--targetcall-root", type=Path)
    pipeline_parser.add_argument("--bonito", default="bonito")
    pipeline_parser.add_argument("--keep-fastq", action="store_true")
    pipeline_parser.add_argument("--rebuild-index", action="store_true")
    pipeline_parser.add_argument("--seed-length", type=int, default=31)
    pipeline_parser.add_argument("--hash-bits", type=int, default=48)
    pipeline_parser.add_argument("--seed-stride", type=int)
    pipeline_parser.add_argument("--accept-hits", type=int, default=3)
    pipeline_parser.add_argument("--reject-hits", type=int, default=0)
    pipeline_parser.add_argument("--position-tolerance", type=int, default=64)
    pipeline_parser.add_argument("--max-hash-occurrences", type=int, default=1000)
    pipeline_parser.add_argument("--minimap2", default="minimap2")
    pipeline_parser.add_argument("--threads", type=int, default=16)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "index":
            payload = build_index(args.reference, args.index, args.seed_length, args.hash_bits)
            print(
                f"indexed {payload['indexed_kmers']} k-mers from "
                f"{len(payload['contigs'])} contig(s) into {args.index}"
            )
            return 0

        if args.command == "pipeline":
            decisions, accepted = run_hybrid_pipeline(
                args.reads_dir,
                args.reference,
                args.model,
                args.output_dir,
                args.index,
                args.targetcall_root,
                args.bonito,
                args.keep_fastq,
                args.rebuild_index,
                args.seed_length,
                args.hash_bits,
                args.seed_stride,
                args.accept_hits,
                args.reject_hits,
                args.position_tolerance,
                args.max_hash_occurrences,
                args.minimap2,
                args.threads,
            )
            counts = Counter(decision.initial_decision for decision in decisions)
            print(
                f"reads={len(decisions)} accepted={len(accepted)} "
                f"strong={counts['accept']} gray={counts['gray']} rejected={counts['reject']}"
            )
            return 0

        index = load_index(args.index)
        reference_path = args.reference or Path(index["reference_path"])
        if not reference_path.exists():
            raise ValueError("reference path is missing; pass --reference")
        if _file_sha256(reference_path) != index["reference_sha256"]:
            raise ValueError("reference does not match the indexed reference")
        seed_stride = args.seed_stride or index["seed_length"]
        decisions, accepted = filter_reads(
            index,
            args.reads,
            args.output,
            reference_path,
            seed_stride,
            args.accept_hits,
            args.reject_hits,
            args.position_tolerance,
            args.max_hash_occurrences,
            args.minimap2,
            args.threads,
            args.decisions,
            args.gray_sam,
        )
        counts = Counter(decision.initial_decision for decision in decisions)
        print(
            f"reads={len(decisions)} accepted={len(accepted)} "
            f"strong={counts['accept']} gray={counts['gray']} rejected={counts['reject']}"
        )
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
