"""Run the TargetCall filtering pipeline."""

import argparse
import subprocess
import sys
from pathlib import Path


MODEL_SETTINGS = {
    "default": (512, "default"),
    "TINYX0111": (1600, "tinynoskipx0111"),
    "TINYX011": (3200, "tinynoskipx011"),
    "TINYX01": (6400, "tinynoskipx01"),
    "TINYX2": (12800, "tinynoskipx2"),
    "TINYX3": (25600, "tinynoskipx3"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reads_directory")
    parser.add_argument("reference")
    parser.add_argument("model", choices=MODEL_SETTINGS)
    parser.add_argument("output_directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batchsize", type=int)
    parser.add_argument("--max-reads", type=int, default=0)
    return parser.parse_args()


def run(args):
    repository = Path(__file__).resolve().parent.parent
    script_directory = repository / "src"
    model_directory = repository / "bonito" / "models" / args.model
    output_directory = Path(args.output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    default_batchsize, modeltype = MODEL_SETTINGS[args.model]
    batchsize = args.batchsize or default_batchsize
    fastq_path = output_directory / "output.fastq"
    fasta_path = output_directory / "output.fasta"
    sam_path = output_directory / "output.sam"
    readids_path = output_directory / "readids.txt"

    basecaller_command = [
        sys.executable,
        "-m",
        "bonito",
        "basecaller",
        str(model_directory),
        str(Path(args.reads_directory).resolve()),
        "--batchsize",
        str(batchsize),
        "--modeltype",
        modeltype,
        "--device",
        args.device,
    ]
    if args.max_reads:
        basecaller_command.extend(["--max-reads", str(args.max_reads)])

    print("\n" + " ".join(basecaller_command))
    with fastq_path.open("w") as fastq_file:
        subprocess.run(
            basecaller_command,
            cwd=repository,
            stdout=fastq_file,
            check=True,
        )

    fasta_command = [
        sys.executable,
        str(script_directory / "fastq_to_fasta.py"),
        str(fastq_path),
        str(fasta_path),
    ]
    alignment_command = [
        "minimap2",
        "-a",
        "-x",
        "map-ont",
        "-t",
        "16",
        str(Path(args.reference).resolve()),
        str(fasta_path),
    ]
    filtering_command = [
        sys.executable,
        str(script_directory / "extract_filtered.py"),
        str(sam_path),
        str(readids_path),
    ]

    print("\n" + " ".join(fasta_command))
    subprocess.run(fasta_command, check=True)

    print("\n" + " ".join(alignment_command))
    with sam_path.open("w") as sam_file:
        subprocess.run(alignment_command, stdout=sam_file, check=True)

    print("\n" + " ".join(filtering_command))
    subprocess.run(filtering_command, check=True)


if __name__ == "__main__":
    run(parse_args())
