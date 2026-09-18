"""
Bonito Basecaller
"""

import os
import sys
import numpy as np
import torch
from tqdm import tqdm
from time import perf_counter
from functools import partial
from contextlib import nullcontext
from datetime import timedelta
from itertools import islice as take
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from fast_ctc_decode import beam_search, viterbi_search

from aligner import align_map, Aligner
from bio import CTCWriter, Writer, biofmt
from fast5 import get_reads, get_read_groups, read_chunks
from parallel import process_cancel, process_map
from util import __models__, column_to_set, load_model, init
from util import mean_qscore_from_qstring, chunk, stitch, batchify, unbatchify, permute


def basecall(model, reads, beamsize=5, chunksize=0, overlap=0, batchsize=1, qscores=False, reverse=None):
    """
    Basecalls a set of reads.
    """
    chunks = (
        (read, chunk(torch.tensor(read.signal), chunksize, overlap)) for read in reads
    )
    scores = unbatchify(
        (k, compute_scores(model, v)) for k, v in batchify(chunks, batchsize)
    )
    scores = (
        (read, {'scores': stitch(v, chunksize, overlap, len(read.signal), model.stride)}) for read, v in scores
    )
    decode_scores = partial(
        ctc_decode,
        alphabet=model.alphabet,
        qscale=model.qscale,
        qbias=model.qbias,
    )
    decoder = partial(
        decode,
        decode=decode_scores,
        beamsize=beamsize,
        qscores=qscores,
        stride=model.stride,
    )
    basecalls = process_map(decoder, scores, n_proc=4)
    return basecalls


def compute_scores(model, batch):
    """
    Compute scores for model.
    """
    with torch.no_grad():
        device = next(model.parameters()).device
        chunks = batch.to(device)
        if device.type == 'cuda':
            chunks = chunks.to(torch.half)
            autocast = torch.cuda.amp.autocast()
        else:
            chunks = chunks.to(torch.float32)
            autocast = nullcontext()
        with autocast:
            probs = permute(model(chunks), 'TNC', 'NTC')

    return probs.cpu().to(torch.float32)


def ctc_decode(x, alphabet, qscale=1.0, qbias=0.0, beamsize=5,
               threshold=1e-3, qscores=False, return_path=False):
    """Decode CTC scores without requiring a decode method on the model."""
    x = x.exp().cpu().numpy().astype(np.float32)
    if beamsize == 1 or qscores:
        seq, path = viterbi_search(x, alphabet, qscores, qscale, qbias)
    else:
        seq, path = beam_search(x, alphabet, beamsize, threshold)
    if return_path:
        return seq, path
    return seq


def decode(scores, decode, beamsize=5, qscores=False, stride=1):
    """
    Convert the network scores into a sequence.
    """
    # do a greedy decode to get a sensible qstring to compute the mean qscore from
    seq, path = decode(scores['scores'], beamsize=1, qscores=True, return_path=True)
    seq, qstring = seq[:len(path)], seq[len(path):]
    mean_qscore = mean_qscore_from_qstring(qstring)

    # beam search will produce a better sequence but doesn't produce a sensible qstring/path
    if not (qscores or beamsize == 1):
        try:
            seq = decode(scores['scores'], beamsize=beamsize)
            path = None
            qstring = '*'
        except:
            pass
    sig_move = None
    if path is not None:
        sig_move = np.full(path.size * stride, False)
        sig_move[np.where(path)[0] * stride] = True
    return {'sequence': seq, 'qstring': qstring, 'mean_qscore': mean_qscore, 'path': path, 'sig_move': sig_move}


def main(args):

    init(args.seed, args.device)

    sys.stderr.write(f"> loading model {args.model_directory}\n")
    try:
        model = load_model(
            args.model_directory,
            args.device,
            modeltype=args.modeltype,
            weights=int(args.weights),
            chunksize=args.chunksize,
            overlap=args.overlap,
            batchsize=args.batchsize,
            quantize=args.quantize,
        )
    except FileNotFoundError:
        sys.stderr.write(f"> error: failed to load {args.model_directory}\n")
        sys.stderr.write(f"> available models:\n")
        if os.path.isdir(__models__):
            for model in sorted(os.listdir(__models__)):
                sys.stderr.write(f" - {model}\n")
        exit(1)

    if args.verbose:
        sys.stderr.write(f"> model basecaller params: {model.config['basecaller']}\n")

    if args.reference:
        sys.stderr.write("> loading reference\n")
        aligner = Aligner(args.reference, preset='ont-map', best_n=1)
        if not aligner:
            sys.stderr.write("> failed to load/build index\n")
            exit(1)
    else:
        aligner = None

    fmt = biofmt(aligned=args.reference is not None)

    if args.reference and args.reference.endswith(".mmi") and fmt.name == "cram":
        sys.stderr.write("> error: reference cannot be a .mmi when outputting cram\n")
        exit(1)
    elif args.reference and fmt.name == "fastq":
        sys.stderr.write(f"> warning: did you really want {fmt.aligned} {fmt.name}?\n")
    else:
        sys.stderr.write(f"> outputting {fmt.aligned} {fmt.name}\n")

    if args.save_ctc and not args.reference:
        sys.stderr.write("> a reference is needed to output ctc training data\n")
        exit(1)

    if fmt.name != 'fastq':
        groups = get_read_groups(
            args.reads_directory, args.model_directory,
            n_proc=8, recursive=args.recursive,
            read_ids=column_to_set(args.read_ids), skip=args.skip,
            cancel=process_cancel()
        )
    else:
        groups = []

    reads = get_reads(
        args.reads_directory, n_proc=8, recursive=args.recursive,
        read_ids=column_to_set(args.read_ids), skip=args.skip,
        cancel=process_cancel()
    )

    if args.max_reads:
        reads = take(reads, args.max_reads)

    if args.save_ctc:
        reads = (
            chunk for read in reads
            for chunk in read_chunks(
                read,
                chunksize=model.config["basecaller"]["chunksize"],
                overlap=model.config["basecaller"]["overlap"]
            )
        )
        ResultsWriter = CTCWriter
    else:
        ResultsWriter = Writer

    results = basecall(
        model, reads, reverse=args.revcomp,
        batchsize=model.config["basecaller"]["batchsize"],
        chunksize=model.config["basecaller"]["chunksize"],
        overlap=model.config["basecaller"]["overlap"]
    )

    if aligner:
        results = align_map(aligner, results)

    writer = ResultsWriter(
        fmt.mode, tqdm(results, desc="> calling", unit=" reads", leave=False),
        aligner=aligner, group_key=args.model_directory,
        ref_fn=args.reference, groups=groups,
    )

    t0 = perf_counter()
    writer.start()
    writer.join()
    duration = perf_counter() - t0
    num_samples = sum(num_samples for read_id, num_samples in writer.log)

    sys.stderr.write("> completed reads: %s\n" % len(writer.log))
    sys.stderr.write("> duration: %s\n" % timedelta(seconds=np.round(duration)))
    sys.stderr.write("> samples per second %.1E\n" % (num_samples / duration))
    sys.stderr.write("> done\n")


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("model_directory")
    parser.add_argument("reads_directory")
    parser.add_argument("--reference")
    parser.add_argument("--read-ids")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=25, type=int)
    parser.add_argument("--weights", default="0", type=str)
    parser.add_argument("--skip", action="store_true", default=False)
    parser.add_argument("--save-ctc", action="store_true", default=False)
    parser.add_argument("--revcomp", action="store_true", default=False)
    parser.add_argument("--recursive", action="store_true", default=False)
    quant_parser = parser.add_mutually_exclusive_group(required=False)
    quant_parser.add_argument("--quantize", dest="quantize", action="store_true")
    quant_parser.add_argument("--no-quantize", dest="quantize", action="store_false")
    parser.set_defaults(quantize=None)
    parser.add_argument("--overlap", default=None, type=int)
    parser.add_argument("--chunksize", default=None, type=int)
    parser.add_argument("--batchsize", default=None, type=int)
    parser.add_argument("--max-reads", default=0, type=int)
    parser.add_argument('-v', '--verbose', action='count', default=0)
    parser.add_argument(
        "--modeltype",
        default=None,
        type=str,
        choices=[
            "default",
            "tinynoskipx0111",
            "tinynoskipx011",
            "tinynoskipx01",
            "tinynoskipx2",
            "tinynoskipx3",
        ],
    )
    return parser
