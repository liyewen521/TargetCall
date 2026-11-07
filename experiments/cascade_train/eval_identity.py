#!/usr/bin/env python3
"""Greedy-CTC identity of the trained cascade model vs the original TINYX3."""

import argparse
import os
import sys

import numpy as np
import parasail
import torch


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

from model_cascade import Model  # noqa: E402
from train_cascade import Chunks, fold_input  # noqa: E402
from inference import load_model  # noqa: E402

LABELS = ["N", "A", "C", "G", "T"]


def greedy(probs):
    ids = probs.argmax(axis=-1)
    out = []
    prev = -1
    for i in ids:
        if i != prev and i != 0:
            out.append(LABELS[i])
        prev = i
    return "".join(out)


def identity(ref, seq):
    if not ref or not seq:
        return 0.0
    aln = parasail.nw_trace_striped_32(seq, ref, 8, 4, parasail.dnafull)
    r = aln.traceback.ref
    q = aln.traceback.query
    m = sum(a == b for a, b in zip(r, q))
    return 100.0 * m / max(len(r), 1)


def run(model, data, n, batch, device):
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, n, batch):
            idx = np.arange(start, min(start + batch, n))
            xb, _, _ = data.batch(idx)
            xb, _ = fold_input(xb)
            probs = model(xb.to(device)).squeeze(1).cpu().numpy()  # [B,T,5]
            for j in range(probs.shape[0]):
                scores.append(probs[j])
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/example_data_dna_r9.4.1_v0/validation")
    ap.add_argument("--ckpt", default=os.path.join(REPO, "checkpoints", "tinyx3_ivfs.pt"))
    ap.add_argument("--n", type=int, default=978)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    data = Chunks(args.data)
    n = min(args.n, data.n)
    refs = [ "".join(LABELS[e] for e in np.asarray(data.refs[i]) if e) for i in range(n) ]

    teacher = load_model("TINYX3", device=args.device)
    student = Model().to(args.device)
    checkpoint = torch.load(args.ckpt, map_location=args.device)
    student.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)

    for name, model in [("teacher", teacher), ("student", student)]:
        probs = run(model, data, n, args.batch, args.device)
        idents = [identity(refs[i], greedy(probs[i])) for i in range(n)]
        print("[%s] n=%d  mean identity %.2f%%  median %.2f%%"
              % (name, n, float(np.mean(idents)), float(np.median(idents))))
        print("  sample:", greedy(probs[0])[:60])


if __name__ == "__main__":
    main()
