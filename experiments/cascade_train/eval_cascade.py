#!/usr/bin/env python3
"""Compare init vs trained cascade TINYX3 against the original teacher."""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

from model_cascade import Model  # noqa: E402
from train_cascade import Chunks, fold_input, init_cascades, load_partial  # noqa: E402
from inference import load_model  # noqa: E402


def metrics(student, teacher, xb, device):
    student.eval()
    with torch.no_grad():
        p = student(xb.to(device))
        t = teacher(xb.to(device))
    kl = F.kl_div(torch.log(p.clamp_min(1e-8)).reshape(-1, 5), t.reshape(-1, 5),
                  reduction="batchmean").item()
    agree = (p.argmax(-1) == t.argmax(-1)).float().mean().item()
    return kl, agree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/example_data_dna_r9.4.1_v0/validation")
    ap.add_argument("--teacher", default=os.path.join(REPO, "checkpoints", "TINYX3.pt"))
    ap.add_argument("--ckpt", default=os.path.join(REPO, "checkpoints", "tinyx3_ivfs.pt"))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    val = Chunks(args.data)
    idx = np.arange(args.n)
    xb, _, _ = val.batch(idx)
    xb, tf = fold_input(xb)
    print("[eval] val batch", tuple(xb.shape), "T =", tf)

    teacher = load_model("TINYX3", weights=args.teacher, device=args.device)
    teacher.eval()

    state = torch.load(args.teacher, map_location="cpu")
    init = Model().to(args.device)
    init_cascades(init, state, "truncate")
    load_partial(init, state)
    kl0, ag0 = metrics(init, teacher, xb, args.device)
    print("[init    ] KL %.4f  argmax-agree %.4f" % (kl0, ag0))

    trained = Model().to(args.device)
    checkpoint = torch.load(args.ckpt, map_location=args.device)
    trained.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    kl1, ag1 = metrics(trained, teacher, xb, args.device)
    print("[trained ] KL %.4f  argmax-agree %.4f" % (kl1, ag1))


if __name__ == "__main__":
    main()
