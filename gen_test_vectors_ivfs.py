#!/usr/bin/env python3
"""Generate folded golden vectors for the cascade model tinyx3_ivfs.

Each case i (000..049):
  input_%03d.npy   float32 [1, 1, 3, 1334]   (folded from a 4000-sample chunk)
  output_%03d.npy  float32 [1, 1, 1334, 5]   (softmax probabilities)
The chunks are the same 50 real chunks (chunksize=4000, overlap=400) used by
test_vectors/TINYX3, folded as M[r, c] = [0, x, 0][3*c + r].
"""

import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from fast5 import get_raw_data, read_chunks  # noqa: E402
from inference import load_model  # noqa: E402

MODEL = "tinyx3_ivfs"
CHUNKSIZE = 4000
OVERLAP = 400
N_CASES = 50
OUTDIR = os.path.join(HERE, "test_vectors", MODEL + "_folded")


def fold(x):
    """[1,1,1,4000] -> [1,1,3,1334] using M[r,c] = [0,x,0][3c+r]."""
    assert x.shape[-1] == CHUNKSIZE, x.shape
    y = F.pad(x, (1, 1))  # [1,1,1,4002]
    return y.reshape(1, 1, (CHUNKSIZE + 2) // 3, 3).transpose(2, 3).contiguous()


def main():
    model = load_model(MODEL, device="cpu")
    model.eval()

    reads = []
    for f in sorted(glob.glob(os.path.join(HERE, "sample_data/fast5/*.fast5"))):
        reads.extend(get_raw_data(Path(f)))
    chunks_of = [(r.read_id, list(read_chunks(r, CHUNKSIZE, OVERLAP))) for r in reads]

    cases = []
    round_idx = 0
    while len(cases) < N_CASES:
        added = False
        for read_id, chunks in chunks_of:
            if round_idx < len(chunks):
                cases.append((read_id, round_idx + 1, chunks[round_idx].signal))
                added = True
                if len(cases) >= N_CASES:
                    break
        if not added:
            break
        round_idx += 1

    os.makedirs(OUTDIR, exist_ok=True)
    for p in glob.glob(os.path.join(OUTDIR, "*.npy")):
        os.remove(p)

    lines = ["idx\tread_id\tchunk"]
    xf = y = None
    for i, (read_id, chunk_no, sig) in enumerate(cases):
        x = torch.from_numpy(sig.astype(np.float32)).reshape(1, 1, 1, CHUNKSIZE)
        xf = fold(x)
        with torch.no_grad():
            y = model(xf)
        np.save(os.path.join(OUTDIR, "input_%03d.npy" % i), xf.numpy().astype("<f4"))
        np.save(os.path.join(OUTDIR, "output_%03d.npy" % i), y.numpy().astype("<f4"))
        lines.append("%03d\t%s\t%d" % (i, read_id, chunk_no))

    with open(os.path.join(OUTDIR, "cases.tsv"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    # round-trip check on the first case
    xi = torch.from_numpy(np.load(os.path.join(OUTDIR, "input_000.npy")))
    yi = torch.from_numpy(np.load(os.path.join(OUTDIR, "output_000.npy")))
    with torch.no_grad():
        yc = model(xi)
    md = float(torch.max(torch.abs(yc - yi)))

    print("cases:", len(cases))
    print("input shape:", tuple(xf.shape), "output shape:", tuple(y.shape))
    print("roundtrip max|diff|:", md)
    print("saved to", OUTDIR)


if __name__ == "__main__":
    main()
