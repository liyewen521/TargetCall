#!/usr/bin/env python3
"""Generate golden input/output vectors for hardware verification.

Model : TINYX3
Each case i (000..049):
  input_%03d.npy  float32 [1, 1, 1, 4000]   (NCHW, singleton height)
  output_%03d.npy float32 [1, 1, 1334, 5]   (NDWC softmax probabilities)
Inputs are pipeline chunks (chunksize=4000, overlap=400) of real ont-fast5
reads' med/mad-normalised signal, i.e. the exact tensors fed to the NN.
cases.tsv maps each index to (read_id, chunk number).
"""
import os
import sys
import glob
from pathlib import Path
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from fast5 import get_raw_data, read_chunks
from convert_legacy import flatten_legacy_state_dict
from inference import create_model

MODEL = "TINYX3"
CHUNKSIZE = 4000
OVERLAP = 400
N_CASES = 50
OUTDIR = os.path.join(HERE, "test_vectors", MODEL)

model = create_model(MODEL)
raw = torch.load(os.path.join(HERE, "checkpoints", MODEL, "weights_1.tar"), map_location="cpu")
model.load_state_dict(flatten_legacy_state_dict(MODEL, raw), strict=True)
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
for i, (read_id, chunk_no, sig) in enumerate(cases):
    x = torch.from_numpy(sig.astype(np.float32)).reshape(1, 1, 1, CHUNKSIZE)
    with torch.no_grad():
        y = model(x)
    np.save(os.path.join(OUTDIR, "input_%03d.npy" % i), x.numpy().astype("<f4"))
    np.save(os.path.join(OUTDIR, "output_%03d.npy" % i), y.numpy().astype("<f4"))
    lines.append("%03d\t%s\t%d" % (i, read_id, chunk_no))

with open(os.path.join(OUTDIR, "cases.tsv"), "w") as fh:
    fh.write("\n".join(lines) + "\n")

print("cases:", len(cases))
print("input  shape:", tuple(x.shape), "output shape:", tuple(y.shape))
print("saved to", OUTDIR)
