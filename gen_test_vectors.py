#!/usr/bin/env python3
"""Generate a golden input/output pair for hardware verification.

Model : TINYX3
Input : float32 [1, 1, 1, 4000]  (NCHW, singleton height)
        = one pipeline chunk (chunksize=4000, overlap=400) of a real ont-fast5
          read's med/mad-normalised signal, i.e. the exact tensor fed to the NN.
Output: float32 [1, 1, 1334, 5]  (NDWC softmax probabilities)

Saved as little-endian float32 .npy under test_vectors/TINYX3/.
"""
import os
import sys
import glob
from pathlib import Path
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from fast5 import get_raw_data
from convert_legacy import flatten_legacy_state_dict
from inference import create_model

MODEL = "TINYX3"
CHUNKSIZE = 4000
OVERLAP = 400
OUTDIR = os.path.join(HERE, "test_vectors", MODEL)

fast5 = sorted(glob.glob(os.path.join(HERE, "sample_data/fast5/GXB*.fast5")))[0]
read = next(get_raw_data(Path(fast5)))

signal = read.signal
offset = (len(signal) - CHUNKSIZE) % (CHUNKSIZE - OVERLAP)
chunk = signal[offset:offset + CHUNKSIZE].astype(np.float32)
assert chunk.shape == (CHUNKSIZE,), chunk.shape
x = torch.from_numpy(chunk).reshape(1, 1, 1, CHUNKSIZE)

model = create_model(MODEL)
raw = torch.load(os.path.join(HERE, "checkpoints", MODEL, "weights_1.tar"), map_location="cpu")
model.load_state_dict(flatten_legacy_state_dict(MODEL, raw), strict=True)
model.eval()
with torch.no_grad():
    y = model(x)

os.makedirs(OUTDIR, exist_ok=True)
np.save(os.path.join(OUTDIR, "input.npy"), x.numpy().astype("<f4"))
np.save(os.path.join(OUTDIR, "output.npy"), y.numpy().astype("<f4"))

print("fast5   :", os.path.basename(fast5))
print("read_id :", read.read_id)
print("input   :", tuple(x.shape), x.dtype, "range [%.5f, %.5f]" % (x.min(), x.max()))
print("output  :", tuple(y.shape), y.dtype, "range [%.5f, %.5f]" % (y.min(), y.max()))
print("out sum/pos = %.6f" % y.sum(-1).mean())
print("saved   :", OUTDIR)
