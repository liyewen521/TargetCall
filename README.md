# TargetCall

TargetCall is the first pre-basecalling filter that is applicable to a wide range of use cases. TargetCall’s key idea is to quickly filter out off-target reads (i.e., reads that are dissimilar to the target reference.) before the basecalling step to eliminate the wasted computation in basecalling. TargetCall is based on ONT basecaller Bonito.

## Prerequisites

TargetCall requires minimap2 to be installed. Minimap2 can be installed via [Minimap2 (v2.24)](https://github.com/lh3/minimap2/releases/tag/v2.24)

## Installation
TargetCall is tested on Linux with conda version 4.7.12.

```bash
$ git clone https://github.com/CMU-SAFARI/TargetCall
$ cd TargetCall
$ conda create --name targetcall python=3.8.10
$ conda activate targetcall
(targetcall) $ pip install --upgrade pip
(targetcall) $ pip install -r requirements.txt
(targetcall) $ python setup.py develop
```

You may need to use requirements-cuda111.txt or requirements-cuda113.txt depending on your cuda version.

## Usage

```bash
$ cd src
$ python targetcall.py ../sample_data/fast5/ ../sample_data/Monkeypox_virus.fasta TINYX011 ../sample_data/
```
This will create three output files under ../sample_data/
- output.fasta: contains noisy basecalled reads of fast5 files using model TINYX011
- output.sam: contains alignment of noisy reads to Monkeypox_virus reference.
- readids.txt: the read IDs of reads that are accepted by the filter.

Read IDs can be used as an input to Bonito for basecalling only the reads that are accepted by the filter using the --read-ids option.

## Provided Models

You can find all models listed under bonito/models/.

| Model Name  | Model Name in the Paper | # of Parameters  | Basecalling Accuracy |
| ------------- | ------------- | ------------- | ------------- |
| default  | Bonito  | 9739K  | 94.60%  |
| TINYX0111  | LC-Main*2  | 565K  | 90.91%  |
| TINYX011  | LC-Main  | 292K  | 89.75%  |
| TINYX01  | LC-Main/2  | 146K  | 86.83%  |
| TINYX2  | LC-Main/4 | 52K  | 80.82%  |
| TINYX3  | LC-Main/8  | 21K  | 70.42%  |

## Standalone inference model

[`model.py`](./model.py) is a self-contained PyTorch implementation of all six
TargetCall models. It only depends on PyTorch: it does not import Bonito, does
not read `config.toml`, and does not use the original
`Encoder`/`Block`/`TCSConv1d`/`Decoder` wrappers. Every block is built from
primitive `Conv1d` and `BatchNorm1d` layers, which keeps the graph simple for
inference, ONNX export, or quantization.

The TargetCall application uses these classes directly, so its model loading and
basecalling path no longer reads model architecture from `config.toml`.

### Requirements

Only `torch` is required to build a model and run a forward pass. Decoding a
sequence additionally requires `fast_ctc_decode` (installed by
`requirements.txt`).

### Quick start

```python
import torch

from model import load_model

model = load_model(
    "TINYX011",
    "bonito/models/TINYX011/weights_1.tar",
    device="cpu",
)

signal = torch.randn(1, 1, 4000)
with torch.no_grad():
    log_probabilities = model(signal)
```

- Input layout: `[batch, 1, samples]`.
- Output layout: `[time, batch, 5]` log probabilities, where
  `time = (samples - 1) // stride + 1` and `stride` is `3`.
- Supported names: `default`, `TINYX0111`, `TINYX011`, `TINYX01`, `TINYX2`,
  `TINYX3` (the same names listed in the table above).

### Building a model without weights

```python
from model import create_model

model = create_model("TINYX3")
```

`create_model` is the standalone equivalent of the old
`bonito.ctc.create_model` and raises `ValueError` for unknown names.

### Loading weights

`load_weights` accepts either the original TargetCall checkpoints (whose nested
legacy key names are remapped automatically) or an already-flattened
`state_dict` whose keys match the model:

```python
from model import create_model, load_weights

model = create_model("TINYX011")
load_weights(model, "bonito/models/TINYX011/weights_1.tar")
```

Checkpoints wrapped in a `state_dict` or `model_state_dict` mapping, and keys
prefixed with `module.`, are handled automatically. Pass `strict=False` to
allow unexpected extra keys.

### Model attributes

Each model exposes the metadata needed for basecalling:

- `model.alphabet` — output labels, i.e. `["N", "A", "C", "G", "T"]`.
- `model.stride` — number of input samples per output time step (`3`).
- `model.features` — number of channels feeding the decoder.
- `model.qscale` / `model.qbias` — qscore scaling used during decoding
  (`1.0` and `0.0` for the provided models).

### Decoding

`model.py` returns log probabilities. Turn them into a sequence with
`fast_ctc_decode` directly, or with the same helper the basecaller uses:

```python
from bonito.ctc.basecall import ctc_decode

scores = log_probabilities[:, 0, :]  # [time, 5] log probabilities
sequence = ctc_decode(
    scores,
    model.alphabet,
    model.qscale,
    model.qbias,
    beamsize=5,
)
```

Use `beamsize=1, qscores=True, return_path=True` for Viterbi decoding with a
quality string and move path.

### Using it through the basecaller CLI

The application path (`bonito basecaller`) loads the same classes:

```bash
python -m bonito basecaller bonito/models/TINYX011 reads/ \
    --modeltype tinynoskipx011 --device cuda:0 --batchsize 3200
```

`--modeltype` maps to the standalone classes as `default`, `tinynoskipx0111`,
`tinynoskipx011`, `tinynoskipx01`, `tinynoskipx2`, and `tinynoskipx3`.

### Tests

```bash
PYTHONPATH=. python test/test_static_models.py
```

The tests verify that the application loads a model without calling
`toml.load`, and that each standalone model matches the original config-based
model bit-for-bit.

## Reproducing the results in the paper

We explain how to reproduce the results we show in the TargetCall paper in the [test directory](./test/).

# <a name="cite"></a>Citing TargetCall

 TargetCall is described and evaluated in the following paper. If you find the repository and the code useful, please cite:

> Meryem Banu Cavlak, Gagandeep Singh, Mohammed Alser, Can Firtina, Joël Lindegger, 
> Mohammad Sadrosadati, Nika Mansouri Ghiasi, Can Alkan, and Onur Mutlu,
> ["TargetCall: Eliminating the Wasted Computation in Basecalling via Pre-Basecalling Filtering,"](https://arxiv.org/abs/2212.04953)
> *arXiv* (2022). [DOI](https://doi.org/10.48550/arXiv.2212.04953)


BIB:

```bibtex
@article{cavlak_targetcall_2022,
  title = {{TargetCall: Eliminating the Wasted Computation in Basecalling via Pre-Basecalling Filtering}},
  url = {https://doi.org/10.48550/arXiv.2212.04953},
  journal = {arXiv},
  author = {Cavlak, Meryem Banu and Singh, Gagandeep and Alser, Mohammed and Firtina, Can and Lindegger, Joël and Sadrosadati, Mohammad and Ghiasi, Nika Mansouri and Alkan, Can and Mutlu, Onur},
  year = {2022},
  month = dec,
}
```
