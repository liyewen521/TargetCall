# tinyx3_ivfs int8 test vectors

Same 50 cases as `../tinyx3_ivfs_folded`, quantised symmetrically to int8 with
the scales embedded in `onnx_models/tinyx3_ivfs.onnx`:

| tensor | shape | dtype | scale | multiplier (1/scale) |
|--------|-------|-------|-------|----------------------|
| input  | `[1, 1, 3, 1334]` | int8 | 0.038132638 | 26.22425 |
| output | `[1, 1, 1334, 5]` | int8 | 0.007812353 | 128.00240 |

Dequantise with `x = q * scale`, quantise with `q = clip(round(x / scale), -128, 127)`.
None of the 50 cases saturates.

Files:

- `input_XXX.npy`, `output_XXX.npy` — int8 arrays (numpy `.npy`, little endian)
- `scales.tsv` — the two boundary scales above
- `layer_scales.tsv` — every annotated node of the ONNX with its input/output scale
- `act_maxabs.json` — calibration data: `max|activation|` per tensor over these
  50 cases, i.e. the input to `attach_quant.py --act-maxabs`

Accuracy against the float model:

| measurement | value |
|---|---|
| output quantisation alone | `max abs dP = scale/2 = 0.0039`, median 0.0019 |
| int8 input -> model -> float output | median `abs dP = 0.00089`, mean 0.0043, p99 0.049 |
| argmax agreement with the float model | 98.74% mean, 97.23% worst case |

Only the attributes changed: the annotated ONNX produces bit-identical float
outputs to the unannotated export, so any mismatch against these vectors comes
from the int8 input quantisation, not from the annotation.
