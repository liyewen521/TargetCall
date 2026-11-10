# tinyx3_ivfs int8 test vectors

Same 50 cases as `../tinyx3_ivfs_folded`, quantised symmetrically to int8 with
the multipliers embedded in `onnx_models/tinyx3_ivfs.onnx`:

| tensor | shape | dtype | scale (multiplier) | divisor (1/scale) |
|--------|-------|-------|--------------------|-------------------|
| input  | `[1, 1, 3, 1334]` | int8 | 26.224254608 | 0.038132638 |
| output | `[1, 1, 1334, 5]` | int8 | 128.002410889 | 0.007812353 |

Every `scale` attribute in the model is a **multiplier**:

```
q_int8 = clip(round(x * scale), -128, 127)      x = q_int8 / scale
```

which is the same convention as `weight_ch_scales` (`127 / max|w|`). None of the
50 cases saturates: `max|q| = 127` exactly on the calibration maximum.

Files:

- `input_XXX.npy`, `output_XXX.npy` — int8 arrays (numpy `.npy`, little endian)
- `scales.tsv` — the two boundary multipliers above
- `layer_scales.tsv` — every annotated node of the ONNX with its input/output
  multiplier
- `act_maxabs.json` — calibration data: `max|activation|` per tensor over these
  50 cases, i.e. the input to `attach_quant.py --act-maxabs`
- `README.md` — this file

Accuracy against the float model:

| measurement | value |
|---|---|
| output quantisation alone | `max abs dP = 1/(2*128.0) = 0.0039`, median 0.0019 |
| int8 input -> model -> float output | median `abs dP = 0.00089`, mean 0.0043, p99 0.049 |
| argmax agreement with the float model | 98.74% mean, 97.23% worst case |

Only the attributes changed: the annotated ONNX produces bit-identical float
outputs to the unannotated export, so any mismatch against these vectors comes
from the int8 input quantisation, not from the annotation.

## Softmax scales

`exp_scale` on the Softmax node is the ratio `output_scale / input_scale`
(7.398239), so

```
input_scale  = 17.301739      = 127 / max|logit|      (max|logit| = 7.3403)
output_scale = 128.002410889  = 127 / max|prob|       (max|prob|  = 0.992169)
output_scale == input_scale * exp_scale               (relative error ~8e-8)
127 / output_scale = 0.992169 <= 1
```

A real softmax compresses logits into `[0, 1]`, so the two multipliers differ by
~7.4x. Do not assume the placeholder `exp_scale = 2` of the uncalibrated model:
both `output_scale` and `input_scale * exp_scale` now give the same number.
