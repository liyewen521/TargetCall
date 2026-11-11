# tinyx3_ivfs int8 test vectors

Same 50 cases as `../tinyx3_ivfs_folded`, quantised symmetrically to int8 with
the multipliers embedded in `onnx_models/tinyx3_ivfs_q8.onnx`:

| tensor | shape | dtype | scale (multiplier) | divisor (1/scale) |
|--------|-------|-------|--------------------|-------------------|
| input  | `[1, 1, 3, 1334]` | int8 | 41.968292236 | 0.023827512 |
| output | `[1, 1, 1334, 5]` | int8 | 128.002410889 | 0.007812353 |

Every `scale` attribute in the model is a **multiplier**:

```
q_int8 = clip(round(x * scale), -128, 127)      x = q_int8 / scale
```

which is the same convention as `weight_ch_scales` (`127 / max|w|`).

The input multiplier is a **p99.9** clip, not `127 / max|x|`: over the 50 cases
`198 / 200100` samples (0.099%) saturate at 127 by design. The output multiplier
is still `127 / max|prob|`, so `output_XXX.npy` is unchanged from the previous
revision of this directory; only the inputs moved.

Files:

- `input_XXX.npy`, `output_XXX.npy` — int8 arrays (numpy `.npy`, little endian)
- `scales.tsv` — the two boundary multipliers above
- `layer_scales.tsv` — every annotated node of the ONNX with its input/output
  bit depth and multiplier; per-channel layers carry the full list in one cell
- `act_maxabs.json` — calibration data: `max|activation|` per tensor over these
  50 cases. Hidden layers are annotated with a p99.9 clip instead, so this file
  is documentation, not the numbers used by the model
- `README.md` — this file

Regenerate with `experiments/cascade_train/make_test_vectors.py` (see
`experiments/cascade_train/QUANT_V2.md` for the quantisation scheme).

Accuracy against the float model:

| measurement | value |
|---|---|
| output quantisation alone | `max abs dP = 1/(2*128.0) = 0.0039`, median 0.0019 |
| int8 input -> model -> float output | median `abs dP = 0.00060`, mean 0.00299, p99 0.034 |
| argmax agreement with the float model | 99.12% mean, 96.85% worst case |

Only the attributes changed: the annotated ONNX produces bit-identical float
outputs to the unannotated export, so any mismatch against these vectors comes
from the int8 boundary quantisation, not from the annotation.

The full int8 model (`tinyx3_ivfs_q8.onnx`, every layer quantised) is a different
number: 90.43% frame agreement and 79.80% decoded identity against the float
model on vectors 040-049.

## Softmax scales

`exp_scale` on the Softmax node is the ratio `output_scale / input_scale`
(5.418577), so

```
input_scale  = 23.622881       = 127 / max|logit|      (max|logit| = 5.3762)
output_scale = 128.002410889   = 127 / max|prob|       (max|prob|  = 0.992169)
output_scale == input_scale * exp_scale                (relative error ~1e-7)
127 / output_scale = 0.992169 <= 1
```

A real softmax compresses logits into `[0, 1]`, so the two multipliers differ by
~5.4x. Do not assume the placeholder `exp_scale = 2` of the uncalibrated model:
both `output_scale` and `input_scale * exp_scale` now give the same number.

The decoder conv, the reshape/transpose chain and the softmax use **scalar**
multipliers on purpose: a per-class scale on the logits would change the argmax
itself.

