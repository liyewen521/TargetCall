# tinyx3_ivfs RHB-Lite INT8 vectors

The 50 cases pair quantized inputs from `../tinyx3_ivfs_folded` with
CModel-aligned output codes. The input quantization uses the divisor in
`onnx_models/tinyx3_ivfs_q8.onnx`. Calibration uses cases 000–039.

| Tensor | Shape | Dtype | Scale divisor |
|---|---|---|---|
| Input | `[1, 1, 3, 1334]` | int8 | 0.0186600964516 |
| CModel output | `[1, 1, 1334, 5]` | int8 | 0.0078125 (1/128) |

Inputs use nearest-even rounding and symmetric saturation to [-127, 127].
The p99.5 input range clips 857 of 200100 samples. CModel uses the effective
output divisor encoded by the Softmax instruction (`sm_shift0=7`).

- `input_XXX.npy` is the quantized input for seed XXX.
- `output_XXX.npy` is the replayed INT8 output verified against CModel's
  `net_csim_output.npz["out_84"]` for the same seed. All 50 cases matched
  element by element with the compiled model recorded in `provenance.json`.
- `scales.tsv` records the divisors for the files in this directory.
- `layer_scales.tsv` records the source ONNX annotations.
- `act_maxabs.json` records float tensor maxabs values over cases 000–039.
- `layer_ref/` contains the TargetCall software pipeline reference for case
  000. It uses float32 requantization and float Softmax.

The demo's default `make cmodel seed=0 model=ctc_test.tinyx3` reads this
directory. Compile the multiplier-scale export of `tinyx3_ivfs_q8.onnx` and
use the instruction and model hashes in `provenance.json` to identify the
compiled version used for these output vectors.

Run `make_test_vectors.py --dst /tmp/tinyx3_float_boundary_int8` to regenerate
diagnostic float-boundary vectors. The float golden arrays are in
`../tinyx3_ivfs_folded`. See [QUANT_V2.md](../../experiments/cascade_train/QUANT_V2.md)
for the CModel replay and software evaluation commands.
