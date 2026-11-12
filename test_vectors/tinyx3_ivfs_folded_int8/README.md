# tinyx3_ivfs int8 boundary vectors

The same 50 cases as `../tinyx3_ivfs_folded`, quantized using the **divisor**
scales in `onnx_models/tinyx3_ivfs_q8.onnx`. Calibration uses cases 000–039 only.

| tensor | shape | dtype | scale (divisor) |
|---|---|---|---|
| input | `[1, 1, 3, 1334]` | int8 | 0.0186600964516 |
| output | `[1, 1, 1334, 5]` | int8 | 0.00781235331669 |

```text
q = clip(round(x / scale), -127, 127)
x ≈ q * scale
```

Rounding is nearest-even. Input ranges use p99.5 clipping; 857 / 200100 input
samples (0.4283%) exceed the symmetric code range across all 50 cases.
Softmax probabilities use absmax. Weight scales are per output channel;
activation scales are per tensor. Bias uses the explicitly annotated int16 range.

- `input_XXX.npy`: quantized model inputs.
- `output_XXX.npy`: **float golden outputs quantized at the boundary**, not
  outputs of the full quantized pipeline.
- `scales.tsv`: boundary divisors and their reciprocal multipliers.
- `layer_scales.tsv`: node bit depths and divisor scales, including bias bits.
- `act_maxabs.json`: float tensor maxabs values over calibration cases 000–039;
  the q8 model uses percentiles for input/Conv ranges instead.
- `provenance.json`: model SHA-256, calibration interval and clipping count.
- `layer_ref/`: the full quantized pipeline reference for case 000.

Regenerate with `experiments/cascade_train/make_test_vectors.py` after model
annotation changes. Full-pipeline held-out frame agreement / decoded identity
against the float model is **86.42% / 72.67%** on cases 040–049. These are model
agreement metrics, not accuracy against ground-truth bases.

See `experiments/cascade_train/QUANT_V2.md` for the arithmetic, reproduction
commands and TargetCall reference tests. No ACModelSight changes are required;
end-to-end agreement with the unmodified quantizer and hardware is unverified.
No `exp_scale` relation is assumed:
ACModelSight's RHB-Lite Softmax exports its two divisor scales and `mode=0`.
