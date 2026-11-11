# tinyx3_ivfs 量化方案 v2（受支持配置：per-tensor + p99.5 截断，全 int8）

生成：`attach_quant_v2.py`　校验：`check_quant.py`　向量重建：`make_test_vectors.py`　逐层参考：`dump_layers.py`

```bash
python experiments/cascade_train/attach_quant_v2.py        # -> onnx_models/tinyx3_ivfs_q8.onnx
python experiments/cascade_train/check_quant.py           # 整数仿真，打印每条 read 的 identity
python experiments/cascade_train/make_test_vectors.py     # 重建 test_vectors/*_int8
python experiments/cascade_train/dump_layers.py --idx 0   # 重建 layer_ref
```

## 约束（按编译器实际支持的能力，已对照参考文件核实）

1. **激活只能 per-tensor**。`ac_model_sight/quantizer/utils/act_quant.py` 里 per-channel 分支直接
   `raise NotImplementedError("quant_dim not implemented")`，Conv 的 `input`/`output` 不按通道量化。
   （权重仍走 `weight_ch_scales`，per-channel，这是支持的。）
2. **一个张量只有一个 scale**。参考文件 `ctc_test.tinyx3.onnx` 里 29 个 ReLU 的 `output_scale`
   与前一个 Conv 完全相等，120 个 Conv 的 `input_scale` 全部等于上游 `output_scale`，零例外。
   所以 ReLU / Reshape / Transpose 的 scale 是**继承**，不是单独标定。
3. 位深全 int8（权重最多 8 bit；激活抬到 16 bit 实测无收益，见文末）。

## 改动

| # | 项 | 旧 `tinyx3_ivfs.onnx` | 新 `tinyx3_ivfs_q8.onnx` |
|---|---|---|---|
| 1 | 标定统计量 | absmax | p99.5 分位截断 |
| 2 | ReLU / Reshape / Transpose | 各自 absmax（恰好与上游相等） | 显式继承上游张量 scale |
| 3 | 位深 | 全 int8 | 全 int8 |

权重张量逐字节未变（脚本内逐字节比对，`initializers unchanged: True`），生成后自检
`invariant violations: 0`。scale 仍是**乘数**（`q = round(real × scale)`，`scale = qmax / 参考幅度`），
与参考文件一致（其 `max|w| × weight_ch_scales = 128`）。

## 实测

留出集 vector 040–049（标定只用 000–039）：

| 截断 | 帧 argmax | identity |
|---|---|---|
| absmax（旧方案等价） | 78.19% | 64.38% |
| p99 | 85.10% | 68.47% |
| **p99.5（采用）** | **86.72%** | **72.79%** |
| p99.9 | 86.00% | 72.43% |
| p99.99 | 83.04% | 69.21% |

仓库里的文件用全部 50 条标定，在 040–049 上是 88.67% argmax / 77.88% identity（这 10 条参与了标定，
数字偏乐观）；**72.8% 是可信的外推值**。

per-channel 版本（不受支持）能到 79.8%，但那不是编译器能吃的东西，不作为交付目标。

## 16 bit 的结论

| 方案 | 帧 argmax | identity |
|---|---|---|
| 全 int8 | 90.04% | **80.65%**（per-channel 版） |
| depthwise 激活抬 int16、权重 int8 | 90.13% | 79.21% |

权重受限于 8 bit 时，抬激活精度没有收益（略低，属噪声），所以保持全 int8。

## 已知边界

- 仿真里 softmax 按理想 exp 计算，编译器实际走 `exp_scale` 查表，可能再差一点。
- 剩余约 27 pt 来自 120 层逐层 requant 的累积噪声。per-tensor 激活是主要瓶颈——原本靠 per-channel
  能再拿 7 pt，但硬件不支持。再往上只能靠 block 内不 requant（融合）或 QAT。

