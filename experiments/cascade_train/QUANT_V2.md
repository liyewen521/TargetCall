# tinyx3_ivfs 量化方案 v2（per-channel + p99.9 + int16 depthwise）

生成脚本：`attach_quant_v2.py`　校验脚本：`check_quant.py`

```bash
python experiments/cascade_train/attach_quant_v2.py            # -> onnx_models/tinyx3_ivfs_q8.onnx
python experiments/cascade_train/check_quant.py                # 用文件自身的注释做整数仿真
python experiments/cascade_train/check_quant.py --model onnx_models/tinyx3_ivfs.onnx   # 对比旧文件
```

## 改了什么

| # | 改动 | 旧 (`tinyx3_ivfs.onnx`) | 新 (`tinyx3_ivfs_q8.onnx`) |
|---|---|---|---|
| 1 | 激活 scale 粒度 | per-tensor 标量 `input_scale`/`output_scale` | per-channel `input_scales`/`output_scales` + `input_quant_dim`/`output_quant_dim` = `[1]` |
| 2 | 标定统计量 | absmax | p99.9 分位截断（允许少量饱和） |
| 3 | depthwise 区块精度 | 全 int8 | 激活与权重 int16（`*_bitdepth = 16`，`weight_ch_scales` 按 32767 定标） |

保持不变：scale 仍是**乘数**（`q = round(real * scale)`，`scale = qmax / 参考幅度`），与编译器参考文件 `ctc_test.tinyx3.onnx` 一致（其 `max|w| × weight_ch_scales = 128`）。
decoder conv、Reshape/Transpose、Softmax 保持**标量 int8**——对 logits 做 per-class scale 会直接改变 argmax。
权重张量一个字节都没动（脚本里做了初始量逐字节比对，`initializers unchanged: True`）。

## 实测（留出集：vector 040–049，标定未使用）

| 方案 | 帧 argmax 一致 | 解码 identity |
|---|---|---|
| 旧文件（per-tensor/absmax/int8） | 78.31% | 64.10% |
| 新文件（1+2+3） | **93.28%** | **83.30%** |
| 仅改 2（per-tensor + p99.9） | 85.76% | 73.08% |
| 理想上界（只量化输出，不可实现） | — | 97.85% |

分项贡献（40 向量标定 / 10 向量评估）：per-channel +14.5 pt identity，p99.9 +8.9 pt，int16 depthwise +3.8 pt。

## 编译器侧需要确认的两件事

1. 新文件对隐藏层用的是 `input_scales` / `output_scales`（FLOATS 列表）+ `*_quant_dim = [1]`，参考文件里只有标量 `input_scale`。`net_csim` 必须支持逐通道激活 scale，否则这条走不通——退路是只上第 2 条，标量 schema 不变，identity 73.1%。
2. depthwise 的 `weight_bitdepth = 16`，`weight_ch_scales` 按 32767 定标；下游 8 bit 节点读取 16 bit 张量时，requant 需要按 `s_out/(s_in*s_w)` 自行换位深。

## 已知边界

- 仿真里软层是按“理想 exp”算的，编译器实际走 `exp_scale` 查表，可能再差一点点。
- 剩余 ~17 pt 主要来自 120 层逐层 requant 的累积噪声，不是标定问题；再往上要么融合 depthwise block（块内不 requant），要么 QAT。

