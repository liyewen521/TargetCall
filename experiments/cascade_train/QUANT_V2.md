# tinyx3_ivfs 量化方案 v2（per-channel 激活 + p99.9 截断，全 int8）

生成：`attach_quant_v2.py`　校验：`check_quant.py`

```bash
python experiments/cascade_train/attach_quant_v2.py     # -> onnx_models/tinyx3_ivfs_q8.onnx
python experiments/cascade_train/check_quant.py         # 用文件自身的注释做整数仿真
python experiments/cascade_train/check_quant.py --model onnx_models/tinyx3_ivfs.onnx   # 对比旧文件
```

## 改了什么

| # | 改动 | 旧 `tinyx3_ivfs.onnx` | 新 `tinyx3_ivfs_q8.onnx` |
|---|---|---|---|
| 1 | 激活 scale 粒度 | per-tensor 标量 `input_scale` / `output_scale` | per-channel `input_scales` / `output_scales` + `*_quant_dim = [1]` |
| 2 | 标定统计量 | absmax | p99.9 分位截断（允许少量饱和） |
| — | 位深 | 全 int8 | **全 int8**（权重 int8，激活 int8） |

权重张量逐字节未变（脚本内做初始量逐字节比对，`initializers unchanged: True`）。
scale 仍是**乘数**（`q = round(real × scale)`，`scale = qmax / 参考幅度`），与编译器参考文件 `ctc_test.tinyx3.onnx` 一致（其 `max|w| × weight_ch_scales = 128`）。
decoder conv、Reshape/Transpose、Softmax 保持**标量 int8**——对 logits 做 per-class scale 会直接改变 argmax。

## 实测（留出集 vector 040–049，标定未使用）

| 方案 | 帧 argmax | 解码 identity |
|---|---|---|
| 旧文件（per-tensor / absmax） | 78.31% | 64.10% |
| **`tinyx3_ivfs_q8.onnx`（①+②）** | **90.43%** | **79.80%** |
| 理想上界（只量化输出，不可实现） | — | 97.85% |

分项（40 向量标定 / 10 向量评估）：per-channel +14.5 pt，p99.9 单独 +8.9 pt，两者叠加 +16.5 pt。

## 16 bit 的结论

按硬件约束（权重最多 8 bit）复测：

| 方案 | 帧 argmax | identity |
|---|---|---|
| ①+② 全 int8 | 90.04% | **80.65%** |
| ①+② depthwise 激活抬 int16、权重仍 int8 | 90.13% | 79.21% |

**16 bit 激活没有收益**（略低，属噪声），瓶颈在 8 bit 权重。所以方案保持全 int8，不需要混合位深。

## 编译器侧仍需确认

新文件对隐藏层用 `input_scales` / `output_scales`（FLOATS 列表）+ `*_quant_dim = [1]`，参考文件里只有标量 `input_scale`。

若编译器暂不支持，可用 `attach_quant_v2.py --scalar` 生成标量版（schema 与旧文件完全一致，实测 identity 73.48%），但这份不随仓库提供，以免和主方案混淆。

## 已知边界

- 仿真里 softmax 按理想 exp 计算，编译器实际走 `exp_scale` 查表，可能再差一点。
- 剩余约 20 pt 来自 120 层逐层 requant 的累积噪声，不是标定问题；再往上要么把 depthwise block 做成块内不 requant，要么 QAT。

