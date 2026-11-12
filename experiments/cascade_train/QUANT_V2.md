# tinyx3_ivfs 的 RHB-Lite 量化与数值参考

数值约定以 ACModelSight `yuhao_quant@1be153e` 的
`backend/base/quant_modules/quantizer.py` 和 `backend/rhblite/quant_modules/conv.py`
为只读参考。本目录的修复仅涉及 TargetCall，不需要安装或修改 ACModelSight。
软件参考采用激活按张量量化、权重按输出通道量化，二者均为 int8；
与未修改的量化器及芯片的端到端一致性尚未验证。

## 芯片测试交接

本次上板使用 `onnx_models/tinyx3_ivfs_q8.onnx`；其 SHA-256 为
`61ad54a0ceb0ab773e7117d65f242ff253a512bf7da825211e0ad08fa989b379`。
`tinyx3_ivfs.onnx` 是另一组 absmax 标定参数，不能与现有 q8 向量混用。
记录本次 checkout 的 `git rev-parse HEAD`、编译器版本和实际芯片量化配置。

| 文件（仓库根目录下） | 测试用途 |
|---|---|
| `test_vectors/tinyx3_ivfs_folded_int8/input_000.npy` | 首次上板输入，int8、`[1,1,3,1334]`，已折叠且已量化，直接使用整数码 |
| 同目录 `input_001.npy` 至 `input_049.npy` | 后续测试输入；040–049 为留出样本 |
| 同目录 `layer_ref/all_layers.i8.npz` | case 000 的 152 个节点参考；`Softmax` 键是 `[1,1,1334,5]` 的最终 int8 输出 |
| 同目录 `layer_ref/manifest.tsv` | 每个节点的名称、形状和 scale |
| 同目录 `layer_ref/conv0_Conv.acc.i32.npy` | 首层含整数 bias 的 int32 累加结果，形状 `[1,12,1,1334]` |

编译时保留本模型的量化标注和静态 scale；若重新标定，需同步重建所有向量与参考。
ONNX 权重仍是浮点初始化器，由支持这些量化属性的编译流程处理。
`layer_ref/_runtime.onnx` 是去掉标注后的浮点诊断模型，不能用作此次量化部署模型。
`output_XXX.npy` 是浮点 golden 在输出边界的量化结果，不能用作整条整数流水线的逐值期望。

**芯片 bias 位宽已确认为 int16，与当前产物一致。** 编译器需要使用相同的 bias 位宽、
`s_input * s_weight[channel]` 除数量化和对称饱和规则；配置不同应先调整 TargetCall 产物。
硬件 Softmax LUT 和乘法/移位近似尚未验证，逐层差异需结合实际实现定位。
本次交接不要求修改 ACModelSight。

安装下文依赖后，在仓库根目录核对文件并运行软件检查：

```bash
git rev-parse HEAD
sha256sum onnx_models/tinyx3_ivfs_q8.onnx
python -m pytest -q experiments/cascade_train/tests
python experiments/cascade_train/compare_layers.py --sim chip_case000.npz
```

芯片 dump 的 NPZ 键使用 ONNX 节点名或其归一化名称（`/` 改为 `_`，移除开头的 `_`），
数组使用有符号整数码及 `manifest.tsv` 中的逻辑形状。需先还原芯片内部的打包布局；
不把输入、权重或累加器放入这个逐层输出 NPZ。默认比较检查全部 152 个节点，
返回 0 表示与软件参考逐值一致，1 表示数值差异，2 表示缺层或格式问题。

需要测试其他 case 时，为该 case 新建参考目录，例如 case 040：

```bash
python experiments/cascade_train/dump_layers.py --idx 40 --out /tmp/ivfs_case040_ref
python experiments/cascade_train/compare_layers.py --ref /tmp/ivfs_case040_ref/all_layers.i8.npz --sim chip_case040.npz
```

`dump_layers.py --idx` 自动读取对应浮点输入并量化；不要将已有 int8 输入传给 `--vector` 再量化。

## 固定数值规则

- 所有 scale 都是**除数**：`q = clip(round(x / s), -127, 127)`，`x ≈ q * s`。
  舍入使用 nearest-even；不使用 -128，不允许强制转换时回绕。
- 权重 scale 为 `max(abs(w_channel)) / 127`，保存于 `weight_ch_scales`。
- bias 按 `s_input * s_weight[channel]` 量化并饱和后加入整数累加器。
  本项目软件参考显式配置 **int16 bias**（范围 -32767 到 32767）；见 `rhblite.yaml`。
  该位宽与用户确认的芯片配置一致。
  `attach_quant_v2.py --bias-bits` 可设 8 或 16，生成的 Conv 必须携带 `bias_bitdepth`，
  仿真禁止在缺少该属性时猜测位宽。该属性由 TargetCall 标注工具写入，
  并非上述版本的 ACModelSight Conv 默认导出字段。
- 每个激活张量有一个 scalar scale；Conv 输入继承生产者的输出 scale。
  本模型给 ReLU 选用与输入相同的 scale；Reshape/Transpose 必须保持 scale。
  `*_quant_dim: [0]` 对本模型固定 batch=1 的输入表示一个 scalar。
- Softmax 使用量化器的浮点 `softmax` 后量化，导出 input/output scale 与 `mode=0`。
  不再添加没有源码依据的 `exp_scale`。

`quant_reference.py` 是评估和逐层导出的共同实现。卷积用精确整数值计算并检查
int32 累加范围，再按 ACModelSight float32 wrapper 的运算顺序重新量化。
这是软件数值参考；硬件中乘法/移位近似、溢出策略或 Softmax LUT 仍需编译器及上板验证。

## 校准和交付文件

| 文件 | 静态范围选择 | 标定向量 |
|---|---|---|
| `onnx_models/tinyx3_ivfs.onnx` | absmax | 000–039 |
| `onnx_models/tinyx3_ivfs_q8.onnx` | input / Conv 输出按 p99.5 截断；Softmax 概率按 absmax | 000–039 |

040–049 不参与标定。p99.5 是本项目的离线范围选择策略，**不是** ACModelSight 默认的
absmax 标定流程；它不改变 RHB-Lite 的固定量化算术。两份 ONNX 的浮点权重保持不变。
模型 metadata 记录 scale 语义、backend、bias 位宽、标定范围和 percentile。

旧版本引用的 `quantizer/utils/act_quant.py` 不是当前 public API 的实际入口；
旧 `ctc_test.tinyx3.onnx` 的乘数约定也不再作为本项目的接口依据。

## 安装 TargetCall 工具依赖

```bash
# 使用 Python 3.11 的虚拟环境
python -m pip install -r experiments/cascade_train/requirements.txt
```

`rhblite.yaml` 记录本次软件参考的位宽配置。本目录的 ONNX 标注工具显式覆盖
Conv、ReLU、Reshape、Transpose 和 Softmax 节点；重建流程不调用 ACModelSight。

## 重建与验证

在 TargetCall 根目录执行：

```bash
python experiments/cascade_train/attach_quant_v2.py --pct 100 --out onnx_models/tinyx3_ivfs.onnx
python experiments/cascade_train/attach_quant_v2.py --stats-out test_vectors/tinyx3_ivfs_folded_int8/act_maxabs.json
python experiments/cascade_train/make_test_vectors.py
python experiments/cascade_train/dump_layers.py --idx 0
python experiments/cascade_train/check_quant.py --json experiments/cascade_train/evaluation_q8.json
python experiments/cascade_train/check_quant.py --model onnx_models/tinyx3_ivfs.onnx --json experiments/cascade_train/evaluation_absmax.json
python -m pytest -q experiments/cascade_train/tests
```

TargetCall 的 10 项测试覆盖饱和与舍入、整数 bias、层间整数码传递、比较器失败状态，
以及全部 50 组边界向量和 case 000 的 152 个节点 dump 与本项目参考实现的一致性。
提交前在临时目录完整重建了两份 ONNX、标定统计、129 个向量及参考文件和两份评估记录，
与交付产物一致；比较器命令行的通过、单个整数码差异和缺层场景分别返回 0、1、2。
这些检查不构成与原版 ACModelSight 的独立交叉验证。
原版 RHB-Lite Conv 的 bias 广播问题使本次直接调用未能完成；量化器源码保持不变。

留出向量 040–049，相对浮点 golden（**不是相对真实碱基标签的准确率**）：

| 静态范围 | 帧 argmax 一致率 | 解码 identity |
|---|---:|---:|
| absmax | 78.44% | 64.37% |
| p99.5 | 86.42% | 72.67% |

完整结果与模型 SHA-256 在 `evaluation_absmax.json` / `evaluation_q8.json`。
旧版 `[-256,255]` 仿真、浮点 bias 和全 50 条标定产生的指标已撤下，不能与这些结果混用。

## 逐层比较

```bash
python experiments/cascade_train/compare_layers.py --sim sim_layers.npz
```

默认要求所有节点、形状和整数码完全一致。缺层、额外节点、重复归一化名称、非整数或空数组、
形状不一致返回 2；数值不一致返回 1；通过返回 0。诊断部分 dump 时必须显式使用
`--allow-partial`，允许展平布局比较时用 `--flatten`，容忍一定错误比例时用 `--thr`。
这些放宽选项不用于默认验收。
