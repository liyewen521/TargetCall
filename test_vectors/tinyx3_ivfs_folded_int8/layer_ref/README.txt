逐层参考数据（input_000）：编译器 / 仿真器中间结果对比
==================================================

用途
----
量化仿真最后的输出对不上时，需要判断“从哪一层开始偏”。
这里给出 input_000 在 float 模型和 int8 语义下每一层的参考值，
逐层比 int8 codes 即可定位第一个分歧点。

复现
----
    python experiments/cascade_train/dump_layers.py --idx 0 \
        --out test_vectors/tinyx3_ivfs_folded_int8/layer_ref

    --model   默认 onnx_models/tinyx3_ivfs.onnx
              （md5 c7b25b35ea92163f72f8cedbf1117c3f）
    --vector  默认 test_vectors/tinyx3_ivfs_folded/input_%03d.npy

模型上的自定义量化属性 onnxruntime / onnx.checker 不认识，
脚本会在内存里剥掉属性、并把每个节点的输出都挂成图输出后再跑。

文件
----
000_input.i8.npy            网络输入的 int8 codes（scale=26.2242546 = 127/max|input|）
                            与 test_vectors/tinyx3_ivfs_folded_int8/input_000.npy 逐字节相同
NNN_<node>.i8.npy           该节点输出的 int8 codes（用该节点的 output_scale 量化）
NNN_<node>.f32.npy          该节点输出的 float 参考值
conv0_Conv.acc.i32.npy      conv0 的 int32 累加器（含 bias），12x1334
all_layers.i8.npz           全部 152 个节点的 int8 codes，key = onnx 节点名
manifest.tsv                节点 -> scale -> 文件 的对照表

比对方法
--------
1. 输入必须是同一份 000_input.i8.npy，不要各自量化。
2. 只比 int8 codes，不要比 float：
   int8 权重的结果与 float 权重再量化本身就差 ±1
   （conv0 上 24% 元素差 1、0.1% 差 2），这是量化噪声不是 bug。
3. 每层看三项：不等比例、差值直方图、|差|>1 的个数。
   第一个“不等比例 > 5%”的节点就是问题点。

    python experiments/cascade_train/compare_layers.py \
        --ref all_layers.i8.npz --sim <仿真器 dump>.npz

conv0 的快速判据
----------------
* 差值在每个通道上是同一个常数 -> 丢了 bias。偏移应为
      -8 -6 -2 -2 -13 +3 -14 -6 -1 +5 -1 -15     (= round(bias * output_scale))
* 差值随数值成比例 -> 权重 scale 用错（per-channel 当成 per-tensor，
  或把 scale 当成了除数）。
* 只有少数点差很多 -> 饱和 / 溢出。本向量所有节点的饱和计数都是 0。

注意
----
* Relu / Reshape / Transpose 的 output_scale 就是上游的 output_scale，
  不要重复量化，这几个节点应当与上游逐位相等。
* conv0 的 requant 乘数 m_c = s_out / (s_in * s_w[c])，约 0.0021~0.011；
  累加器范围 [-26972, 25172]，int16 会溢出，必须 int32。

