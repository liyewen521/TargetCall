# Table 2: Operator Specification

| pyTorch Operator | ONNX Operator | Remarks |
| --- | --- | --- |
| `torch.nn.Conv2d` | Conv | Kernel H/W: [1-11], Stride: [1,2], Dilation: [1,2] |
| `torch.nn.MaxPool2d` | MaxPool | Kernel H/W: [1-11], Stride: [1,2]] |
| `torch.nn.AvgPool2d` | AveragePool | Kernel H/W: [1-11], Stride: [1,2] |
| `+`, `*`, `/` | Add/Mul/Div | May be pipelined with other operators |
| `torch.concat` | Concat | Concat on C/H/W (Convolutional Neural Network (CNN)) or head/W/C (Attention) |
| `torch.nn.Linear` | Gemm | |
| `torch.sum` | ReduceSum | High efficiency with ReduceSum on C direction |
| `torch.matmul` | MatMul | |
| `torch.nn.Linear` | MatMul | |
| `fn.interpolate` | Resize | Scale Factor = 2 or 4, mode=bilinear or bi-cubic |
| `fn.softmax` | Softmax | High efficiency with Softmax on C direction |
| `torch.nn.LayerNorm` | LayerNormalization | elementwise_affine=False, eps=0.65234375, High efficiency with Layernorm on C direction |
| `torch.view`<br>`torch.reshape` | Reshape | |
| `torch.permute` | Transpose | Transpose on X/Y |
| `fn.relu` | Relu | |
| `fn.nn.PReLU` | PRelu | |
| `fn.leaky_relu` | LeakyRelu | |
| `fn.sigmoid`,<br>`fn.hardswish`, `fn.tanh`,<br>`fn.GELU`, `torch.erf` | Sigmoid,<br>Hardswish,<br>Tanh, Gelu,<br>Erf | Using PWL for approximation |
| `fn.pixel_shuffle` | DepthToSpace | upscale_factor=2 |
| `fn.pixel_unshuffle` | Space2Depth | downscale_factor=2 |
| `torch.roll` | Split + Concat | Roll on H/W |
| `torch.sin`, `torch.cos` | Sin, Cos | Approximated with LUT |
| `[:a, :, ...]` | Slice | |
