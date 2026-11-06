"""Explicit inference network for the TINYX3 model.

Every convolution and batch-norm layer is declared explicitly in
__init__; there are no block wrappers or generated layer specs.
The original stride-3 stem is rewritten as a stride-1 3x3 convolution over a
folded input, so the network only uses stride 1/2 convolutions.
A depthwise kernel wider than 15 is a cascade of depthwise convolutions of
width <= 15 with the same receptive field. The new weights are not a split of
the original long kernel.

Input:  [batch, 1, 3, time]  (samples folded as M[r, c] = [0, x, 0][3*c + r])
Output: [batch, 1, time, 5] probabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


MAX_FILTER = 15

input_layouts=["CHW"]
ifmap_sz=[[1,3,1334]]

def cascade_depthwise(channels, kernel_w, maxk=MAX_FILTER):
    """Depthwise cascade whose receptive field is kernel_w and each kernel <= maxk.

    Stride-1 layers add (kernel - 1) to the receptive field. kernel_w is odd, so
    every piece is odd and same padding keeps the width unchanged.
    """
    layers = []
    remaining = kernel_w - 1
    while remaining > 0:
        step = min(maxk - 1, remaining)
        kernel = step + 1
        layers.append(nn.Conv2d(
            channels, channels, kernel_size=(1, kernel), stride=(1, 1),
            padding=(0, step // 2), dilation=(1, 1), groups=channels, bias=False))
        remaining -= step
    return nn.Sequential(*layers)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(1, 12, kernel_size=(3, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 1), groups=1, bias=False)
        self.bn0 = nn.BatchNorm2d(12, eps=0.001, momentum=0.1)
        self.depthwise1 = cascade_depthwise(12, 75)
        self.pointwise1 = nn.Conv2d(12, 20, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False)
        self.bn1 = nn.BatchNorm2d(20, eps=0.001, momentum=0.1)
        self.depthwise2 = cascade_depthwise(20, 75)
        self.pointwise2 = nn.Conv2d(20, 20, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False)
        self.bn2 = nn.BatchNorm2d(20, eps=0.001, momentum=0.1)
        self.depthwise3 = cascade_depthwise(20, 31)
        self.pointwise3 = nn.Conv2d(20, 20, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False)
        self.bn3 = nn.BatchNorm2d(20, eps=0.001, momentum=0.1)
        self.depthwise4 = cascade_depthwise(20, 31)
        self.pointwise4 = nn.Conv2d(20, 20, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False)
        self.bn4 = nn.BatchNorm2d(20, eps=0.001, momentum=0.1)
        self.depthwise5 = cascade_depthwise(20, 31)
        self.pointwise5 = nn.Conv2d(20, 20, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False)
        self.bn5 = nn.BatchNorm2d(20, eps=0.001, momentum=0.1)
        self.depthwise6 = cascade_depthwise(20, 123)
        self.pointwise6 = nn.Conv2d(20, 12, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False)
        self.bn6 = nn.BatchNorm2d(12, eps=0.001, momentum=0.1)
        self.depthwise7 = cascade_depthwise(12, 55)
        self.pointwise7 = nn.Conv2d(12, 12, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False)
        self.bn7 = nn.BatchNorm2d(12, eps=0.001, momentum=0.1)
        self.depthwise8 = cascade_depthwise(12, 55)
        self.pointwise8 = nn.Conv2d(12, 12, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False)
        self.bn8 = nn.BatchNorm2d(12, eps=0.001, momentum=0.1)
        self.depthwise9 = nn.Conv2d(12, 12, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), dilation=(1, 1), groups=12, bias=False)
        self.pointwise9 = nn.Conv2d(12, 12, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False)
        self.bn9 = nn.BatchNorm2d(12, eps=0.001, momentum=0.1)
        self.depthwise10 = nn.Conv2d(12, 12, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), dilation=(1, 1), groups=12, bias=False)
        self.pointwise10 = nn.Conv2d(12, 12, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False)
        self.bn10 = nn.BatchNorm2d(12, eps=0.001, momentum=0.1)
        self.depthwise11 = nn.Conv2d(12, 12, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 1), groups=12, bias=False)
        self.pointwise11 = nn.Conv2d(12, 12, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False)
        self.bn11 = nn.BatchNorm2d(12, eps=0.001, momentum=0.1)
        self.depthwise12 = nn.Conv2d(12, 12, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 1), groups=12, bias=False)
        self.pointwise12 = nn.Conv2d(12, 12, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False)
        self.bn12 = nn.BatchNorm2d(12, eps=0.001, momentum=0.1)
        self.depthwise13 = nn.Conv2d(12, 12, kernel_size=(1, 9), stride=(1, 1), padding=(0, 4), dilation=(1, 1), groups=12, bias=False)
        self.pointwise13 = nn.Conv2d(12, 12, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False)
        self.bn13 = nn.BatchNorm2d(12, eps=0.001, momentum=0.1)
        self.depthwise14 = nn.Conv2d(12, 12, kernel_size=(1, 9), stride=(1, 1), padding=(0, 4), dilation=(1, 1), groups=12, bias=False)
        self.pointwise14 = nn.Conv2d(12, 12, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False)
        self.bn14 = nn.BatchNorm2d(12, eps=0.001, momentum=0.1)
        self.depthwise15 = cascade_depthwise(12, 115)
        self.pointwise15 = nn.Conv2d(12, 12, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False)
        self.bn15 = nn.BatchNorm2d(12, eps=0.001, momentum=0.1)
        self.depthwise16 = cascade_depthwise(12, 115)
        self.pointwise16 = nn.Conv2d(12, 12, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False)
        self.bn16 = nn.BatchNorm2d(12, eps=0.001, momentum=0.1)
        self.depthwise17 = cascade_depthwise(12, 55)
        self.pointwise17 = nn.Conv2d(12, 12, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False)
        self.bn17 = nn.BatchNorm2d(12, eps=0.001, momentum=0.1)
        self.depthwise18 = cascade_depthwise(12, 55)
        self.pointwise18 = nn.Conv2d(12, 12, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False)
        self.bn18 = nn.BatchNorm2d(12, eps=0.001, momentum=0.1)
        self.depthwise19 = cascade_depthwise(12, 25)
        self.pointwise19 = nn.Conv2d(12, 8, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False)
        self.bn19 = nn.BatchNorm2d(8, eps=0.001, momentum=0.1)
        self.depthwise20 = cascade_depthwise(8, 25)
        self.pointwise20 = nn.Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False)
        self.bn20 = nn.BatchNorm2d(8, eps=0.001, momentum=0.1)
        self.depthwise21 = nn.Conv2d(8, 8, kernel_size=(1, 9), stride=(1, 1), padding=(0, 4), dilation=(1, 1), groups=8, bias=False)
        self.pointwise21 = nn.Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False)
        self.bn21 = nn.BatchNorm2d(8, eps=0.001, momentum=0.1)
        self.depthwise22 = nn.Conv2d(8, 8, kernel_size=(1, 9), stride=(1, 1), padding=(0, 4), dilation=(1, 1), groups=8, bias=False)
        self.pointwise22 = nn.Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False)
        self.bn22 = nn.BatchNorm2d(8, eps=0.001, momentum=0.1)
        self.depthwise23 = cascade_depthwise(8, 25)
        self.pointwise23 = nn.Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False)
        self.bn23 = nn.BatchNorm2d(8, eps=0.001, momentum=0.1)
        self.depthwise24 = nn.Conv2d(8, 8, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), dilation=(1, 1), groups=8, bias=False)
        self.pointwise24 = nn.Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False)
        self.bn24 = nn.BatchNorm2d(8, eps=0.001, momentum=0.1)
        self.depthwise25 = nn.Conv2d(8, 8, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), dilation=(1, 1), groups=8, bias=False)
        self.pointwise25 = nn.Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False)
        self.bn25 = nn.BatchNorm2d(8, eps=0.001, momentum=0.1)
        self.depthwise26 = cascade_depthwise(8, 123)
        self.pointwise26 = nn.Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False)
        self.bn26 = nn.BatchNorm2d(8, eps=0.001, momentum=0.1)
        self.depthwise27 = nn.Conv2d(8, 8, kernel_size=(1, 9), stride=(1, 1), padding=(0, 4), dilation=(1, 1), groups=8, bias=False)
        self.pointwise27 = nn.Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False)
        self.bn27 = nn.BatchNorm2d(8, eps=0.001, momentum=0.1)
        self.conv28 = nn.Conv2d(8, 12, kernel_size=(1, 9), stride=(1, 1), padding=(0, 4), dilation=(1, 1), groups=1, bias=False)
        self.bn28 = nn.BatchNorm2d(12, eps=0.001, momentum=0.1)
        self.decoder = nn.Conv2d(12, 5, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=True)

        self.alphabet = ["N", "A", "C", "G", "T"]
        self.stride = 3
        self.features = 12
        self.qbias = 0.0
        self.qscale = 1.0

    def load_state_dict(self, state_dict, strict=True):
        """Accept the original 1x9 conv0 weights or already-folded 3x3 ones."""
        if "conv0.weight" in state_dict:
            weight = state_dict["conv0.weight"]
            if tuple(weight.shape[-2:]) != (3, 3):
                state_dict = dict(state_dict)
                state_dict["conv0.weight"] = (
                    weight.reshape(12, 1, 3, 3).permute(0, 1, 3, 2).contiguous()
                )
        return super().load_state_dict(state_dict, strict=strict)

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = F.relu(x)
        x = self.depthwise1(x)
        x = self.pointwise1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.depthwise2(x)
        x = self.pointwise2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.depthwise3(x)
        x = self.pointwise3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.depthwise4(x)
        x = self.pointwise4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.depthwise5(x)
        x = self.pointwise5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.depthwise6(x)
        x = self.pointwise6(x)
        x = self.bn6(x)
        x = F.relu(x)
        x = self.depthwise7(x)
        x = self.pointwise7(x)
        x = self.bn7(x)
        x = F.relu(x)
        x = self.depthwise8(x)
        x = self.pointwise8(x)
        x = self.bn8(x)
        x = F.relu(x)
        x = self.depthwise9(x)
        x = self.pointwise9(x)
        x = self.bn9(x)
        x = F.relu(x)
        x = self.depthwise10(x)
        x = self.pointwise10(x)
        x = self.bn10(x)
        x = F.relu(x)
        x = self.depthwise11(x)
        x = self.pointwise11(x)
        x = self.bn11(x)
        x = F.relu(x)
        x = self.depthwise12(x)
        x = self.pointwise12(x)
        x = self.bn12(x)
        x = F.relu(x)
        x = self.depthwise13(x)
        x = self.pointwise13(x)
        x = self.bn13(x)
        x = F.relu(x)
        x = self.depthwise14(x)
        x = self.pointwise14(x)
        x = self.bn14(x)
        x = F.relu(x)
        x = self.depthwise15(x)
        x = self.pointwise15(x)
        x = self.bn15(x)
        x = F.relu(x)
        x = self.depthwise16(x)
        x = self.pointwise16(x)
        x = self.bn16(x)
        x = F.relu(x)
        x = self.depthwise17(x)
        x = self.pointwise17(x)
        x = self.bn17(x)
        x = F.relu(x)
        x = self.depthwise18(x)
        x = self.pointwise18(x)
        x = self.bn18(x)
        x = F.relu(x)
        x = self.depthwise19(x)
        x = self.pointwise19(x)
        x = self.bn19(x)
        x = F.relu(x)
        x = self.depthwise20(x)
        x = self.pointwise20(x)
        x = self.bn20(x)
        x = F.relu(x)
        x = self.depthwise21(x)
        x = self.pointwise21(x)
        x = self.bn21(x)
        x = F.relu(x)
        x = self.depthwise22(x)
        x = self.pointwise22(x)
        x = self.bn22(x)
        x = F.relu(x)
        x = self.depthwise23(x)
        x = self.pointwise23(x)
        x = self.bn23(x)
        x = F.relu(x)
        x = self.depthwise24(x)
        x = self.pointwise24(x)
        x = self.bn24(x)
        x = F.relu(x)
        x = self.depthwise25(x)
        x = self.pointwise25(x)
        x = self.bn25(x)
        x = F.relu(x)
        x = self.depthwise26(x)
        x = self.pointwise26(x)
        x = self.bn26(x)
        x = F.relu(x)
        x = self.depthwise27(x)
        x = self.pointwise27(x)
        x = self.bn27(x)
        x = F.relu(x)
        x = self.conv28(x)
        x = self.bn28(x)
        x = F.relu(x)
        x = self.decoder(x)          # [B, 5, 1, T]
        batch, channels, _, length = x.shape
        x = x.reshape(batch, 1, channels, length)
        x = x.permute(0, 1, 3, 2)
        return F.softmax(x, dim=-1)


TinyX3IVFSModel = Model
