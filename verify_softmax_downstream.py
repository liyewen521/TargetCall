#!/usr/bin/env python3
"""Verify downstream softmax change: load model, check forward outputs probs, run decode."""
import sys, glob, torch, toml
sys.path.insert(0, '/Projects/ACCESS_IVFS/TargetCall')

from bonito.util import load_symbol

# load TINYX011
config = toml.load('/Projects/ACCESS_IVFS/TargetCall/bonito/models/TINYX011/config.toml')
model = load_symbol(config, 'ModelTinyNoSkipX011')(config)
weights = sorted(glob.glob('/Projects/ACCESS_IVFS/TargetCall/bonito/models/TINYX011/weights_*.tar'),
                 key=lambda x: int(x.split('_')[-1].replace('.tar', '')))[-1]
ckpt = torch.load(weights, map_location='cpu')
state_dict = ckpt.get('model_state_dict', ckpt)

if set(model.state_dict().keys()) != set(state_dict.keys()):
    from bonito.util import match_names
    remap = match_names(state_dict, model)
    state_dict = {mk: state_dict[ck] for ck, mk in remap.items()}
model.load_state_dict(state_dict, strict=False)
model.eval()

# forward
x = torch.randn(1, 1, 4000)
with torch.no_grad():
    y = model(x)

print(f'forward 输出 shape: {tuple(y.shape)}')
print(f'输出数值范围: [{y.min().item():.4f}, {y.max().item():.4f}]')
print(f'每位置概率和 (sum over 5 碱基): min={y.sum(-1).min().item():.4f} max={y.sum(-1).max().item():.4f}')

is_prob = (y.min() >= 0) and (y.max() <= 1) and torch.allclose(y.sum(-1), torch.ones_like(y.sum(-1)), atol=1e-5)
print(f'输出是概率 (0~1 且和=1): {is_prob}')

# decode：真实 basecall 流程会 permute('TNC','NTC') + unbatchify 得到 2 维 (T, C)
scores = y.permute(1, 0, 2).squeeze(0)  # (1334,1,5) -> (1,1334,5) -> (1334,5)
seq = model.decode(scores)
print(f'decode 输出序列长度: {len(seq)}')
print(f'decode 输出前 30 个碱基: {seq[:30]}')
print(f'decode 正常: {len(seq) > 0}')
