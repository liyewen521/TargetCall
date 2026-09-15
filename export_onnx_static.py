#!/usr/bin/env python3
"""
Export Bonito (default) and all LightCall models to STATIC-shape ONNX.
Fixed batch=1, length=4000 (Bonito's default chunk size). No dynamic axes.
"""
import os, sys, glob
import torch
import toml

sys.path.insert(0, '/Projects/ACCESS_IVFS/TargetCall')
os.chdir('/Projects/ACCESS_IVFS/TargetCall')

from bonito.util import load_symbol

MODELS_DIR = '/Projects/ACCESS_IVFS/TargetCall/bonito/models'
OUT_DIR = '/Projects/ACCESS_IVFS/onnx_models'
os.makedirs(OUT_DIR, exist_ok=True)

BATCH = 1
LENGTH = 4000

MODELS = {
    'bonito_default': ('default',   None),
    'TINYX0111':      ('TINYX0111', 'tinynoskipx0111'),
    'TINYX011':       ('TINYX011',  'tinynoskipx011'),
    'TINYX01':        ('TINYX01',   'tinynoskipx01'),
    'TINYX2':         ('TINYX2',    'tinynoskipx2'),
    'TINYX3':         ('TINYX3',    'tinynoskipx3'),
}

TINY_CLASS = {
    'tinynoskipx4': 'ModelTinyNoSkipX4',
    'tinynoskipx1': 'ModelTinyNoSkipX1',
    'tinynoskipx2': 'ModelTinyNoSkipX2',
    'tinynoskipx3': 'ModelTinyNoSkipX3',
    'tinynoskipx01': 'ModelTinyNoSkipX01',
    'tinynoskipx011': 'ModelTinyNoSkipX011',
    'tinynoskipx0111': 'ModelTinyNoSkipX0111',
}


def load_model(dirname, modeltype):
    config = toml.load(os.path.join(dirname, 'config.toml'))
    weight_files = sorted(glob.glob(os.path.join(dirname, 'weights_*.tar')),
                          key=lambda x: int(x.split('_')[-1].replace('.tar', '')))
    weights_path = weight_files[-1]

    if modeltype and 'tinynoskip' in modeltype:
        model = load_symbol(config, TINY_CLASS[modeltype])(config)
    else:
        model = load_symbol(config, 'Model')(config)

    checkpoint = torch.load(weights_path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    if model_keys != ckpt_keys:
        from bonito.util import match_names
        remap = match_names(state_dict, model)
        state_dict = {mk: state_dict[ck] for ck, mk in remap.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


for name, (config_dir, modeltype) in MODELS.items():
    print(f"Exporting {name} (static {BATCH}x1x{LENGTH}) ...")
    try:
        model = load_model(f"{MODELS_DIR}/{config_dir}", modeltype)
    except Exception as e:
        print(f"  FAILED: {e}")
        continue

    dummy = torch.randn(BATCH, 1, LENGTH)
    out_path = f"{OUT_DIR}/{name}.onnx"
    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=['signal'],
        output_names=['log_probs'],
        opset_version=14,
        do_constant_folding=True,
        # 不传 dynamic_axes → 导出后 shape 固定
    )
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  -> {out_path} ({size_mb:.1f} MB)")

print(f"\nDone. Static models in: {OUT_DIR}")
