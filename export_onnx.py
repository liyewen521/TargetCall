#!/usr/bin/env python3
"""
Export Bonito (default) and all LightCall models to ONNX.
Run from the ACCESS_IVFS directory.
"""
import os, sys, glob
import torch
import toml

# Add TargetCall to path
sys.path.insert(0, '/Projects/ACCESS_IVFS/TargetCall')
os.chdir('/Projects/ACCESS_IVFS/TargetCall')

from bonito.util import load_symbol
from bonito.nn import layers

MODELS_DIR = '/Projects/ACCESS_IVFS/TargetCall/bonito/models'
OUT_DIR = '/Projects/ACCESS_IVFS/onnx_models'
os.makedirs(OUT_DIR, exist_ok=True)

# Model name -> (config_dir, modeltype)
MODELS = {
    'bonito_default':  ('default',    None),
    'TINYX0111':       ('TINYX0111',  'tinynoskipx0111'),
    'TINYX011':        ('TINYX011',   'tinynoskipx011'),
    'TINYX01':         ('TINYX01',    'tinynoskipx01'),
    'TINYX2':          ('TINYX2',     'tinynoskipx2'),
    'TINYX3':          ('TINYX3',     'tinynoskipx3'),
}


def load_model(dirname, modeltype):
    """Load model with weights, matching bonito's util.load_model."""
    config = toml.load(os.path.join(dirname, 'config.toml'))

    # Find latest weights
    weight_files = sorted(glob.glob(os.path.join(dirname, 'weights_*.tar')),
                          key=lambda x: int(x.split('_')[-1].replace('.tar', '')))
    if not weight_files:
        raise FileNotFoundError(f"No weights in {dirname}")
    weights_path = weight_files[-1]
    print(f"  weights: {os.path.basename(weights_path)}")

    # Instantiate model
    if modeltype and 'tinynoskip' in modeltype:
        class_name = {
            'tinynoskipx4': 'ModelTinyNoSkipX4',
            'tinynoskipx1': 'ModelTinyNoSkipX1',
            'tinynoskipx2': 'ModelTinyNoSkipX2',
            'tinynoskipx3': 'ModelTinyNoSkipX3',
            'tinynoskipx01': 'ModelTinyNoSkipX01',
            'tinynoskipx011': 'ModelTinyNoSkipX011',
            'tinynoskipx0111': 'ModelTinyNoSkipX0111',
        }[modeltype]
        model = load_symbol(config, class_name)(config)
    else:
        model = load_symbol(config, 'Model')(config)

    # Load weights
    checkpoint = torch.load(weights_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Handle key remapping
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    if model_keys != ckpt_keys:
        print(f"  key mismatch: model={len(model_keys)} ckpt={len(ckpt_keys)}", end=" ")
        # Try matching by shape
        from bonito.util import match_names
        remap = match_names(state_dict, model)
        new_state = {}
        for ck, mk in remap.items():
            new_state[mk] = state_dict[ck]
        state_dict = new_state
        print("→ remapped")

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


for name, (config_dir, modeltype) in MODELS.items():
    print(f"\n{'='*50}")
    print(f"Exporting: {name}")
    try:
        model = load_model(f"{MODELS_DIR}/{config_dir}", modeltype)
    except Exception as e:
        print(f"  FAILED: {e}")
        continue

    # Count params
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params:,}")

    # Dummy input: batch=1, channels=1, length=4000 (typical chunk size)
    dummy = torch.randn(1, 1, 4000)
    print(f"  input shape: {dummy.shape}")

    out_path = f"{OUT_DIR}/{name}.onnx"
    try:
        torch.onnx.export(
            model,
            dummy,
            out_path,
            input_names=['signal'],
            output_names=['log_probs'],
            dynamic_axes={
                'signal': {0: 'batch', 2: 'length'},
                'log_probs': {0: 'time', 1: 'batch'}
            },
            opset_version=14,
            do_constant_folding=True,
        )
        size_mb = os.path.getsize(out_path) / (1024*1024)
        print(f"  exported: {out_path} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"  ONNX export FAILED: {e}")

print(f"\nDone. Models in: {OUT_DIR}")
