#!/usr/bin/env python3
"""Generate diagnostic float-boundary vectors using RHB-Lite divisor scales."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from quant_reference import in_scale, out_scale, quantize, read_annotations


def main():
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=Path, default=root / 'onnx_models/tinyx3_ivfs_q8.onnx')
    ap.add_argument('--src', type=Path, default=root / 'test_vectors/tinyx3_ivfs_folded')
    ap.add_argument('--dst', type=Path, default=root / 'test_vectors/tinyx3_ivfs_folded_int8')
    args = ap.parse_args()
    provenance = args.dst / 'provenance.json'
    if provenance.is_file() and json.loads(provenance.read_text()).get('output_kind') == 'cmodel_replay':
        ap.error('destination contains CModel vectors. Use --dst for a separate diagnostic directory')
    model, nodes, _ = read_annotations(args.model)
    active = [(n, d) for n, d in nodes if n.op_type != 'Constant']
    si, so = in_scale(active[0][1]), out_scale(active[-1][1])
    files = sorted(args.src.glob('input_*.npy'))
    if not files:
        ap.error('no input vectors found')
    args.dst.mkdir(parents=True, exist_ok=True)
    clipped = total = changed = 0
    for path in files:
        tag = path.stem.split('_')[1]
        x = np.load(path)
        y = np.load(args.src / ('output_%s.npy' % tag))
        for name, data, scale in [('input', x, si), ('output', y, so)]:
            codes = quantize(data, scale)
            target = args.dst / ('%s_%s.npy' % (name, tag))
            if not target.exists() or not np.array_equal(np.load(target), codes):
                np.save(target, codes)
                changed += 1
        clipped += int(np.sum(np.abs(np.rint(x.astype(np.float32) / si)) > 127))
        total += x.size
    rows = ['tensor\tshape\tdtype\tscale(divisor)\tmultiplier(1/scale)\tbits']
    for name, data, scale in [('input', x, si), ('output', y, so)]:
        rows.append('%s\t%s\tint8\t%.12g\t%.12g\t8' %
                    (name, ','.join(map(str, data.shape)), scale, 1.0 / scale))
    (args.dst / 'scales.tsv').write_text('\n'.join(rows) + '\n')
    rows = ['op\tname\tinput_bitdepth\toutput_bitdepth\tbias_bitdepth\tinput_scale\toutput_scale']
    for node, attrs in active:
        rows.append('\t'.join(map(str, [node.op_type, node.name, attrs.get('input_bitdepth', ''),
            attrs['output_bitdepth'], attrs.get('bias_bitdepth', ''), attrs.get('input_scale', ''), attrs['output_scale']])))
    (args.dst / 'layer_scales.tsv').write_text('\n'.join(rows) + '\n')
    (args.dst / 'provenance.json').write_text(json.dumps({
        'model': args.model.name, 'model_sha256': hashlib.sha256(args.model.read_bytes()).hexdigest(),
        'scale_convention': 'divisor', 'output_kind': 'quantized float golden, not pipeline',
        'quantization': {p.key: p.value for p in model.metadata_props},
        'cases': len(files), 'input_clipped': clipped, 'input_elements': total,
    }, indent=2) + '\n')
    print('wrote %s (%d files changed)' % (args.dst, changed))
    print('input codes clipped: %d / %d (%.4f%%)' % (clipped, total, 100 * clipped / total))


if __name__ == '__main__':
    main()
