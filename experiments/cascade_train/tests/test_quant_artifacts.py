import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quant_reference import in_scale, out_scale, quantize, read_annotations, run_integer, stripped

ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / 'onnx_models/tinyx3_ivfs_q8.onnx'
FLOAT = ROOT / 'test_vectors/tinyx3_ivfs_folded'
INT8 = ROOT / 'test_vectors/tinyx3_ivfs_folded_int8'


def test_annotations_and_evaluation_are_bound_to_the_current_model():
    model, nodes, _ = read_annotations(MODEL)
    onnx.checker.check_model(stripped(model))
    assert all('exp_scale' not in attrs for _, attrs in nodes)
    metadata = {p.key: p.value for p in model.metadata_props}
    assert metadata['quantization.calibration_vectors'] == '000-039'
    for path in [INT8 / 'provenance.json', INT8 / 'layer_ref/provenance.json',
                 ROOT / 'experiments/cascade_train/evaluation_q8.json']:
        assert json.loads(path.read_text())['model_sha256'] == hashlib.sha256(MODEL.read_bytes()).hexdigest()
    evaluation = json.loads((ROOT / 'experiments/cascade_train/evaluation_q8.json').read_text())
    assert [r['index'] for r in evaluation['vectors']] == list(range(40, 50))


def test_all_boundary_vectors_follow_divisor_scales_and_symmetric_range():
    _, nodes, _ = read_annotations(MODEL)
    si, so = in_scale(nodes[0][1]), out_scale(nodes[-1][1])
    for i in range(50):
        for name, scale in [('input', si), ('output', so)]:
            filename = '%s_%03d.npy' % (name, i)
            actual = np.load(INT8 / filename)
            np.testing.assert_array_equal(actual, quantize(np.load(FLOAT / filename), scale))
            assert actual.dtype == np.int8 and actual.min() >= -127


def test_layer_dump_matches_the_pipeline_and_has_no_negative_relu():
    torch.set_num_threads(1)
    _, nodes, initializers = read_annotations(MODEL)
    xq = np.load(INT8 / 'input_000.npy')
    _, codes, acc = run_integer(nodes, initializers, xq)
    with np.load(INT8 / 'layer_ref/all_layers.i8.npz') as stored:
        assert len(stored.files) == len(codes) == 152
        for node, attrs in nodes:
            if node.op_type == 'Constant':
                continue
            name = node.name.replace('/', '_').lstrip('_')
            np.testing.assert_array_equal(stored[name], codes[node.name])
            if node.op_type == 'Relu':
                assert stored[name].min() >= 0
    np.testing.assert_array_equal(np.load(INT8 / 'layer_ref/conv0_Conv.acc.i32.npy'), acc['/conv0/Conv'])
