import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quant_reference import in_scale, quantize, read_annotations, run_integer, stripped

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


def test_all_inputs_follow_divisor_scale_and_symmetric_range():
    _, nodes, _ = read_annotations(MODEL)
    scale = in_scale(nodes[0][1])
    for i in range(50):
        filename = 'input_%03d.npy' % i
        actual = np.load(INT8 / filename)
        np.testing.assert_array_equal(actual, quantize(np.load(FLOAT / filename), scale))
        assert actual.dtype == np.int8 and actual.min() >= -127


def test_cmodel_output_vectors_match_provenance():
    provenance = json.loads((INT8 / 'provenance.json').read_text())
    assert provenance['output_kind'] == 'cmodel_replay'
    cmodel = provenance['cmodel']
    assert cmodel['softmax_output_shift'] == 7
    assert [case['seed'] for case in cmodel['cases']] == list(range(50))
    for case in cmodel['cases']:
        tag = '%03d' % case['seed']
        input_path = INT8 / ('input_%s.npy' % tag)
        output_path = INT8 / ('output_%s.npy' % tag)
        assert hashlib.sha256(input_path.read_bytes()).hexdigest() == case['input_sha256']
        assert hashlib.sha256(output_path.read_bytes()).hexdigest() == case['output_sha256']
        output = np.load(output_path, allow_pickle=False)
        assert output.shape == (1, 1, 1334, 5) and output.dtype == np.int8 and output.min() >= 0


def test_float_boundary_generator_preserves_cmodel_destination(tmp_path, monkeypatch):
    from make_test_vectors import main

    manifest = tmp_path / 'provenance.json'
    metadata = (INT8 / 'provenance.json').read_bytes()
    manifest.write_bytes(metadata)
    monkeypatch.setattr(sys, 'argv', ['make_test_vectors.py', '--dst', str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert list(tmp_path.iterdir()) == [manifest]
    assert manifest.read_bytes() == metadata


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


def test_compiler_copy_converts_divisors_and_preserves_graph_parameters():
    from export_cmodel_onnx import SCALE_ATTRIBUTES, compiler_model

    original, _, _ = read_annotations(MODEL)
    converted = compiler_model(MODEL)
    metadata = {p.key: p.value for p in converted.metadata_props}
    assert metadata["quantization.scale_convention"] == "multiplier"
    assert original.graph.initializer == converted.graph.initializer
    for old, new in zip(original.graph.node, converted.graph.node):
        for source, target in zip(old.attribute, new.attribute):
            assert source.name == target.name
            if source.name in SCALE_ATTRIBUTES:
                values = onnx.helper.get_attribute_value(source)
                expected = np.asarray(1.0 / np.asarray(values, dtype=np.float64), dtype=np.float32)
                np.testing.assert_array_equal(onnx.helper.get_attribute_value(target), expected)
            else:
                assert source == target
