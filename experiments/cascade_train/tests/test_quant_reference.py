import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quant_reference import conv_integer, quantize, run_integer
from compare_layers import compare, load
from check_quant import edit_identity


def conv_attrs(si=0.5, sw=0.25, so=0.125, bias_bits=8):
    return dict(input_scale=si, weight_ch_scales=[sw], output_scale=so,
                input_bitdepth=8, weight_bitdepth=8, output_bitdepth=8,
                bias_bitdepth=bias_bits, strides=[1, 1], pads=[0, 0, 0, 0])


def test_symmetric_saturation_divisor_and_ties_to_even():
    values = np.array([-1000, -1.25, -0.75, 0.75, 1.25, 1000])
    np.testing.assert_array_equal(quantize(values, 0.5), [-127, -2, -2, 2, 2, 127])
    assert quantize(values, 0.5).dtype == np.int8


def test_integer_bias_saturates_before_accumulation():
    node = helper.make_node('Conv', ['x', 'w', 'b'], ['out'], name='conv')
    initializers = {'w': np.full((1, 1, 1, 1), 0.5, np.float32), 'b': np.array([100.0], np.float32)}
    codes, accumulator = conv_integer(node, conv_attrs(), initializers, np.ones((1, 1, 1, 1), np.int8))
    # wq=2, bq=clip(100/(.5*.25))=127, accumulator=1*2+127.
    assert accumulator.item() == 129
    assert accumulator.dtype == np.int32
    assert codes.item() == 127


def test_fractional_bias_is_rounded_before_output_requantization():
    node = helper.make_node('Conv', ['x', 'w', 'b'], ['out'], name='conv')
    initializers = {'w': np.ones((1, 1, 1, 1), np.float32), 'b': np.array([0.6], np.float32)}
    codes, acc = conv_integer(node, conv_attrs(si=1, sw=1, so=0.5), initializers, np.zeros((1, 1, 1, 1), np.int8))
    assert acc.item() == 1
    assert codes.item() == 2  # Adding float bias instead would yield round(1.2)=1.


def test_next_layer_consumes_the_same_saturated_codes_as_the_dump():
    nodes = [
        (helper.make_node('Conv', ['input', 'w1'], ['a'], name='conv1'), conv_attrs(si=1, sw=1, so=1)),
        (helper.make_node('Relu', ['a'], ['b'], name='relu'), dict(output_scale=1, output_bitdepth=8)),
        (helper.make_node('Conv', ['b', 'w2'], ['output'], name='conv2'), conv_attrs(si=1, sw=1, so=1)),
    ]
    ini = {'w1': np.full((1, 1, 1, 1), 2, np.float32), 'w2': np.ones((1, 1, 1, 1), np.float32)}
    out, codes, acc = run_integer(nodes, ini, np.full((1, 1, 1, 1), 100, np.int8))
    assert acc['conv1'].item() == 200
    assert codes['conv1'].item() == codes['relu'].item() == acc['conv2'].item() == out.item() == 127


def test_comparator_rejects_missing_wrong_and_wrong_shape_layers():
    ref = {'first': np.array([1, 2], np.int8), 'second': np.array([3], np.int8)}
    assert compare(ref, ref) == 0
    assert compare(ref, {'first': ref['first']}) == 2
    assert compare(ref, {'first': np.array([0, 2], np.int8), 'second': ref['second']}) == 1
    assert compare(ref, {'first': ref['first'][None], 'second': ref['second']}) == 2
    assert compare(ref, {'first': ref['first']}, allow_partial=True) == 0


def test_comparator_rejects_duplicate_normalized_names(tmp_path):
    path = tmp_path / 'duplicate.npz'
    np.savez(path, **{'/a/b': np.array([1]), 'a_b': np.array([1])})
    with pytest.raises(ValueError, match='duplicate'):
        load(path)


def test_empty_decodes_are_well_defined():
    assert edit_identity([], []) == 1
    assert edit_identity([], [1]) == 0



def test_cmodel_instruction_fields_accept_grouped_decimal(tmp_path):
    from replay_cmodel import instruction_fields

    path = tmp_path / "softmax.inst"
    path.write_text(
        "| | sm_matrix_y | 24 |  9 | [ 1,333, 0x535] | |\n"
        "| | sm_matrix_x | 40 | 25 | [     4,   0x4] | |\n"
        "| | sm_shift2   | 53 | 49 | [     0,     0] | |\n"
    )
    assert instruction_fields(path) == {"sm_matrix_y": 1333, "sm_matrix_x": 4, "sm_shift2": 0}


def test_cmodel_softmax_replay_pads_transposed_logits_and_uses_instruction_scales():
    from replay_cmodel import replay_softmax

    logits = np.arange(30, dtype=np.int8).reshape(1, 5, 1, 6).transpose(0, 2, 3, 1)
    fields = dict(sm_matrix_y=5, sm_matrix_x=4, tile_mode=0, spu_op=0, sm_mode=0,
                  sm_shift1=4, sm_shift2=2, sm_shift0=7)

    def kernel(padded, rows, cols, input_scale, exp_scale, output_scale, *, tile_mode, sm_mode):
        assert padded.is_contiguous() and padded.shape == (1, 1, 16, 16)
        np.testing.assert_array_equal(padded[:, :, :rows, :cols].numpy(), logits)
        assert (input_scale.item(), exp_scale.item(), output_scale.item()) == (1/16, 1/4, 1/128)
        assert (tile_mode, sm_mode) == (0, 0)
        return padded.view(-1).tolist(), None, None

    np.testing.assert_array_equal(replay_softmax(logits, fields, kernel), logits)
