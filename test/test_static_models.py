import os
import unittest
from unittest.mock import patch

import torch

from bonito.ctc.basecall import compute_scores, ctc_decode
from bonito.util import load_model as load_application_model
from model_inference import MODEL_CLASSES as STANDALONE_MODEL_CLASSES


class StaticModelTest(unittest.TestCase):

    def test_application_loads_standalone_model_without_toml(self):
        models_root = os.path.join(os.path.dirname(__file__), '..', 'bonito', 'models')
        model_dir = os.path.join(models_root, 'TINYX3')

        with patch(
            'toml.load',
            side_effect=AssertionError('application must not read model TOML'),
        ):
            model = load_application_model(
                model_dir,
                'cpu',
                modeltype='tinynoskipx3',
                weights=1,
                chunksize=48,
                overlap=12,
                batchsize=1,
            )

        self.assertIsInstance(model, STANDALONE_MODEL_CLASSES['TINYX3'])
        self.assertEqual(next(model.parameters()).dtype, torch.float32)

        scores = compute_scores(model, torch.randn(1, 1, 48))
        self.assertEqual(scores.shape, (1, 16, 5))
        sequence, path = ctc_decode(
            scores[0],
            model.alphabet,
            model.qscale,
            model.qbias,
            beamsize=1,
            qscores=True,
            return_path=True,
        )
        self.assertGreaterEqual(len(sequence), len(path))

    def test_explicit_models_are_flat_primitive_layers(self):
        for model_name, model_class in STANDALONE_MODEL_CLASSES.items():
            with self.subTest(model=model_name):
                model = model_class()
                self.assertTrue(all(
                    isinstance(layer, (torch.nn.Conv1d, torch.nn.BatchNorm1d))
                    for layer in model.children()
                ))


if __name__ == '__main__':
    unittest.main()
