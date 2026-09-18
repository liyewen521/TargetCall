import os
import unittest
from unittest.mock import patch

import toml
import torch

from bonito.ctc.basecall import compute_scores, ctc_decode
from bonito.ctc.model import Model as DefaultLegacyModel
from bonito.ctc.modeltinynoskipx0111 import ModelTinyNoSkipX0111
from bonito.ctc.modeltinynoskipx011 import ModelTinyNoSkipX011
from bonito.ctc.modeltinynoskipx01 import ModelTinyNoSkipX01
from bonito.ctc.modeltinynoskipx2 import ModelTinyNoSkipX2
from bonito.ctc.modeltinynoskipx3 import ModelTinyNoSkipX3
from bonito.util import load_model as load_application_model
from model import MODEL_CLASSES as STANDALONE_MODEL_CLASSES
from model import load_weights as load_standalone_weights


MODEL_CASES = {
    'default': ('default', 'weights_4.tar'),
    'TINYX0111': ('TINYX0111', 'weights_1.tar'),
    'TINYX011': ('TINYX011', 'weights_1.tar'),
    'TINYX01': ('TINYX01', 'weights_1.tar'),
    'TINYX2': ('TINYX2', 'weights_1.tar'),
    'TINYX3': ('TINYX3', 'weights_1.tar'),
}

LEGACY_MODEL_CLASSES = {
    'default': DefaultLegacyModel,
    'TINYX0111': ModelTinyNoSkipX0111,
    'TINYX011': ModelTinyNoSkipX011,
    'TINYX01': ModelTinyNoSkipX01,
    'TINYX2': ModelTinyNoSkipX2,
    'TINYX3': ModelTinyNoSkipX3,
}


class StaticModelTest(unittest.TestCase):

    def test_application_loads_standalone_model_without_toml(self):
        models_root = os.path.join(os.path.dirname(__file__), '..', 'bonito', 'models')
        model_dir = os.path.join(models_root, 'TINYX3')

        with patch(
            'bonito.util.toml.load',
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

    def test_static_models_match_config_models(self):
        models_root = os.path.join(os.path.dirname(__file__), '..', 'bonito', 'models')

        for model_name, (directory_name, weights_name) in MODEL_CASES.items():
            with self.subTest(model=model_name):
                model_dir = os.path.join(models_root, directory_name)
                config = toml.load(os.path.join(model_dir, 'config.toml'))
                model_class = LEGACY_MODEL_CLASSES[model_name]

                static_model = model_class()
                config_model = model_class(config)
                self.assertEqual(
                    list(static_model.state_dict()),
                    list(config_model.state_dict()),
                )
                self.assertEqual(
                    [value.shape for value in static_model.state_dict().values()],
                    [value.shape for value in config_model.state_dict().values()],
                )

                weights = torch.load(
                    os.path.join(model_dir, weights_name),
                    map_location='cpu',
                )
                static_model.load_state_dict(weights, strict=True)
                config_model.load_state_dict(weights, strict=True)
                static_model.eval()
                config_model.eval()

                sample = torch.randn(1, 1, 48)
                with torch.no_grad():
                    static_output = static_model(sample)
                    config_output = config_model(sample)
                torch.testing.assert_close(
                    static_output,
                    config_output,
                    rtol=0,
                    atol=0,
                )

    def test_standalone_models_match_legacy_models(self):
        models_root = os.path.join(os.path.dirname(__file__), '..', 'bonito', 'models')

        for model_name, (directory_name, weights_name) in MODEL_CASES.items():
            with self.subTest(model=model_name):
                model_dir = os.path.join(models_root, directory_name)
                config = toml.load(os.path.join(model_dir, 'config.toml'))
                weights_path = os.path.join(model_dir, weights_name)

                legacy_model = LEGACY_MODEL_CLASSES[model_name](config)
                legacy_model.load_state_dict(
                    torch.load(weights_path, map_location='cpu'),
                    strict=True,
                )
                standalone_model = STANDALONE_MODEL_CLASSES[model_name]()
                load_standalone_weights(standalone_model, weights_path)
                legacy_model.eval()
                standalone_model.eval()

                self.assertTrue(all(
                    isinstance(layer, (torch.nn.Conv1d, torch.nn.BatchNorm1d))
                    for layer in standalone_model.children()
                ))

                self.assertEqual(
                    sum(parameter.numel() for parameter in legacy_model.parameters()),
                    sum(parameter.numel() for parameter in standalone_model.parameters()),
                )

                sample = torch.randn(1, 1, 48)
                with torch.no_grad():
                    legacy_output = legacy_model(sample)
                    standalone_output = standalone_model(sample)
                torch.testing.assert_close(
                    standalone_output,
                    legacy_output,
                    rtol=0,
                    atol=0,
                )


if __name__ == '__main__':
    unittest.main()
