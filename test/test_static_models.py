import os
import unittest

import toml
import torch

from bonito.ctc import MODEL_CLASSES, create_model


MODEL_CASES = {
    'default': ('default', 'weights_4.tar'),
    'TINYX0111': ('TINYX0111', 'weights_1.tar'),
    'TINYX011': ('TINYX011', 'weights_1.tar'),
    'TINYX01': ('TINYX01', 'weights_1.tar'),
    'TINYX2': ('TINYX2', 'weights_1.tar'),
    'TINYX3': ('TINYX3', 'weights_1.tar'),
}


class StaticModelTest(unittest.TestCase):

    def test_factory_constructs_models_without_config(self):
        for model_name, model_class in MODEL_CLASSES.items():
            with self.subTest(model=model_name):
                model = create_model(model_name)
                self.assertIsInstance(model, model_class)
                self.assertIsNone(model.config)

    def test_static_models_match_config_models(self):
        models_root = os.path.join(os.path.dirname(__file__), '..', 'bonito', 'models')

        for model_name, (directory_name, weights_name) in MODEL_CASES.items():
            with self.subTest(model=model_name):
                model_dir = os.path.join(models_root, directory_name)
                config = toml.load(os.path.join(model_dir, 'config.toml'))
                model_class = MODEL_CLASSES[model_name]

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


if __name__ == '__main__':
    unittest.main()
