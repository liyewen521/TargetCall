from .model import Model
from .basecall import basecall
from .modeltinynoskipx4 import ModelTinyNoSkipX4
from .modeltinynoskipx1 import ModelTinyNoSkipX1
from .modeltinynoskipx0111 import ModelTinyNoSkipX0111
from .modeltinynoskipx011 import ModelTinyNoSkipX011
from .modeltinynoskipx01 import ModelTinyNoSkipX01
from .modeltinynoskipx2 import ModelTinyNoSkipX2
from .modeltinynoskipx3 import ModelTinyNoSkipX3

from .modelb1 import ModelB1
from .modelb1b2 import ModelB1B2
from .modelb1x2 import ModelB1X2


DefaultModel = Model
TinyX1Model = ModelTinyNoSkipX1
TinyX0111Model = ModelTinyNoSkipX0111
TinyX011Model = ModelTinyNoSkipX011
TinyX01Model = ModelTinyNoSkipX01
TinyX2Model = ModelTinyNoSkipX2
TinyX3Model = ModelTinyNoSkipX3
TinyX4Model = ModelTinyNoSkipX4


MODEL_CLASSES = {
    'default': DefaultModel,
    'TINYX1': TinyX1Model,
    'TINYX0111': TinyX0111Model,
    'TINYX011': TinyX011Model,
    'TINYX01': TinyX01Model,
    'TINYX2': TinyX2Model,
    'TINYX3': TinyX3Model,
    'TINYX4': TinyX4Model,
}


def create_model(name='default'):
    """Create a built-in CTC model without reading a TOML configuration."""
    try:
        model_class = MODEL_CLASSES[name]
    except KeyError as error:
        available = ', '.join(MODEL_CLASSES)
        raise ValueError(f"unknown model '{name}'; choose from: {available}") from error
    return model_class()
