import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import keras  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

INPUT_SHAPE = (32, 32, 3)
LAYER_NAME = "target_conv"
N_CLASSES = 4
N_FILTERS = 8


def _build(seed, activation="softmax"):
    keras.utils.set_random_seed(seed)
    return keras.Sequential(
        [
            keras.layers.Input(INPUT_SHAPE),
            keras.layers.Conv2D(4, 3, activation="relu", padding="same", name="conv1"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(N_FILTERS, 3, activation="relu", padding="same", name=LAYER_NAME),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(N_CLASSES, activation=activation, name="predictions"),
        ],
        name="tiny_cnn",
    )


@pytest.fixture(scope="module")
def model():
    """A tiny randomly-initialised CNN. No downloads, no ImageNet weights."""
    return _build(seed=0)


@pytest.fixture(scope="module")
def img_array():
    rng = np.random.default_rng(0)
    return rng.uniform(0.0, 255.0, size=(1, *INPUT_SHAPE)).astype("float32")


@pytest.fixture
def build_model():
    return _build


@pytest.fixture(scope="module")
def list_input_model():
    """A model whose inputs were declared as a list of one.

    `ResNet50(input_tensor=Input(...)).input` returns a list, so the natural
    `Model(backbone.input, head)` in the DAGM example builds one of these. Keras
    3 warns and falls back if such a model is called with a bare array.
    """
    keras.utils.set_random_seed(2)
    inputs = keras.layers.Input(INPUT_SHAPE)
    x = keras.layers.Conv2D(4, 3, activation="relu", padding="same", name="conv1")(inputs)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(N_FILTERS, 3, activation="relu", padding="same",
                            name=LAYER_NAME)(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(N_CLASSES, activation="softmax", name="predictions")(x)
    return keras.Model([inputs], outputs)
