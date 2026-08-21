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
