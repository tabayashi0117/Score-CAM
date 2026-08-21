"""Image loading helpers."""

import numpy as np
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from keras.utils import img_to_array, load_img

__all__ = ["read_img", "read_and_preprocess_img"]


def read_img(path, size=(224, 224)):
    """Load an image as a raw ``(1, H, W, 3)`` float array in ``[0, 255]``.

    This is what Score-CAM's Hadamard product is defined on; pair it with
    ``preprocess_fn`` when calling :func:`scorecam.ScoreCam`.
    """
    img = load_img(path, target_size=size)
    return np.expand_dims(img_to_array(img), axis=0)


def read_and_preprocess_img(path, size=(224, 224), preprocess_fn=None):
    """Load an image and apply a model's preprocessing.

    ``preprocess_fn`` defaults to VGG16's, which is what this function always
    used before v0.2. Pass the preprocessing that matches *your* model -- the
    ResNet example in the notebook was silently using VGG16's.
    """
    if preprocess_fn is None:
        preprocess_fn = vgg16_preprocess_input
    return preprocess_fn(read_img(path, size=size))
