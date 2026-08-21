"""Small helpers shared by every CAM implementation."""

import contextlib

import keras
import numpy as np
import tensorflow as tf

__all__ = ["softmax", "sigmoid"]


def softmax(x, axis=-1):
    """Numerically stable softmax."""
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return (e / np.sum(e, axis=axis, keepdims=True)).astype(np.float32)


def sigmoid(x, a, b, c):
    """Steep sigmoid used to emphasise a heatmap before superimposing it."""
    return c / (1 + np.exp(-a * (np.asarray(x, dtype=np.float64) - b)))


def rescale(cam):
    """ReLU a raw CAM and scale it into [0, 1].

    Every public CAM function funnels through here so that they all satisfy the
    same contract: 2-D float32, in [0, 1], free of NaN/Inf even when the CAM is
    identically zero (which happens for an untrained model, and used to produce
    a division by zero).
    """
    cam = np.maximum(np.asarray(cam, dtype=np.float32), 0.0)
    peak = float(np.max(cam))
    if peak > 0.0:
        cam = cam / peak
    return cam


def resolve_class(preds, class_index):
    """Pick the class to explain: an explicit index, or the model's top-1."""
    if class_index is None:
        return int(np.argmax(np.asarray(preds)[0]))
    return int(class_index)


def submodel(model, layer_name):
    """A model returning ``(activations of layer_name, final output)``."""
    layer_output = model.get_layer(layer_name).output
    return keras.Model(model.inputs, [layer_output, model.outputs[0]])


def model_input_hw(model):
    """Spatial size ``(height, width)`` the model expects."""
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    return int(shape[1]), int(shape[2])


@contextlib.contextmanager
def logit_output(model, enabled=True):
    """Temporarily strip a final softmax so gradients see the raw class score.

    Grad-CAM (eq. 1), Grad-CAM++ and Score-CAM are all defined on the score
    *before* the softmax. Keras application models such as ``VGG16`` end in a
    softmax, so we swap the last activation for a linear one and restore it on
    the way out. The model is left exactly as we found it.
    """
    layer = model.layers[-1]
    original = getattr(layer, "activation", None)
    if not enabled or original is None or original is keras.activations.linear:
        yield
        return
    layer.activation = keras.activations.linear
    try:
        yield
    finally:
        layer.activation = original


def as_tensor(img_array):
    return tf.convert_to_tensor(np.asarray(img_array, dtype=np.float32))
