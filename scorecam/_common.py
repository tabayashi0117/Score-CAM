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


def model_inputs(model):
    """The model's input spec, unwrapped when there is exactly one.

    ``model.inputs`` is always a list. Building a sub-model from it makes that
    sub-model expect a list, so calling it with a bare array makes Keras 3 warn
    ("The structure of `inputs` doesn't match the expected structure") and fall
    back. Unwrapping keeps the sub-model's signature the same as the original's.
    """
    inputs = model.inputs
    if isinstance(inputs, (list, tuple)) and len(inputs) == 1:
        return inputs[0]
    return inputs


def call_model(model, x, training=False):
    """Call a model with a bare array, matching how its inputs are structured.

    ``Model(inputs, outputs)`` remembers whether ``inputs`` was a tensor or a
    list of one, and Keras 3 warns and falls back when a call does not match.
    The two are easy to mix up: ``ResNet50(input_tensor=Input(...)).input``
    returns a *list*, so the natural ``Model(backbone.input, head)`` produces a
    list-structured model, while ``VGG16()`` and ``Sequential`` produce
    tensor-structured ones. Callers should not have to know which they have.
    """
    try:
        expects_list = isinstance(model.input, (list, tuple))
    except (AttributeError, ValueError):
        expects_list = False
    return model([x] if expects_list else x, training=training)


def submodel(model, layer_name):
    """A model returning ``(activations of layer_name, final output)``."""
    layer_output = model.get_layer(layer_name).output
    return keras.Model(model_inputs(model), [layer_output, model.outputs[0]])


def activation_model(model, layer_name):
    """A model returning just the activations of ``layer_name``."""
    return keras.Model(model_inputs(model), model.get_layer(layer_name).output)


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
