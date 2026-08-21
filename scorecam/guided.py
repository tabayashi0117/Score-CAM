"""Guided Backpropagation.

The pre-v0.2 implementation registered a gradient named ``GuidedBackProp`` on
``tensorflow.python.framework.ops._gradient_registry`` and relied on
``tf.compat.v1.get_default_graph().gradient_override_map``. Both are TF1 graph
APIs; the registry was private and neither survives in Keras 3. The same effect
is achieved here with a public ``tf.custom_gradient`` activation that is swapped
into a clone of the model.
"""

import keras
import numpy as np
import tensorflow as tf

from ._common import activation_model, as_tensor, call_model

__all__ = ["build_guided_model", "GuidedBackPropagation"]


@tf.custom_gradient
def _guided_relu(x):
    """ReLU whose backward pass also discards negative incoming gradients."""

    def grad(dy):
        return dy * tf.cast(dy > 0.0, dy.dtype) * tf.cast(x > 0.0, x.dtype)

    return tf.nn.relu(x), grad


def _is_relu(activation):
    return activation is keras.activations.relu


def _clone_layer(layer):
    config = layer.get_config()

    if isinstance(layer, keras.layers.ReLU):
        return keras.layers.Activation(_guided_relu, name=config["name"])

    if config.get("activation") == "relu":
        config["activation"] = _guided_relu
        return layer.__class__.from_config(config)

    if isinstance(layer, keras.layers.Activation) and _is_relu(layer.activation):
        return keras.layers.Activation(_guided_relu, name=config["name"])

    return layer.__class__.from_config(config)


def build_guided_model(model_or_builder):
    """Return a copy of the model with every ReLU replaced by a guided ReLU.

    Accepts either a built model or, for backwards compatibility with the
    pre-v0.2 API, a zero-argument callable that builds one.
    """
    model = model_or_builder() if callable(model_or_builder) and not isinstance(
        model_or_builder, keras.Model
    ) else model_or_builder

    guided = keras.models.clone_model(model, clone_function=_clone_layer)
    guided.set_weights(model.get_weights())
    return guided


def GuidedBackPropagation(model, img_array, layer_name):
    """Saliency of the input w.r.t. the strongest activation of ``layer_name``.

    ``model`` should be the output of :func:`build_guided_model`; passing a plain
    model computes ordinary (non-guided) backpropagation.
    """
    act_model = activation_model(model, layer_name)
    x = as_tensor(img_array)

    with tf.GradientTape() as tape:
        tape.watch(x)
        layer_output = call_model(act_model, x)
        max_output = tf.reduce_max(layer_output, axis=3)

    grads = tape.gradient(max_output, x)
    if grads is None:
        return np.zeros(np.shape(img_array)[1:], dtype=np.float32)
    return np.clip(grads[0].numpy(), 0.0, 1.0).astype(np.float32)
