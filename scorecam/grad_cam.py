"""Grad-CAM and Grad-CAM++.

Grad-CAM:   https://arxiv.org/abs/1610.02391
Grad-CAM++: https://arxiv.org/abs/1710.11063
"""

import numpy as np
import tensorflow as tf

from ._common import as_tensor, logit_output, rescale, resolve_class, submodel

__all__ = ["GradCam", "GradCamPlusPlus"]


def GradCam(model, img_array, layer_name, class_index=None, use_logits=True):
    """Grad-CAM.

    Args:
        model: a built Keras model.
        img_array: model-ready input of shape ``(1, H, W, C)``.
        layer_name: convolutional layer to explain.
        class_index: class to explain. ``None`` uses the model's top-1.
        use_logits: differentiate the pre-softmax score, as in eq. 1 of the
            paper. Set ``False`` to differentiate the softmax probability
            instead (what this repository did before v0.2).

    Returns:
        2-D ``float32`` array scaled into ``[0, 1]``.
    """
    grad_model = submodel(model, layer_name)
    x = as_tensor(img_array)

    with logit_output(model, use_logits):
        with tf.GradientTape() as tape:
            conv_output, preds = grad_model(x, training=False)
            cls = resolve_class(preds.numpy(), class_index)
            y_c = preds[:, cls]
        grads = tape.gradient(y_c, conv_output)

    output = conv_output[0].numpy()
    grads_val = grads[0].numpy()

    # eq. 2: neuron importance = global-average-pooled gradient
    weights = np.mean(grads_val, axis=(0, 1))

    # eq. 3: ReLU over the weighted combination of feature maps
    return rescale(np.dot(output, weights))


def GradCamPlusPlus(model, img_array, layer_name, class_index=None, use_logits=True):
    """Grad-CAM++ (eq. 10, 11 and 19 of the paper).

    The paper defines its derivatives on ``Y^c = exp(S^c)`` where ``S^c`` is the
    penultimate, pre-softmax score. That substitution is what makes the closed
    form of eq. 19 valid::

        d^n Y^c / dA^n = exp(S^c) * (dS^c / dA)^n

    so we differentiate ``S^c`` once and raise it to the required power, exactly
    as the paper prescribes. Passing a post-softmax probability in as ``S^c``
    (the pre-v0.2 behaviour, still available via ``use_logits=False``) breaks
    that identity, because a probability is bounded in ``[0, 1]`` and ``exp`` of
    it barely varies.
    """
    grad_model = submodel(model, layer_name)
    x = as_tensor(img_array)

    with logit_output(model, use_logits):
        with tf.GradientTape() as tape:
            conv_output, preds = grad_model(x, training=False)
            cls = resolve_class(preds.numpy(), class_index)
            y_c = preds[:, cls]
        grads = tape.gradient(y_c, conv_output)

    score = float(y_c[0].numpy())
    conv_output = conv_output[0].numpy()
    grads_val = grads[0].numpy()

    # exp(S^c) is a positive constant that cancels in the alpha ratio below; we
    # keep it only in the first-order term, and clip its exponent so that a
    # large logit cannot overflow to inf.
    exp_score = float(np.exp(np.clip(score, -60.0, 60.0)))
    first = exp_score * grads_val
    second = exp_score * grads_val**2
    third = exp_score * grads_val**3

    # eq. 10: alpha_ij^kc
    global_sum = np.sum(conv_output.reshape((-1, conv_output.shape[2])), axis=0)
    alpha_num = second
    alpha_denom = second * 2.0 + third * global_sum.reshape((1, 1, -1))
    alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, 1.0)
    alphas = alpha_num / alpha_denom

    # Spatial re-normalisation of the alphas. Not part of eq. 10, but present in
    # the authors' released code and in every port of it; kept for comparability.
    alpha_norm = np.sum(alphas, axis=(0, 1)).reshape((1, 1, -1))
    alphas = np.divide(alphas, alpha_norm, out=np.zeros_like(alphas), where=alpha_norm != 0.0)

    # eq. 11: w_k^c = sum_ij alpha_ij^kc * relu(dY^c / dA_ij^k)
    weights = np.maximum(first, 0.0)
    deep_linearization_weights = np.sum(
        (weights * alphas).reshape((-1, conv_output.shape[2])), axis=0
    )

    # eq. 20
    return rescale(np.sum(deep_linearization_weights * conv_output, axis=2))
