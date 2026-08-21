"""Score-CAM and Faster-Score-CAM.

Score-CAM: https://arxiv.org/abs/1910.01279

Where the paper and the authors' reference implementation disagree
-----------------------------------------------------------------
The authors' PyTorch code diverges from their own Algorithm 1 in two places.
Both are exposed here as keyword arguments. **The defaults reproduce the
reference implementation**, which is what every published Score-CAM figure --
including the ones in this repository's README -- was produced with. Opt into
the literal paper behaviour explicitly; the second note below explains why it is
not the default.

1. ``mask_target`` -- what the upsampled mask multiplies.
   Algorithm 1 writes ``M_l^k <- s(Up(A_l^k)) o X_0``, the Hadamard product with
   the input *image*. The reference code multiplies the already-normalised
   tensor instead. The difference is the implied baseline: masking raw pixels
   drives suppressed regions to black, masking a mean-subtracted tensor drives
   them to the dataset mean (grey). Pass ``raw_img_array`` and ``preprocess_fn``
   to get the paper behaviour.

2. ``weight_mode`` -- how the channel weights are formed.
   Algorithm 1 takes the softmax **over channels k** of the target-class logit
   differences::

       S_k^c = f^c(M_k) - f^c(X_b)
       alpha_k^c = exp(S_k^c) / sum_k exp(S_k^c)

   The reference code instead takes the softmax over *classes* and reads off the
   target-class probability. ``weight_mode="paper"`` implements the former,
   ``"reference"`` the latter (and reproduces this repository's pre-v0.2 output).

   Note that the baseline term ``f^c(X_b)`` is a constant with respect to ``k``
   and therefore cancels exactly inside the channel-wise softmax, so no baseline
   forward pass is needed in ``"paper"`` mode.

   Be aware that ``"paper"`` is close to degenerate on a confident model.
   Logits span a wide range, so exponentiating them concentrates almost all the
   mass on a couple of channels. Measured on VGG16 / ``block5_conv3`` with
   ``image/hummingbird.jpg``, where the target-class logits of the 512 masked
   inputs run from -0.06 to 21.61::

       weight_mode   largest weight   top-10 share   effective channels exp(H)
       "paper"                67.7 %         99.4 %            2.5 of 512
       "reference"             2.2 %         22.4 %           79.0 of 512

   In other words, Algorithm 1 read literally makes Score-CAM little more than
   "show the two or three best channels", which is presumably why the authors'
   own code does something else. Reproduce the numbers above with
   ``python -m scorecam.diagnostics`` if you want to check this on your model.
"""

import numpy as np

from ._common import (
    activation_model,
    logit_output,
    model_input_hw,
    rescale,
    resolve_class,
    softmax,
)

__all__ = ["ScoreCam"]

_INTER_LINEAR = 1  # cv2.INTER_LINEAR, without importing cv2 at module scope


def _resize(act_map, size_hw):
    import cv2

    height, width = size_hw
    return cv2.resize(act_map, (width, height), interpolation=_INTER_LINEAR)


def _select_channels(act_map_array, max_N):
    """Faster-Score-CAM: keep only the ``max_N`` highest-variance channels."""
    n_channels = act_map_array.shape[3]
    if max_N is None or max_N < 0 or max_N >= n_channels:
        return act_map_array
    stds = np.std(act_map_array[0], axis=(0, 1))
    unsorted = np.argpartition(-stds, max_N)[:max_N]
    indices = unsorted[np.argsort(-stds[unsorted])]
    return act_map_array[:, :, :, indices]


def ScoreCam(
    model,
    img_array,
    layer_name,
    max_N=-1,
    class_index=None,
    batch_size=32,
    weight_mode="reference",
    raw_img_array=None,
    preprocess_fn=None,
):
    """Score-CAM, and Faster-Score-CAM when ``max_N`` is a positive integer.

    Args:
        model: a built Keras model.
        img_array: model-ready input of shape ``(1, H, W, C)``.
        layer_name: convolutional layer to explain.
        max_N: keep only the ``max_N`` highest-variance activation maps.
            ``-1`` (default) keeps all of them, i.e. plain Score-CAM.
        class_index: class to explain. ``None`` uses the model's top-1.
        batch_size: masked inputs are scored in batches of this size. The
            previous implementation built and scored all of them at once, which
            for VGG16's 512 channels meant a single 300 MB array.
        weight_mode: ``"reference"`` (default, the authors' released code) or
            ``"paper"`` (Algorithm 1 as written) -- see the module docstring.
        raw_img_array: the *un-preprocessed* image, same spatial size as
            ``img_array``. Supply it together with ``preprocess_fn`` to mask raw
            pixels as Algorithm 1 specifies.
        preprocess_fn: the model's preprocessing function, applied to each
            masked raw image.

    Returns:
        2-D ``float32`` array scaled into ``[0, 1]``.
    """
    if weight_mode not in ("paper", "reference"):
        raise ValueError(f"weight_mode must be 'paper' or 'reference', got {weight_mode!r}")

    mask_raw = raw_img_array is not None and preprocess_fn is not None
    if not mask_raw and (raw_img_array is not None) != (preprocess_fn is not None):
        raise ValueError(
            "raw_img_array and preprocess_fn must be supplied together to mask "
            "raw pixels; pass neither to mask the preprocessed tensor instead."
        )
    img_array = np.asarray(img_array, dtype=np.float32)
    base = np.asarray(raw_img_array, dtype=np.float32) if mask_raw else img_array

    # Every forward pass below is an eager __call__ rather than .predict().
    # Keras caches the tf.function that .predict() traces, so a model called
    # through it would keep its softmax even inside logit_output() -- which
    # silently made the weights depend on batch_size.
    cls = resolve_class(model(img_array, training=False).numpy(), class_index)

    act_model = activation_model(model, layer_name)
    act_map_array = np.asarray(act_model(img_array, training=False), dtype=np.float32)
    act_map_array = _select_channels(act_map_array, max_N)
    n_channels = act_map_array.shape[3]

    input_hw = model_input_hw(model)

    def masked_input(k):
        # 1. upsample the activation map to the model's input size
        act_map = _resize(act_map_array[0, :, :, k], input_hw)
        # 2. normalise into [0, 1]  --  eq. 8, s(A) = (A - min) / (max - min)
        lo, hi = float(np.min(act_map)), float(np.max(act_map))
        if hi - lo != 0.0:
            act_map = (act_map - lo) / (hi - lo)
        else:
            act_map = np.zeros_like(act_map)
        # 3. project onto the input by a Hadamard product
        masked = base[0] * act_map[..., None]
        return preprocess_fn(masked.copy()) if mask_raw else masked

    # 4. feed the masked inputs through the model, in batches
    with logit_output(model, enabled=(weight_mode == "paper")):
        scores = []
        for start in range(0, n_channels, batch_size):
            batch = np.stack(
                [masked_input(k) for k in range(start, min(start + batch_size, n_channels))]
            )
            scores.append(np.asarray(model(batch, training=False), dtype=np.float32))
        scores = np.concatenate(scores, axis=0)

    # 5. turn the target-class scores into channel weights
    if weight_mode == "paper":
        # softmax over the channel axis of the target-class logits. The baseline
        # term f^c(X_b) is constant in k and cancels here.
        weights = softmax(scores[:, cls], axis=0)
    else:
        weights = softmax(scores, axis=1)[:, cls]

    # 6. ReLU over the linear combination of activation maps
    return rescale(np.dot(act_map_array[0], weights))
