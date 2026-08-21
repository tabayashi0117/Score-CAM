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

1. What the upsampled mask multiplies (``raw_img_array`` / ``preprocess_fn``).
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
   target-class probability, i.e. ``alpha_k^c = P(c | M_k)``.
   ``weight_mode="paper"`` implements the former, ``"reference"`` the latter.

   Both modes are computed from logits. A Keras classifier usually *ends* in a
   softmax, so the pre-v0.2 code -- which applied its own softmax on top of the
   model's output -- was applying it twice, flattening the weights towards
   uniform. Torch models emit logits, which is why the reference code gets this
   right with a single softmax.

   Note that the baseline term ``f^c(X_b)`` is a constant with respect to ``k``
   and therefore cancels exactly inside the channel-wise softmax, so no baseline
   forward pass is needed in ``"paper"`` mode.

   Be aware that ``"paper"`` is close to degenerate on a confident model.
   Logits span a wide range, so exponentiating them concentrates almost all the
   mass on a couple of channels. Measured on VGG16 / ``block5_conv3`` with
   ``image/hummingbird.jpg``, where the target-class logits of the 512 masked
   inputs run from 1.29 to 20.54::

       weight_mode   largest weight   top-10 share   effective channels exp(H)
       "paper"                82.9 %         96.4 %            2.5 of 512
       "reference"             1.4 %         13.5 %          159.6 of 512

   In other words, Algorithm 1 read literally makes Score-CAM little more than
   "show the two or three best channels", which is presumably why the authors'
   own code does something else. Reproduce the numbers above with
   ``python -m scorecam.diagnostics`` (add ``--paper-mask`` to measure under
   raw-pixel masking instead; the conclusion is the same, 2.5 either way).

   For reference, the pre-v0.2 double softmax landed at 487.3 effective
   channels -- so close to a plain unweighted mean of the activation maps that
   it was barely Score-CAM at all.
"""

import numpy as np

from ._common import (
    activation_model,
    call_model,
    logit_output,
    model_input_hw,
    rescale,
    resize_2d,
    resolve_class,
    softmax,
)

__all__ = ["ScoreCam", "masked_inputs"]


def upsample_and_normalise(act_maps, size_hw):
    """``(h, w, n)`` activation maps -> ``(n, H, W)`` masks in ``[0, 1]``.

    Steps 1 and 2 of the algorithm, vectorised over channels. A constant map
    normalises to zero instead of dividing by a zero range.
    """
    # (h, w, n) is already the (H, W, C) layout tf.image.resize expects, so the
    # spatial dims are the ones that get resized; move the channel axis after.
    maps = np.transpose(resize_2d(act_maps, size_hw), (2, 0, 1))
    lo = maps.min(axis=(1, 2), keepdims=True)
    span = maps.max(axis=(1, 2), keepdims=True) - lo
    # eq. 8: s(A) = (A - min) / (max - min)
    return np.where(span > 0.0, (maps - lo) / np.where(span > 0.0, span, 1.0), 0.0)


def masked_inputs(base_img, act_maps, size_hw, preprocess_fn=None):
    """Step 3: project the masks onto the input by a Hadamard product.

    Shared with :mod:`scorecam.diagnostics` so the two cannot drift apart.
    """
    masks = upsample_and_normalise(act_maps, size_hw)
    batch = base_img[None] * masks[..., None]
    return preprocess_fn(batch) if preprocess_fn is not None else batch


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
    cls = resolve_class(call_model(model, img_array).numpy(), class_index)

    act_model = activation_model(model, layer_name)
    act_map_array = np.asarray(call_model(act_model, img_array), dtype=np.float32)
    act_map_array = _select_channels(act_map_array, max_N)
    n_channels = act_map_array.shape[3]

    input_hw = model_input_hw(model)

    # 1-4. upsample, normalise, mask, and score, one batch of channels at a time
    with logit_output(model):
        logits = []
        for start in range(0, n_channels, batch_size):
            end = min(start + batch_size, n_channels)
            batch = masked_inputs(
                base[0],
                act_map_array[0, :, :, start:end],
                input_hw,
                preprocess_fn if mask_raw else None,
            )
            logits.append(np.asarray(call_model(model, batch), dtype=np.float32))
        logits = np.concatenate(logits, axis=0)

    # 5. turn the target-class scores into channel weights
    if weight_mode == "paper":
        # softmax over the channel axis. The baseline term f^c(X_b) is constant
        # in k and cancels here, so it needs no forward pass of its own.
        weights = softmax(logits[:, cls], axis=0)
    else:
        # softmax over the class axis: the target-class probability P(c | M_k).
        weights = softmax(logits, axis=1)[:, cls]

    # 6. ReLU over the linear combination of activation maps
    return rescale(np.dot(act_map_array[0], weights))
