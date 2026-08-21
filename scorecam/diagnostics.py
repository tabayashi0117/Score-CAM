"""Measure how concentrated Score-CAM's channel weights are.

``weight_mode="paper"`` exponentiates raw logits, which on a confident model
puts nearly all the weight on a handful of channels. This module quantifies
that, so the claim in ``score_cam.py`` can be re-checked on any model::

    python -m scorecam.diagnostics
    python -m scorecam.diagnostics --image ./image/collies.JPG --layer block5_conv3
"""

import argparse

import numpy as np

from ._common import (
    activation_model,
    logit_output,
    model_input_hw,
    resolve_class,
    softmax,
)
from .score_cam import _resize

__all__ = ["weight_concentration"]


def _effective_channels(weights):
    """``exp(H)`` of the weight distribution: how many channels really vote."""
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    nonzero = w[w > 0]
    return float(np.exp(-np.sum(nonzero * np.log(nonzero))))


def weight_concentration(model, img_array, layer_name, raw_img_array=None,
                         preprocess_fn=None, class_index=None, batch_size=32):
    """Return per-mode statistics of the Score-CAM channel weights."""
    mask_raw = raw_img_array is not None and preprocess_fn is not None
    base = np.asarray(raw_img_array if mask_raw else img_array, dtype=np.float32)

    cls = resolve_class(model(img_array, training=False).numpy(), class_index)
    act_model = activation_model(model, layer_name)
    act = np.asarray(act_model(img_array, training=False), dtype=np.float32)
    input_hw = model_input_hw(model)

    masked = []
    for k in range(act.shape[3]):
        m = _resize(act[0, :, :, k], input_hw)
        lo, hi = float(m.min()), float(m.max())
        m = (m - lo) / (hi - lo) if hi > lo else np.zeros_like(m)
        img = base[0] * m[..., None]
        masked.append(preprocess_fn(img.copy()) if mask_raw else img)
    masked = np.stack(masked)

    def forward(as_logits):
        with logit_output(model, enabled=as_logits):
            return np.concatenate([
                np.asarray(model(masked[i:i + batch_size], training=False), dtype=np.float32)
                for i in range(0, len(masked), batch_size)
            ])

    logits = forward(True)
    probs = forward(False)

    modes = {
        "paper": softmax(logits[:, cls], axis=0),
        "reference": probs[:, cls] / max(float(probs[:, cls].sum()), 1e-12),
    }
    stats = {
        "class_index": cls,
        "n_channels": int(act.shape[3]),
        "logit_min": float(logits[:, cls].min()),
        "logit_max": float(logits[:, cls].max()),
        "logit_std": float(logits[:, cls].std()),
    }
    for name, w in modes.items():
        w = np.asarray(w, dtype=np.float64) / float(np.sum(w))
        stats[name] = {
            "largest_weight": float(w.max()),
            "top10_share": float(np.sort(w)[-10:].sum()),
            "effective_channels": _effective_channels(w),
        }
    return stats


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="./image/hummingbird.jpg")
    parser.add_argument("--layer", default="block5_conv3")
    args = parser.parse_args()

    from keras.applications.vgg16 import VGG16, preprocess_input

    from .preprocess import read_and_preprocess_img, read_img

    model = VGG16(include_top=True, weights="imagenet")
    stats = weight_concentration(
        model,
        read_and_preprocess_img(args.image),
        args.layer,
        raw_img_array=read_img(args.image),
        preprocess_fn=preprocess_input,
    )

    print(f"{args.image}  layer={args.layer}  class={stats['class_index']}  "
          f"channels={stats['n_channels']}")
    print(f"target-class logits: min={stats['logit_min']:.2f} "
          f"max={stats['logit_max']:.2f} std={stats['logit_std']:.2f}")
    print(f"{'weight_mode':<12}{'largest':>10}{'top-10':>10}{'exp(H)':>10}")
    for mode in ("paper", "reference"):
        s = stats[mode]
        print(f"{mode:<12}{s['largest_weight']:>9.1%}{s['top10_share']:>10.1%}"
              f"{s['effective_channels']:>10.1f}")


if __name__ == "__main__":
    _main()
