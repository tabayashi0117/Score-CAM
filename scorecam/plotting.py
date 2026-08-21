"""Building the comparison figures shown in the README and the notebook.

Both the notebook and ``scripts/regenerate_results.py`` call in here, so a
committed figure can never drift from what the notebook shows.
"""

import numpy as np

from .grad_cam import GradCam, GradCamPlusPlus
from .guided import GuidedBackPropagation
from .preprocess import read_img
from .score_cam import ScoreCam
from .visualize import superimpose

__all__ = ["explain", "comparison_figure"]

METHOD_LABELS = {
    "grad_cam": "Grad-CAM",
    "grad_cam_pp": "Grad-CAM++",
    "score_cam": "Score-CAM",
    "faster_score_cam": "Faster-Score-CAM",
}


def explain(model, img_path, layer_name, preprocess_fn, size=(224, 224), max_N=10,
            guided_model=None, class_index=None, **score_cam_kwargs):
    """Run every method on one image and return the raw maps.

    ``score_cam_kwargs`` is forwarded to :func:`~scorecam.ScoreCam`, so a caller
    can ask for ``weight_mode="paper"`` or paper-faithful raw-pixel masking.
    """
    raw = read_img(img_path, size=size)
    img_array = preprocess_fn(raw.copy())

    maps = {
        "grad_cam": GradCam(model, img_array, layer_name, class_index=class_index),
        "grad_cam_pp": GradCamPlusPlus(model, img_array, layer_name, class_index=class_index),
        "score_cam": ScoreCam(model, img_array, layer_name,
                              class_index=class_index, **score_cam_kwargs),
        "faster_score_cam": ScoreCam(model, img_array, layer_name, max_N=max_N,
                                     class_index=class_index, **score_cam_kwargs),
    }
    if guided_model is not None:
        maps["saliency"] = GuidedBackPropagation(guided_model, img_array, layer_name)
    return maps


def _guided(saliency, cam, shape_hw):
    import cv2

    height, width = shape_hw
    saliency = cv2.resize(saliency, (width, height))
    cam = cv2.resize(cam, (width, height))
    return saliency * cam[..., np.newaxis]


def comparison_figure(img_path, maps, rows=("overlay", "guided"), title=None,
                      emphasize=False):
    """Lay the maps out as a grid of ``len(rows)`` x 5 panels.

    Rows:
        ``"overlay"``   the CAM superimposed on the image
        ``"guided"``    Guided Backprop, and Guided-* for each method
        ``"raw"``       the CAM on its own
    """
    import cv2
    import matplotlib.pyplot as plt
    from keras.utils import load_img

    orig_img = np.array(load_img(img_path), dtype=np.uint8)
    shape_hw = orig_img.shape[:2]
    methods = list(METHOD_LABELS)

    fig, axes = plt.subplots(nrows=len(rows), ncols=5, figsize=(18, 3.6 * len(rows)),
                             squeeze=False)

    for r, row in enumerate(rows):
        if row == "overlay":
            panels = [(orig_img, "input image")] + [
                (superimpose(img_path, maps[m], emphasize=emphasize), METHOD_LABELS[m])
                for m in methods
            ]
        elif row == "guided":
            saliency = maps["saliency"]
            panels = [(cv2.resize(saliency, shape_hw[::-1]), "Guided-BP")] + [
                # NOTE: the pre-v0.2 notebook multiplied the Guided-Faster-Score-CAM
                # panel by score_cam, not faster_score_cam. It showed the wrong map.
                (_guided(saliency, maps[m], shape_hw), f"Guided-{METHOD_LABELS[m]}")
                for m in methods
            ]
        elif row == "raw":
            panels = [(orig_img, "input image")] + [
                (maps[m], METHOD_LABELS[m]) for m in methods
            ]
        else:
            raise ValueError(f"unknown row {row!r}")

        for ax, (image, label) in zip(axes[r], panels):
            ax.imshow(image)
            ax.set_title(label)
            ax.axis("off")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig
