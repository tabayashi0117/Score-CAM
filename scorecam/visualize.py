"""Turning a CAM into something you can look at."""

import numpy as np

from ._common import sigmoid

__all__ = ["superimpose"]


def superimpose(original_img, cam, emphasize=False, heatmap_intensity=0.8):
    """Overlay ``cam`` on an image and return it as RGB ``uint8``.

    Args:
        original_img: a path, or an already-loaded BGR/RGB ``uint8`` array.
        cam: 2-D CAM in ``[0, 1]``.
        emphasize: push the heatmap through a steep sigmoid first, which makes
            weak, diffuse maps (e.g. on the DAGM textures) legible.
        heatmap_intensity: weight of the heatmap relative to the image.
    """
    import cv2

    if isinstance(original_img, (str, bytes)) or hasattr(original_img, "__fspath__"):
        img_bgr = cv2.imread(str(original_img))
        if img_bgr is None:
            raise FileNotFoundError(f"could not read image: {original_img!r}")
    else:
        img_bgr = np.asarray(original_img)

    heatmap = cv2.resize(np.asarray(cam, dtype=np.float32), (img_bgr.shape[1], img_bgr.shape[0]))
    if emphasize:
        heatmap = sigmoid(heatmap, 50, 0.5, 1)
    heatmap = np.uint8(255 * np.clip(heatmap, 0.0, 1.0))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed = heatmap * heatmap_intensity + img_bgr
    superimposed = np.minimum(superimposed, 255.0).astype(np.uint8)
    return cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
