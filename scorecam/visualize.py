"""Turning a CAM into something you can look at."""

import matplotlib
import numpy as np

from ._common import resize_2d, sigmoid

__all__ = ["superimpose"]


def _as_rgb_uint8(image):
    """A path or an array, as an ``(H, W, 3)`` uint8 RGB array.

    Arrays are taken to be **RGB**. Before v0.2 this function read images with
    ``cv2.imread`` and therefore worked in BGR internally, converting back on
    the way out; an array passed in was silently assumed to be BGR too.
    """
    if isinstance(image, (str, bytes)) or hasattr(image, "__fspath__"):
        from keras.utils import load_img

        return np.array(load_img(image), dtype=np.uint8)

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) RGB image, got shape {array.shape}")
    return array


def superimpose(original_img, cam, emphasize=False, heatmap_intensity=0.8,
                colormap="jet"):
    """Overlay ``cam`` on an image and return it as RGB ``uint8``.

    Args:
        original_img: a path, or an already-loaded ``(H, W, 3)`` RGB array.
        cam: 2-D CAM in ``[0, 1]``.
        emphasize: push the heatmap through a steep sigmoid first, which makes
            weak, diffuse maps (e.g. on the DAGM textures) legible.
        heatmap_intensity: weight of the heatmap relative to the image.
        colormap: any Matplotlib colormap name. ``"jet"`` is the default because
            it is what every published CAM figure uses, but it is a poor
            colormap perceptually -- ``"turbo"`` is the modern drop-in, and
            ``"inferno"`` or ``"viridis"`` are better still if you do not need
            the familiar look.
    """
    img = _as_rgb_uint8(original_img)

    heatmap = resize_2d(cam, img.shape[:2])
    if emphasize:
        heatmap = sigmoid(heatmap, 50, 0.5, 1)
    heatmap = np.clip(heatmap, 0.0, 1.0)

    coloured = matplotlib.colormaps[colormap](heatmap)[..., :3] * 255.0
    superimposed = coloured * heatmap_intensity + img
    return np.minimum(superimposed, 255.0).astype(np.uint8)
