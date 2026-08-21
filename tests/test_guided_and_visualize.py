"""Guided Backpropagation, the overlay helper, and the compatibility shim."""

import numpy as np
import pytest

from scorecam import GuidedBackPropagation, build_guided_model, superimpose

from .conftest import LAYER_NAME


def test_guided_model_predicts_identically(model, img_array):
    """The guided ReLU only changes the *backward* pass; forward behaviour, and
    therefore every prediction, must be bit-for-bit unaffected."""
    guided = build_guided_model(model)
    np.testing.assert_allclose(
        guided.predict(img_array, verbose=0),
        model.predict(img_array, verbose=0),
        atol=1e-6,
    )


def test_guided_model_accepts_a_builder_callable(build_model, img_array):
    """The pre-v0.2 API passed a zero-argument model factory."""
    guided = build_guided_model(lambda: build_model(seed=0))
    assert guided.get_layer(LAYER_NAME) is not None


def test_guided_backpropagation_shape_and_range(model, img_array):
    guided = build_guided_model(model)
    saliency = GuidedBackPropagation(guided, img_array, LAYER_NAME)

    assert saliency.shape == img_array.shape[1:]
    assert saliency.dtype == np.float32
    assert np.isfinite(saliency).all()
    assert saliency.min() >= 0.0 and saliency.max() <= 1.0


def test_superimpose_accepts_an_array(model, img_array):
    cam = np.linspace(0.0, 1.0, 16 * 16).reshape(16, 16).astype("float32")
    original = img_array[0].astype("uint8")

    overlaid = superimpose(original, cam)

    assert overlaid.shape == original.shape
    assert overlaid.dtype == np.uint8


def test_superimpose_emphasize(model, img_array):
    cam = np.linspace(0.0, 1.0, 16 * 16).reshape(16, 16).astype("float32")
    overlaid = superimpose(img_array[0].astype("uint8"), cam, emphasize=True)
    assert overlaid.dtype == np.uint8


def test_superimpose_reports_a_missing_file():
    cam = np.zeros((4, 4), dtype="float32")
    with pytest.raises(FileNotFoundError):
        superimpose("./image/does-not-exist.png", cam)


def test_shim_reexports_the_package():
    import gradcamutils
    import scorecam

    for name in gradcamutils.__all__:
        assert getattr(gradcamutils, name) is getattr(scorecam, name)
