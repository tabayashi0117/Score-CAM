"""The contract every CAM function must satisfy.

These are property tests, not golden-value tests: they survive a change of
implementation but catch the failure modes this repository actually had --
breaking on a new TensorFlow, and dividing by zero on a flat CAM.
"""

import numpy as np
import pytest

from scorecam import GradCam, GradCamPlusPlus, ScoreCam

from .conftest import LAYER_NAME, N_CLASSES

SPATIAL = (16, 16)  # LAYER_NAME sits after one 2x2 pool


def call(fn, model, img_array, **kwargs):
    if fn is ScoreCam:
        kwargs.setdefault("batch_size", 4)
    return fn(model, img_array, LAYER_NAME, **kwargs)


CAMS = [GradCam, GradCamPlusPlus, ScoreCam]


@pytest.mark.parametrize("fn", CAMS, ids=lambda f: f.__name__)
def test_returns_2d_float32_in_unit_range(fn, model, img_array):
    cam = call(fn, model, img_array)

    assert cam.ndim == 2, "CAM must be a 2-D map"
    assert cam.shape == SPATIAL
    assert cam.dtype == np.float32
    assert np.isfinite(cam).all(), "CAM must not contain NaN or Inf"
    assert cam.min() >= 0.0 and cam.max() <= 1.0


@pytest.mark.parametrize("fn", CAMS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("class_index", range(N_CLASSES))
def test_accepts_an_explicit_class(fn, model, img_array, class_index):
    cam = call(fn, model, img_array, class_index=class_index)
    assert cam.shape == SPATIAL
    assert np.isfinite(cam).all()


@pytest.mark.parametrize("fn", CAMS, ids=lambda f: f.__name__)
def test_flat_input_does_not_divide_by_zero(fn, model):
    """A zero image yields zero post-ReLU activations, hence a CAM whose max is
    0. The pre-v0.2 code did `cam /= np.max(cam)` and returned NaN here."""
    zeros = np.zeros((1, 32, 32, 3), dtype="float32")
    cam = call(fn, model, zeros)
    assert np.isfinite(cam).all()


@pytest.mark.parametrize("fn", CAMS, ids=lambda f: f.__name__)
def test_leaves_the_model_untouched(fn, model, img_array):
    """Stripping the final softmax must be reverted, whatever happens."""
    before = model.layers[-1].activation
    before_preds = model.predict(img_array, verbose=0)

    call(fn, model, img_array)

    assert model.layers[-1].activation is before
    np.testing.assert_allclose(model.predict(img_array, verbose=0), before_preds, atol=1e-6)


@pytest.mark.parametrize("fn", [GradCam, GradCamPlusPlus], ids=lambda f: f.__name__)
def test_logits_and_probabilities_both_work(fn, model, img_array):
    for use_logits in (True, False):
        cam = call(fn, model, img_array, use_logits=use_logits)
        assert cam.shape == SPATIAL
        assert np.isfinite(cam).all()


def test_linear_output_model_needs_no_softmax_stripping(build_model, img_array):
    """A model that already ends in a linear layer must work unchanged."""
    linear_model = build_model(seed=1, activation="linear")
    cam = GradCam(linear_model, img_array, LAYER_NAME)
    assert np.isfinite(cam).all()


@pytest.mark.parametrize("fn", CAMS, ids=lambda f: f.__name__)
def test_list_input_models_are_called_correctly(fn, list_input_model, img_array):
    """`Model([inputs], outputs)` expects a list. Calling it with a bare array
    makes Keras warn and fall back -- and pytest is configured to turn that
    UserWarning into an error, so this test fails loudly if a call is added
    that bypasses call_model()."""
    assert isinstance(list_input_model.input, list)

    cam = call(fn, list_input_model, img_array)

    assert cam.shape == SPATIAL
    assert np.isfinite(cam).all()


def test_guided_backprop_on_a_list_input_model(list_input_model, img_array):
    from scorecam import GuidedBackPropagation, build_guided_model

    saliency = GuidedBackPropagation(build_guided_model(list_input_model),
                                     img_array, LAYER_NAME)
    assert saliency.shape == img_array.shape[1:]
    assert np.isfinite(saliency).all()
