"""Score-CAM specifics: Faster-Score-CAM, batching, and the two divergences
between the paper and the authors' reference implementation."""

import numpy as np
import pytest

from scorecam import ScoreCam
from scorecam._common import activation_model, rescale, softmax
from scorecam.preprocess import read_and_preprocess_img  # noqa: F401  (import smoke)
from scorecam.score_cam import _select_channels

from .conftest import LAYER_NAME, N_FILTERS

SPATIAL = (16, 16)


@pytest.mark.parametrize("max_N", [-1, 1, 3, N_FILTERS, N_FILTERS + 5])
def test_max_N_never_changes_the_output_shape(model, img_array, max_N):
    """Faster-Score-CAM drops channels; `max_N` larger than the channel count
    must clamp rather than crash in np.argpartition."""
    cam = ScoreCam(model, img_array, LAYER_NAME, max_N=max_N, batch_size=4)
    assert cam.shape == SPATIAL
    assert np.isfinite(cam).all()


def test_channel_selection_keeps_highest_variance_first():
    maps = np.zeros((1, 4, 4, 3), dtype="float32")
    maps[0, :, :, 0] = 1.0                       # std 0.0
    maps[0, :, :, 1] = np.arange(16).reshape(4, 4)   # largest std
    maps[0, :, :, 2] = np.arange(16).reshape(4, 4) * 0.5

    selected = _select_channels(maps, max_N=2)

    assert selected.shape[3] == 2
    np.testing.assert_allclose(selected[0, :, :, 0], maps[0, :, :, 1])
    np.testing.assert_allclose(selected[0, :, :, 1], maps[0, :, :, 2])


@pytest.mark.parametrize("batch_size", [1, 3, 64])
def test_batching_does_not_change_the_result(model, img_array, batch_size):
    """Regression: the weights once depended on batch_size, because .predict()
    cached a traced function that still contained the final softmax."""
    model.predict(img_array, verbose=0)  # poison the predict-function cache
    reference = ScoreCam(model, img_array, LAYER_NAME, batch_size=N_FILTERS)
    cam = ScoreCam(model, img_array, LAYER_NAME, batch_size=batch_size)
    np.testing.assert_allclose(cam, reference, atol=1e-5)


@pytest.mark.parametrize("weight_mode", ["paper", "reference"])
def test_both_weight_modes_are_valid_cams(model, img_array, weight_mode):
    cam = ScoreCam(model, img_array, LAYER_NAME, weight_mode=weight_mode, batch_size=4)
    assert cam.shape == SPATIAL
    assert cam.min() >= 0.0 and cam.max() <= 1.0


def test_rejects_unknown_weight_mode(model, img_array):
    with pytest.raises(ValueError, match="weight_mode"):
        ScoreCam(model, img_array, LAYER_NAME, weight_mode="nonsense")


def test_masking_raw_pixels_requires_both_arguments(model, img_array):
    with pytest.raises(ValueError, match="preprocess_fn"):
        ScoreCam(model, img_array, LAYER_NAME, raw_img_array=img_array)


def test_masking_raw_pixels_runs_and_differs(model, img_array):
    """Algorithm 1 masks the raw image; the reference implementation masks the
    preprocessed tensor. Both must work, and they must not be the same thing."""
    raw = img_array
    preprocessed = raw - 128.0

    paper = ScoreCam(
        model, preprocessed, LAYER_NAME, batch_size=4,
        raw_img_array=raw, preprocess_fn=lambda x: x - 128.0,
    )
    reference_impl = ScoreCam(model, preprocessed, LAYER_NAME, batch_size=4)

    assert np.isfinite(paper).all()
    assert not np.allclose(paper, reference_impl)


def test_default_weight_mode_is_the_reference_implementation(model, img_array):
    """The published README figures were produced with the authors' released
    behaviour; Algorithm 1 read literally is near-degenerate (see the module
    docstring), so it stays opt-in."""
    default = ScoreCam(model, img_array, LAYER_NAME, batch_size=4)
    reference = ScoreCam(model, img_array, LAYER_NAME, weight_mode="reference", batch_size=4)
    np.testing.assert_allclose(default, reference, atol=1e-6)


def test_weight_concentration_reports_both_modes(model, img_array):
    from scorecam.diagnostics import weight_concentration

    stats = weight_concentration(model, img_array, LAYER_NAME, batch_size=4)

    assert stats["n_channels"] == N_FILTERS
    for mode in ("paper", "reference"):
        s = stats[mode]
        assert 0.0 < s["largest_weight"] <= 1.0
        assert 1.0 <= s["effective_channels"] <= N_FILTERS + 1e-6


def test_reference_weights_are_class_probabilities_not_a_double_softmax(model, img_array):
    """A Keras classifier ends in a softmax. The pre-v0.2 code applied its own
    softmax on top of that output, flattening the weights towards uniform; the
    authors' Torch code applies exactly one, to logits. Reproduce both and check
    we match the latter."""
    from scorecam._common import logit_output
    from scorecam.score_cam import masked_inputs

    layer = model.get_layer(LAYER_NAME)
    act = np.asarray(activation_model(model, LAYER_NAME)(img_array, training=False))
    cls = int(np.argmax(model(img_array, training=False).numpy()))

    masks = masked_inputs(np.asarray(img_array, dtype=np.float32)[0], act[0], (32, 32))

    with logit_output(model):
        logits = np.asarray(model(masks, training=False))
    probs = np.asarray(model(masks, training=False))

    single = softmax(logits, axis=1)[:, cls]     # what the reference code does
    double = softmax(probs, axis=1)[:, cls]      # what pre-v0.2 did
    assert not np.allclose(single, double)

    expected = rescale(np.dot(act[0], single))
    got = ScoreCam(model, img_array, LAYER_NAME, batch_size=4)
    np.testing.assert_allclose(got, expected, atol=1e-5)
    assert layer is model.get_layer(LAYER_NAME)
