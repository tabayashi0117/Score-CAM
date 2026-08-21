"""The numerical helpers."""

import keras
import numpy as np
import pytest

from scorecam._common import logit_output, rescale, sigmoid, softmax


def test_softmax_survives_large_logits():
    """The pre-v0.2 softmax was exp(x)/sum(exp(x)) with no max subtraction and
    overflowed to NaN on logits this size."""
    x = np.array([[1000.0, 1001.0, 999.0]])
    out = softmax(x, axis=1)

    assert np.isfinite(out).all()
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-6)
    assert out.argmax() == 1


def test_softmax_axis():
    x = np.arange(6.0).reshape(2, 3)
    np.testing.assert_allclose(softmax(x, axis=0).sum(axis=0), [1, 1, 1], atol=1e-6)
    np.testing.assert_allclose(softmax(x, axis=1).sum(axis=1), [1, 1], atol=1e-6)


def test_rescale_maps_to_unit_range_and_relus():
    out = rescale(np.array([[-5.0, 0.0, 2.0]]))
    np.testing.assert_allclose(out, [[0.0, 0.0, 1.0]])
    assert out.dtype == np.float32


def test_rescale_of_an_all_zero_cam_is_not_nan():
    out = rescale(np.zeros((3, 3)))
    assert np.isfinite(out).all()
    assert (out == 0).all()


def test_sigmoid_is_bounded():
    out = sigmoid(np.linspace(-1, 2, 50), 50, 0.5, 1)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_logit_output_restores_the_activation_after_an_exception(model):
    original = model.layers[-1].activation
    with pytest.raises(RuntimeError):
        with logit_output(model):
            assert model.layers[-1].activation is keras.activations.linear
            raise RuntimeError("boom")
    assert model.layers[-1].activation is original


def test_logit_output_can_be_disabled(model):
    original = model.layers[-1].activation
    with logit_output(model, enabled=False):
        assert model.layers[-1].activation is original


def test_logit_output_actually_removes_the_softmax(model, img_array):
    probs = model(img_array, training=False).numpy()
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    with logit_output(model):
        logits = model(img_array, training=False).numpy()

    assert not np.allclose(logits.sum(axis=1), 1.0, atol=1e-3), "softmax was not stripped"
    np.testing.assert_allclose(softmax(logits, axis=1), probs, atol=1e-5)


def test_logit_output_is_not_defeated_by_a_cached_predict_function(model, img_array):
    """Keras caches the tf.function built by .predict(). Anything inside
    logit_output() must therefore call the model eagerly, not via .predict()."""
    model.predict(img_array, verbose=0)  # warm (and cache) the predict function

    with logit_output(model):
        eager = model(img_array, training=False).numpy()

    assert not np.allclose(eager.sum(axis=1), 1.0, atol=1e-3)
