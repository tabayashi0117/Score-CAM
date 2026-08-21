"""The figure builder shared by the notebook and scripts/regenerate_results.py."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from scorecam import build_guided_model  # noqa: E402
from scorecam.plotting import METHOD_LABELS, comparison_figure, explain  # noqa: E402

from .conftest import LAYER_NAME  # noqa: E402

IMG = "./image/hummingbird.jpg"


@pytest.fixture(scope="module")
def maps(model):
    return explain(model, IMG, LAYER_NAME, lambda x: x / 127.5 - 1.0,
                   size=(32, 32), max_N=2, guided_model=build_guided_model(model),
                   batch_size=4)


def test_explain_returns_every_method(maps):
    assert set(METHOD_LABELS) <= set(maps)
    for name in METHOD_LABELS:
        cam = maps[name]
        assert cam.ndim == 2 and np.isfinite(cam).all()


@pytest.mark.parametrize("rows", [("overlay",), ("guided",), ("raw",), ("overlay", "guided")])
def test_comparison_figure_grid(maps, rows):
    fig = comparison_figure(IMG, maps, rows=rows)
    try:
        assert len(fig.axes) == 5 * len(rows)
        assert all(not ax.axison for ax in fig.axes)
    finally:
        plt.close(fig)


def test_guided_faster_score_cam_uses_the_faster_map(maps):
    """Pre-v0.2 the notebook multiplied this panel by score_cam instead of
    faster_score_cam, so it silently showed the wrong method."""
    fig = comparison_figure(IMG, maps, rows=("guided",))
    try:
        titles = [ax.get_title() for ax in fig.axes]
        assert titles == ["Guided-BP", "Guided-Grad-CAM", "Guided-Grad-CAM++",
                          "Guided-Score-CAM", "Guided-Faster-Score-CAM"]
        score, faster = fig.axes[3].images[0], fig.axes[4].images[0]
        assert not np.allclose(score.get_array(), faster.get_array())
    finally:
        plt.close(fig)


def test_unknown_row_is_rejected(maps):
    with pytest.raises(ValueError, match="unknown row"):
        comparison_figure(IMG, maps, rows=("nope",))
