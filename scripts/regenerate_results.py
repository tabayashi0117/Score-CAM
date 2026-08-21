"""Regenerate the VGG16 figures in result/ and image/sample_output.png.

    uv run python scripts/regenerate_results.py

The DAGM figures (result/Class*_result_*.png) are not produced here: they need
the DAGM dataset and six retrained ResNets. See the notebook.
"""

import argparse
import os
import pathlib

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
IMAGES = ["cat_dog.png", "collies.JPG", "water-bird.JPEG", "multiple_dogs.jpg", "snake.JPEG"]
LAYER = "block5_conv3"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs="*", default=IMAGES)
    parser.add_argument("--dpi", type=int, default=72)
    args = parser.parse_args()

    from keras.applications.vgg16 import VGG16, decode_predictions, preprocess_input

    from scorecam import build_guided_model, read_and_preprocess_img
    from scorecam.plotting import comparison_figure, explain

    model = VGG16(include_top=True, weights="imagenet")
    guided_model = build_guided_model(model)

    for basename in args.images:
        img_path = ROOT / "image" / basename
        preds = model.predict(read_and_preprocess_img(img_path), verbose=0)
        label = decode_predictions(preds, top=1)[0][0][1]
        confidence = float(np.max(preds))

        maps = explain(model, str(img_path), LAYER, preprocess_input,
                       guided_model=guided_model)
        fig = comparison_figure(str(img_path), maps)
        out = ROOT / "result" / f"result_{label}.png"
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"{basename:20s} -> {out.name:32s} ({label}, {confidence:.3f})")

    # The README banner: one row, no guided panels.
    banner_path = ROOT / "image" / "hummingbird.jpg"
    maps = explain(model, str(banner_path), LAYER, preprocess_input)
    fig = comparison_figure(str(banner_path), maps, rows=("overlay",))
    fig.savefig(ROOT / "image" / "sample_output.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"{'hummingbird.jpg':20s} -> image/sample_output.png")


if __name__ == "__main__":
    main()
