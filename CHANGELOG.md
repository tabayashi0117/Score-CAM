# Changelog

## 0.2.0

The first release that runs on a currently supported TensorFlow. Everything in
this repository raised `RuntimeError: tf.data.Dataset only supports Python-style
iteration in eager mode` on TensorFlow >= 2.16, because `gradcamutils.py` called
`tf.compat.v1.disable_eager_execution()` at import time.

### Breaking

- **Python >= 3.11, TensorFlow >= 2.16 (Keras 3).** The 3.11 floor is a security
  constraint: keras 3.14, which fixes GHSA-hqp4-2352-xf5r and
  GHSA-4f3f-g24h-fr8m, is not published for 3.10.
- The implementation moved into the `scorecam` package. **`gradcamutils` still
  works** — it re-exports everything — so existing code and the Qiita article
  keep running.
- `read_and_preprocess_img` no longer hard-codes VGG16 preprocessing. It still
  defaults to it; pass `preprocess_fn=` for other models.
- Removed `normalize()`, an unused L2 helper that only worked in graph mode.
- Score-CAM output changes: see the correctness fixes below.

### Correctness

- **Score-CAM eq. 8 was missing its min subtraction**, computing
  `A / (max - min)` instead of `(A - min) / (max - min)`.
- **Score-CAM's channel weights were formed by a double softmax.** The intended
  weight is the target-class probability under each masked input, which is one
  softmax over logits. A Keras classifier already ends in a softmax, so applying
  another compressed the weights towards uniform — 487 of 512 effective
  channels, close to an unweighted mean of the activation maps.
- **Grad-CAM, Grad-CAM++ and Score-CAM differentiated the softmax probability**,
  not the pre-softmax score the papers define them on. On a saturated prediction
  the gradient all but vanishes; old and new Grad-CAM maps correlate at r=0.60.
  `use_logits=False` restores the old behaviour.
- `cam /= np.max(cam)` returned NaN for an all-zero CAM.
- `softmax()` had no max subtraction and overflowed on large logits.
- `max_N` greater than the channel count crashed in `np.argpartition`.
- Models whose inputs were declared as a list — which is what
  `Model(ResNet50(input_tensor=...).input, head)` produces — were called with a
  bare array, making Keras fall back.
- The notebook's Guided-Faster-Score-CAM panel was multiplied by `score_cam`
  instead of `faster_score_cam`, so it showed the wrong method.

### Added

- `class_index=` on every CAM, to explain a class other than the top-1.
- `read_img()`, the raw array Score-CAM's Hadamard product is defined on.
- `weight_mode=` and `raw_img_array=`/`preprocess_fn=`, which select between
  Algorithm 1 of the paper and the authors' released implementation where the
  two disagree. Defaults follow the released implementation; see the README.
- `scorecam.diagnostics`, which measures how concentrated the channel weights
  are under each mode.
- `scorecam.plotting` and `scripts/regenerate_results.py`, so the committed
  figures and the notebook cannot drift apart.
- A pytest suite (68 tests, CPU-only, no downloads, under a second) and CI that
  re-runs it against the newest TensorFlow on the 1st of each month.

### Changed

- Score-CAM scores its masked inputs in batches; it previously built all of them
  at once, 300 MB for VGG16's 512 channels.
- Guided Backpropagation uses `tf.custom_gradient` instead of the private
  `ops._gradient_registry` and `gradient_override_map`. `build_guided_model`
  takes a model directly, though it still accepts the old factory callable.
- The notebook went from 7.2 MB of embedded outputs to 20 KB, and no longer
  carries a second copy of `ScoreCam`.
- The DAGM example names its truncation layer (`conv3_block4_add`) instead of
  indexing `model.layers[-98]`, and feeds the ResNet its own preprocessing.

## 0.1.0

Initial release, accompanying the
[Qiita article](https://qiita.com/futakuchi0117/items/95c518254185ec5ea485).
TensorFlow 2.0-2.3.
