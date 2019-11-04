# Score-CAM

![](./image/sample_output.png)

An implementation of [Score-CAM](https://arxiv.org/abs/1910.01279) with keras

The following methods are compared.

- Grad-CAM
- Grad-CAM++
- Guided Backpropagation
- Faster-Score-CAM (Original faster version of Score-CAM)

In addition, the followings are contained.

- Visualization of anomalous area for [**DAGM dataset**](https://resources.mpi-inf.mpg.de/conference/dagm/2007/prizes.html)
- Sample code for applying Score-CAM to your model.

![](./result/Class6_result_0.png)

Blog post: [Qiita]()

# Faster Score-CAM

We thought that several channels were dominant in generating the final heat map. Faster-Score-CAM adds the processing of “use only channels with large variances as mask images” to Score-CAM. (`max_N = -1` is the original Score-CAM). 

When using VGG16, Faster-Score-CAM is about 10 times faster than Score-CAM.

# Requirement

- Python 3.6.8
- Keras 2.2.4
- tensorflow 1.14.0

# Usage

See `Score-CAM.ipynb`.