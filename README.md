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

# Requirement

- Python 3.6.8
- Keras 2.2.4
- tensorflow 1.14.0

# Usage

See `Score-CAM.ipynb`.