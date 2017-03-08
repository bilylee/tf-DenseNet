# tf-DenseNet
DenseNet implemented in TensorFlow 

## Introduction
This repository implements the paper: [Densely Connected Convolutional Networks](https://github.com/liuzhuang13/DenseNet) and is specifically designed for easy integration.

Other implementations can be found [here](https://github.com/liuzhuang13/DenseNet)

## Usage
```python
import densenet
with slim.arg_scope(densenet.densenet_arg_scope()):
  # DenseNet with bottleneck convolution and feature compression
  net, end_points = densenet.densenet_bc_k12_l40(inputs, 10, is_training=False)
  # OR without it
  net, end_points = densenet.densenet_k12_l40(inputs, 10, is_training=False)
```
