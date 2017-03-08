# tf-DenseNet
DenseNet implemented in TensorFlow 

## Usage
```python
import densenet
with slim.arg_scope(densenet.densenet_arg_scope()):
  net, end_points = densenet.densenet_bc_k12_l40(inputs, 10, is_training=False)
  # OR
  net, end_points = densenet.densenet_k12_l40(inputs, 10, is_training=False)
```
