#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2017 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.
"""Contains definitions for DenseNet.

This implementation is based on tensorflow.contrib.slim.nets.resnet_v2

DenseNet was originally proposed in this paper:
  [1] Huang, G., et al. (2016).
      "Densely connected convolutional networks."
      arXiv preprint arXiv:1608.06993.

Typical use:

   import densenet

DenseNet-bc-k12-l40 for image classification into 10 classes:

   # inputs has shape [batch, 224, 224, 3]
   with slim.arg_scope(densenet.densenet_arg_scope()):
      net, end_points = densenet.densenet_bc_k12_l40(inputs, 10, is_training=False)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf

slim = tf.contrib.slim


def densenet_arg_scope(weight_decay=1e-4,
                       dropout_keep_prob=0.8,
                       batch_norm_decay=0.9,
                       batch_norm_epsilon=1e-5,
                       batch_norm_scale=True):
  """Defines the default DenseNet arg scope.

  Args:
    weight_decay: The weight decay to use for regularizing the model.
    dropout_keep_prob: The dropout keep probability used after each convolutional
      layer. It is used for three datasets without data augmentation: CIFAR10,
      CIFAR 100, and SVHN.
    batch_norm_decay: The moving average decay when estimating layer activation
      statistics in batch normalization.
    batch_norm_epsilon: Small constant to prevent division by zero when
      normalizing activations by their variance in batch normalization.
    batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
      activations in the batch normalization layer.

  Returns:
    An `arg_scope` to use for the DenseNet models.
  """
  batch_norm_params = {
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': batch_norm_scale,
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
  }

  with slim.arg_scope(
      [slim.conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=slim.variance_scaling_initializer(),
      activation_fn=tf.nn.relu,
      normalizer_fn=slim.batch_norm,
      normalizer_params=batch_norm_params):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with slim.arg_scope([slim.dropout], keep_prob=dropout_keep_prob) as arg_sc:
        return arg_sc


class Block(collections.namedtuple('Block', 
                                   ['scope', 
                                    'inter_fn', 'inter_args',
                                    'trans_fn', 'trans_args'])):
  """A named tuple describing a DenseNet block.

  A block contains several internal units and an optional transition unit.

  Its parts are:
    scope: The scope of the `Block`.
    inter_fn: The DenseNet internal unit function which takes as input a `Tensor` and
      returns another `Tensor` with the output of the DenseNet internal unit.
    inter_args: A list of length equal to the number of internal units in the `Block`.
      The list contains one (depth, depth_bottleneck) tuple for each unit in the
      block to serve as argument to inter_fn.
    trans_fn: The DenseNet transition layer between two blocks. if None, no
      transition layer will be used which is useful for last layer.
    trans_args: A (keep_fraction,) tuple for argument to trans_fn. keep_fraction is the
      fraction of feature maps that will be remained after transition function.
  """


@slim.add_arg_scope
def stack_blocks_dense(net, blocks, output_stride=None,
                       outputs_collections=None):
  """Stacks DenseNet `Blocks`.

  First, this function creates scopes for the DenseNet in the form of
  'block_name/unit_1', 'block_name/unit_2', etc.

  Most DenseNets consist of 3 DenseNet blocks and subsample the activations by a
  factor of 2 when transitioning between consecutive DenseNet blocks. This results
  to a nominal DenseNet output_stride equal to 8.

  Args:
    net: A `Tensor` of size [batch, height, width, channels].
    blocks: A list of length equal to the number of DenseNet `Blocks`. Each
      element is a DenseNet `Block` object describing the units in the `Block`.
    outputs_collections: Collection to add the DenseNet block outputs.

  Returns:
    net: Output tensor.
  """
  for block in blocks:
    with tf.variable_scope(block.scope, 'block', [net]) as sc:
      for i, unit in enumerate(block.inter_args):
        unit_depth, unit_depth_bottleneck = unit
        net = block.inter_fn(net,
                             depth=unit_depth,
                             depth_bottleneck=unit_depth_bottleneck,
                             scope='unit_%d' % (i + 1))
      if block.trans_fn is not None:
        tran_keep_fraction, = block.trans_args
        net = block.trans_fn(net,
                             keep_fraction=tran_keep_fraction,
                             scope='transition')
      net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)
  return net


@slim.add_arg_scope
def composite_function(inputs, depth, kernel_size=3, scope=None):
  """Composite function from paper H_l

  This function sequentially performs:
    - Batch normalization
    - ReLU 
    - Convolution with specified kernel_size
    - Dropout

  Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth: The depth of the convolutional layer, also known as the growth rate
           k in the paper, typical number is 12.
    kernel_size: The kernel size of convolutional layer.
    scope: Optional variable_scope.
  """
  with tf.variable_scope(scope, 'H', [inputs]):
    output = slim.batch_norm(inputs)
    output = tf.nn.relu(output)
    output = slim.conv2d(output, depth, [kernel_size, kernel_size],
                         stride=1, padding='SAME',
                         biases_initializer=None,
                         normalizer_fn=None,
                         activation_fn=None, 
                         scope='conv')
    output = slim.dropout(output)
    return output


@slim.add_arg_scope
def transition_unit(inputs, keep_fraction, outputs_collections=None, scope=None):
  """Transition unit between two consecutive blocks

  Args:
    inputs: A tensor of size [batch, height, width, channels].
    keep_fraction: A float scalar in [0.0, 1.0], used to determine feature reduction ratio.
    outputs_collections: The collections to which the outputs are added.
    scope: Optional variable_scope.

  Returns:
    The transition unit's output.
  """
  with tf.variable_scope(scope, 'transition', [inputs]) as sc:
    depth = int(int(inputs.get_shape()[-1]) * keep_fraction)
    output = composite_function(inputs, depth, kernel_size=1, scope='compress')
    output = slim.avg_pool2d(output, kernel_size=[2, 2], stride=2, padding='VALID')
    return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.original_name_scope,
                                            output)


@slim.add_arg_scope
def internal_unit(inputs, depth, depth_bottleneck=None, outputs_collections=None, scope=None):
  """Densenet internal unit function.

  This is the original densenet unit proposed in [1]. See Fig. 1(a) of [2] for
  its definition. Note that we use here the bottleneck variant which has an
  extra bottleneck layer.

  When putting together two consecutive DenseNet blocks that use this unit, one
  should use stride = 2 in the transition unit.

  Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth: The depth of the DenseNet unit output.
    depth_bottleneck: The depth of the bottleneck layers. None, if doesn't use bottleneck connection.
    outputs_collections: Collection to add the DenseNet unit output.
    scope: Optional variable_scope.

  Returns:
    The DenseNet unit's output.
  """
  with tf.variable_scope(scope, 'unit', [inputs]) as sc:
    if depth_bottleneck is not None:
      # Use bottleneck connection which is basically a composite function with kernel_size = 1.
      inter = composite_function(inputs, depth_bottleneck, kernel_size=1, scope='bottleneck')
    else:
      inter = inputs
    comp_out = composite_function(inter, depth, kernel_size=3, scope='conv')

    # This is the main implementation difference between DenseNet and ResNet_v2, instead of adding
    # these two feature maps, DenseNet concatenates them.
    output = tf.concat((inputs, comp_out), 3)
    return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.original_name_scope,
                                            output)


def densenet(inputs,
             blocks,
             num_classes=None,
             is_training=True,
             global_pool=True,
             include_root_block=True,
             reuse=None,
             scope=None):
  """Generator for DenseNet models.

  This function generates a family of DenseNet models. See the densenet_*()
  methods for specific model instantiations, obtained by selecting different
  block instantiations that produce DenseNet of various depths.

  Args:
    inputs: A tensor of size [batch, height_in, width_in, channels].
    blocks: A list of length equal to the number of DenseNet blocks. Each element
      is a Block object describing the units in the block.
    num_classes: Number of predicted classes for classification tasks. If None
      we return the features before the logit layer.
    is_training: whether is training or not.
    global_pool: If True, we perform global average pooling before computing the
      logits. Set to True for image classification, False for dense prediction.
    include_root_block: If True, include the initial convolution followed by
      max-pooling, if False excludes it. If excluded, `inputs` should be the
      results of an activation-less convolution.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.

  Returns:
    net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
      If global_pool is False, then height_out and width_out are reduced by a
      factor of output_stride compared to the respective height_in and width_in,
      else both height_out and width_out equal one. If num_classes is None, then
      net is the output of the last DenseNet block, potentially after global
      average pooling. If num_classes is not None, net contains the pre-softmax
      activations.
    end_points: A dictionary from components of the network to the corresponding
      activation.
  """
  with tf.variable_scope(scope, 'densenet', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.name + '_end_points'
    with slim.arg_scope([slim.conv2d, internal_unit, transition_unit,
                         stack_blocks_dense],
                        outputs_collections=end_points_collection):
      with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
        net = inputs
        if include_root_block:
          # We do not include batch normalization or activation functions in
          # conv1 because the first internal unit will perform these.
          with slim.arg_scope([slim.conv2d],
                              activation_fn=None, normalizer_fn=None):
            net = slim.conv2d(net, 16, 3, stride=1, scope='conv1')
        net = stack_blocks_dense(net, blocks)
        # This is needed because the pre-activation variant does not have batch
        # normalization or activation functions in the residual unit output.
        net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
        if global_pool:
          # Global average pooling.
          net = tf.reduce_mean(net, [1, 2], name='global_avg_pool', keep_dims=True)
        if num_classes is not None:
          net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                            normalizer_fn=None, scope='logits')
        # Convert end_points_collection into a dictionary of end_points.
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        if num_classes is not None:
          end_points['predictions'] = slim.softmax(net, scope='predictions')
        return net, end_points


def densenet_bc_k12_l40(inputs,
                        num_classes=None,
                        is_training=True,
                        global_pool=True,
                        reuse=None,
                        scope='densenet_bc_k12_l40'):
  """DenseNet-BC with grow rate 12 and layer 40

  Specifications:
    - growth rate k = 12
    - number of convolutional layers = 40
    - number of bottleneck feature maps = 4k
    - compression fractor theta = 0.5
  """

  depth = 12
  bottleneck_depth = depth * 4
  compress_keep_fraction = 0.5
  layer = 40
  units_num = (layer - 4) // 6  # Divide by 6 since each units contains two
                                # convolutions: bottleneck conv + 3x3 conv

  blocks = [
      Block(
          'block1', internal_unit, [(depth, bottleneck_depth)] * units_num,
                    transition_unit, (compress_keep_fraction,)),
      Block(
          'block2', internal_unit, [(depth, bottleneck_depth)] * units_num,
                    transition_unit, (compress_keep_fraction,)),
      Block(
          'block3', internal_unit, [(depth, bottleneck_depth)] * units_num,
                    None, None),
  ]

  return densenet(inputs, blocks, num_classes, is_training=is_training,
                  global_pool=global_pool,
                  include_root_block=True,
                  reuse=reuse, scope=scope)


def densenet_k12_l40(inputs,
                     num_classes=None,
                     is_training=True,
                     global_pool=True,
                     reuse=None,
                     scope='densenet_k12_l40'):
  """DenseNet with grow rate 12 and layer depth 40

  Specifications:
    - growth rate k = 12
    - number of convolutional layers = 40
    - number of bottleneck feature maps = None
    - compression fractor theta = 1.0
  """

  depth = 12
  bottleneck_depth = None
  compress_keep_fraction = 1.0
  layer = 40
  units_num = (layer - 4) // 3

  blocks = [
      Block(
          'block1', internal_unit, [(depth, bottleneck_depth)] * units_num,
                    transition_unit, (compress_keep_fraction,)),
      Block(
          'block2', internal_unit, [(depth, bottleneck_depth)] * units_num,
                    transition_unit, (compress_keep_fraction,)),
      Block(
          'block3', internal_unit, [(depth, bottleneck_depth)] * units_num,
                    None, None),
  ]

  return densenet(inputs, blocks, num_classes, is_training=is_training,
                  global_pool=global_pool,
                  include_root_block=True,
                  reuse=reuse, scope=scope)
