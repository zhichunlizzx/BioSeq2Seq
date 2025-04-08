#!/usr/bin/env python
# Copyright 2023 Z Zhang

# BioSeq2Seq, Version 1.0;
# you may not use this file except in compliance with the License.
# Use of this code requires following originality guidelines
# and declaring the source of the code.
# email:zhichunli@mail.dlut.edu.cn
# =========================================================================

import tensorflow as tf
import numpy as np
from model_function import attention_model
from model_function.tools import Residual, SoftPlus, Sequential, GELU, pooling_module, exponential_linspace_int, TargetLengthCrop1D
from typing import Dict


class GEModel(tf.keras.Model):
  """Main model."""

  def __init__(self,
               channels: int = 1536,
               num_transformer_layers: int = 11,
               num_heads: int = 8,
               output_channels = 1,
               target_length=896,
               pooling_type: str = 'max',
               name: str = 'GEModel'):

    super(GEModel, self).__init__(name=name)

    # pylint: disable=g-complex-comprehension,g-long-lambda,cell-var-from-loop
    dropout_rate = 0.4
    assert channels % num_heads == 0, ('channels needs to be divisible '
                                       f'by {num_heads}')
    whole_attention_kwargs = {
        'attention_dropout_rate': 0.05,
        'initializer': None,
        'key_size': 64,
        'num_heads': num_heads,
        'num_relative_position_features': channels // num_heads,
        'positional_dropout_rate': 0.01,
        'relative_position_functions': [
            'positional_features_exponential',
            'positional_features_central_mask',
            'positional_features_gamma'
        ],
        'relative_positions': True,
        'scaling': True,
        'value_size': channels // num_heads,
        'zero_initialize': True
    }

    trunk_name_scope = tf.name_scope('trunk')
    trunk_name_scope.__enter__()

    # lambda is used in Sequential to construct the module under tf.name_scope.
    def conv_block(filters, width=1, w_init=None, name='conv_block', **kwargs):
      return Sequential([
          tf.keras.layers.BatchNormalization(scale=True,
                        center=True,
                        momentum=0.9,
                        gamma_initializer=tf.keras.initializers.Ones()),
          GELU(),
          tf.keras.layers.Conv1D(filters=filters, kernel_size=width, kernel_initializer=w_init, padding="same", **kwargs)
      ], name=name)

    stem = Sequential([
        tf.keras.layers.Conv1D(filters=channels // 2, kernel_size=15, padding="same"),
        Residual(conv_block(channels // 2, 1, name='pointwise_conv_block_stem')),
        pooling_module(pooling_type, pool_size=2),
    ], name='stem')

    filter_list = exponential_linspace_int(start=channels // 2, end=channels,
                                           num=6, divisible_by=64)
    conv_tower = Sequential([
        Sequential([
            conv_block(num_filters, 5),
            Residual(conv_block(num_filters, 1, name='pointwise_conv_block_conv')),
            pooling_module(pooling_type, pool_size=2),
            ],
                   name=f'conv_tower_block_{i}')
        for i, num_filters in enumerate(filter_list)], name='conv_tower')

    # Transformer.
    def transformer_mlp():
      return Sequential([
          tf.keras.layers.LayerNormalization(axis=-1, scale=True, center=True, gamma_initializer=tf.keras.initializers.Ones()),
          tf.keras.layers.Dense(channels * 2),
          tf.keras.layers.Dropout(dropout_rate),
          tf.keras.layers.ReLU(),
          tf.keras.layers.Dense(channels),
          tf.keras.layers.Dropout(dropout_rate)], name='mlp')

    transformer = Sequential([
        Sequential([
            Residual(Sequential([
                tf.keras.layers.LayerNormalization(axis=-1,
                              scale=True, center=True,
                              gamma_initializer=tf.keras.initializers.Ones()),

                attention_model.MultiheadAttention(**whole_attention_kwargs,
                                                    name=f'attention_{i}'),
                tf.keras.layers.Dropout(dropout_rate)], name='mha')),
            Residual(transformer_mlp())], name=f'transformer_block_{i}')
        for i in range(num_transformer_layers)], name='transformer')
    
    
    crop_final = TargetLengthCrop1D(target_length, name='target_input')

    final_pointwise_promoters = Sequential([
        conv_block(channels * 2, 1),
        tf.keras.layers.Dropout(dropout_rate / 8),
        GELU()], name='final_pointwise')

    # self._initializer = tf.keras.initializers.VarianceScaling(scale=2.0)
    # self.var = tf.Variable(self._initializer([1], dtype=tf.float32))
    
    self._conv = Sequential([stem,
                            conv_tower],
                            name = 'conv_block_conv')
    
    self._trunk = Sequential([transformer,
                            crop_final,
                            ],
                             name='trunk_others')
    
    trunk_name_scope.__exit__(None, None, None)

    with tf.name_scope('heads'):
      self._heads = Sequential([final_pointwise_promoters,
                                tf.keras.layers.Dense(units=output_channels), SoftPlus()
                                ], name=f'head_promoter')

    
  
  @property
  def conv(self):
    return self._conv

  @property
  def conv_seq(self):
    return self._conv_seq

  @property
  def trunk(self):
    return self._trunk

  @property
  def heads(self):
    return self._heads


  def __call__(self, inputs, is_training) -> Dict[str, tf.Tensor]:
    inputs = inputs[0]

    outputs = self.conv(inputs, training=is_training)

    outputs = self.trunk(outputs, training=is_training)

    outputs = self.heads(outputs, training=is_training)

    return outputs