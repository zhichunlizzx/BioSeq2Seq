#!/usr/bin/env python
# Copyright 2023 Z Zhang

# BioSeq2Seq, Version 1.0;
# you may not use this file except in compliance with the License.
# Use of this code requires following originality guidelines
# and declaring the source of the code.
# email:zhichunli@mail.dlut.edu.cn
# =========================================================================

import tensorflow as tf
from typing import Any, Callable, Dict, Optional, Text, Union, Iterable
import numpy as np
import inspect

class Sigmoid(tf.keras.layers.Layer):
    def __init__(self, approximate: bool = True):
        super(Sigmoid, self).__init__()
        self.approximate = approximate
        self.supports_masking = True

    def call(self, input: tf.Tensor) -> tf.Tensor:
        return sigmoid(input, approximate=self.approximate)

    def get_config(self):
        config = {"approximate": self.approximate}
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape: tf.TensorShape):
        return input_shape


def sigmoid(x: tf.Tensor, approximate: bool = True) -> tf.Tensor:
    return tf.nn.sigmoid(x)


class SoftPlus(tf.keras.layers.Layer):
    def __init__(self, name: Optional[str] = "softplus", **kwargs) -> None:
        super(SoftPlus, self).__init__(name=name, **kwargs)

    def call(self, input: tf.Tensor) -> tf.Tensor:
        return tf.nn.softplus(features=input, name=self.name)

    def get_config(self) -> Dict:
        config = super().get_config()
        return config

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        return input_shape

class GELU(tf.keras.layers.Layer):
    def __init__(
        self, approximate: bool = True, name: Optional[str] = "gelu", **kwargs
    ) -> None:
        super(GELU, self).__init__(name=name, **kwargs)
        self.approximate = approximate
        self.supports_masking = True

    def call(self, input: tf.Tensor) -> tf.Tensor:
        return gelu(input, approximate=self.approximate)

    def get_config(self) -> Dict:
        config = {"approximate": self.approximate}
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        return input_shape


def gelu(x: tf.Tensor, approximate: bool = True) -> tf.Tensor:
    return tf.nn.sigmoid(1.702 * x) * x
    
class Residual(tf.keras.Model):
    def __init__(self, module: tf.Module, **kwargs) -> None:
        super(Residual, self).__init__(**kwargs)
        self._module = module

    def __call__(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
        return inputs + self._module(inputs, training=training)

    def get_config(self) -> Dict:
        config = {"module": self._module}
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        return input_shape

class SoftmaxPooling1D(tf.keras.Model):
    """Pooling operation with optional weights."""

    def __init__(
        self,
        pool_size: int = 2,
        per_channel: bool = False,
        w_init_scale: float = 0.0,
        name: str = "softmax_pooling",
        **kwargs,
    ) -> None:
        super(SoftmaxPooling1D, self).__init__(name=name, **kwargs)
        self._pool_size = pool_size
        self._per_channel = per_channel
        self._w_init_scale = w_init_scale
        self._logit_linear = None

    def _initialize(self, num_features: int) -> None:
        self._logit_linear = tf.keras.layers.Dense(
            units=num_features if self._per_channel else 1,
            use_bias=False,  # Softmax is agnostic to shifts.
            kernel_initializer=tf.keras.initializers.Identity(gain=self._w_init_scale),
        )

    def get_config(self) -> Dict:
        config = {
            "pool_size": self._pool_size,
            "per_channel": self._per_channel,
            "w_init_scale": self._w_init_scale,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        _, length, num_features = inputs.shape
        self._initialize(num_features)
        inputs = tf.reshape(
            inputs, (-1, length // self._pool_size, self._pool_size, num_features)
        )
        return tf.reduce_sum(
            inputs * tf.nn.softmax(self._logit_linear(inputs), axis=-2), 
            axis=-2)

def pooling_module(kind, pool_size):
    """Pooling module wrapper."""
    if kind == "attention":
        return SoftmaxPooling1D(pool_size=pool_size, per_channel=True, w_init_scale=2.0)
    elif kind == "max":
        return tf.keras.layers.MaxPool1D(pool_size=pool_size, padding="same")
    else:
        raise ValueError(f"Invalid pooling kind: {kind}.")


def exponential_linspace_int(start, end, num, divisible_by=1):
  """Exponentially increasing values of integers."""
  def _round(x):
    return int(np.round(x / divisible_by) * divisible_by)

  base = np.exp(np.log(end / start) / (num - 1))
  return [_round(start * base**i) for i in range(num)]


class TargetLengthCrop1D(tf.keras.Model):
    def __init__(self, target_length: int, name="target_length_crop", **kwargs):
        super(TargetLengthCrop1D, self).__init__(name=name, **kwargs)
        self._target_length = target_length

    # @tf.Module.with_name_scope
    def __call__(self, inputs):
        trim = (inputs.shape[-2] - self._target_length) // 2
        if trim < 0:
            raise ValueError("inputs longer than target length")

        return inputs[..., trim:-trim, :]

def accepts_is_training(module):
  return 'training' in list(inspect.signature(module.__call__).parameters)

class Sequential(tf.keras.Model):
  """snt.Sequential automatically passing is_training where it exists."""

  def __init__(self,
               layers: Optional[Union[Callable[[], Iterable[tf.keras.Model]],
                                      Iterable[Callable[..., Any]]]] = None,
               name: Optional[Text] = None):
    super().__init__(name=name)
    if layers is None:
      self._layers = []
    else:
      # layers wrapped in a lambda function to have a common namespace.
      if hasattr(layers, '__call__'):
        with tf.name_scope(name):
          layers = layers()
      self._layers = [layer for layer in layers if layer is not None]

  def __call__(self, inputs: tf.Tensor, training: bool, **kwargs):
    outputs = inputs
    for _, mod in enumerate(self._layers):
      if accepts_is_training(mod):
        outputs = mod(outputs, training=training, **kwargs)
      else:
        outputs = mod(outputs, **kwargs)
    return outputs
