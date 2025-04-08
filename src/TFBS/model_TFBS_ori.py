import tensorflow as tf
import numpy as np
from typing import Any, Callable, Dict, Optional, Text, Union, Iterable
import inspect
from model_function import attention_model


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


class Dhit2(tf.keras.Model):
  """Main model."""

  def __init__(self,
               channels: int = 1536,
               num_transformer_layers: int = 11,
               num_heads: int = 8,
               output_channels = 1,
               target_length=896,
               pooling_type: str = 'max',
               name: str = 'dhit2'):

    super(Dhit2, self).__init__(name=name)

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

    stem_seq = Sequential([
        tf.keras.layers.Conv1D(filters=channels // 2, kernel_size=15, padding="same"),
        Residual(conv_block(channels // 2, 1, name='pointwise_conv_block_seq')),
        pooling_module(pooling_type, pool_size=2),
    ], name='stem_seq')


    filter_list = exponential_linspace_int(start=channels // 2, end=channels,
                                           num=6, divisible_by=64)
    # filter_list = 
    conv_tower = Sequential([
        Sequential([
            conv_block(num_filters, 5),
            Residual(conv_block(num_filters, 1, name='pointwise_conv_block_conv')),
            pooling_module(pooling_type, pool_size=2),
            ],
                   name=f'conv_tower_block_{i}')
        for i, num_filters in enumerate(filter_list)], name='conv_tower')

    conv_tower_seq = Sequential([
        Sequential([
            conv_block(num_filters, 5),
            Residual(conv_block(num_filters, 1, name='pointwise_conv_block_seq')),
            pooling_module(pooling_type, pool_size=2),
            ],
                   name=f'conv_tower_seq_{i}')
        for i, num_filters in enumerate(filter_list)], name='conv_tower_seq')
    
    # Transformer.
    def transformer_mlp():
      return Sequential([
          tf.keras.layers.LayerNormalization(axis=-1, scale=True, center=True, gamma_initializer=tf.keras.initializers.Ones()),
          tf.keras.layers.Dense(channels * 2),
          tf.keras.layers.Dropout(dropout_rate),
          tf.keras.layers.ReLU(),
          tf.keras.layers.Dense(channels),
          tf.keras.layers.Dropout(dropout_rate)], name='mlp')
    
    crop_final = TargetLengthCrop1D(target_length, name='target_input')
    
    self._conv = Sequential([stem,
                            conv_tower
                            ],
                            name = 'conv_block_conv')
    
    self._conv_seq = Sequential([stem_seq,
                            conv_tower_seq
                            ],
                            name = 'conv_block_seq')
    
    trunk_name_scope.__exit__(None, None, None)

    self.num_corr_tf = 9
    # channel_list = [1, 2]
    with tf.name_scope('heads'):
        transformer_0 = Sequential([
                                Sequential([
                                    Residual(Sequential([
                                        tf.keras.layers.LayerNormalization(axis=-1,
                                                    scale=True, center=True,
                                                    gamma_initializer=tf.keras.initializers.Ones()),

                                        attention_model.MultiheadAttention(**whole_attention_kwargs,
                                                                            name=f'attention_{j}'),
                                        tf.keras.layers.Dropout(dropout_rate)], name='mha')),
                                    Residual(transformer_mlp())], name=f'transformer_block_{j}')
                                    for j in range(9)], name='transformer')
        
        transformer_1 = Sequential([
                                Sequential([
                                    Residual(Sequential([
                                        tf.keras.layers.LayerNormalization(axis=-1,
                                                    scale=True, center=True,
                                                    gamma_initializer=tf.keras.initializers.Ones()),

                                        attention_model.MultiheadAttention(**whole_attention_kwargs,
                                                                            name=f'attention_{j}'),
                                        tf.keras.layers.Dropout(dropout_rate)], name='mha')),
                                    Residual(transformer_mlp())], name=f'transformer_block_{j}')
                                    for j in range(9)], name='transformer')
        
        self._heads_tf0 = Sequential([
                                tf.keras.layers.Dense(channels),
                                transformer_0,
                                crop_final,
                                tf.keras.layers.Dense(units=channels),
                                Sequential([conv_block(channels * 2, 1),
                                            tf.keras.layers.Dropout(dropout_rate / 8),
                                            GELU(),
                                            conv_block(channels * 4, 1),
                                            tf.keras.layers.Dropout(dropout_rate / 8),
                                            GELU(),
                                            conv_block(channels * 2, 1),
                                            tf.keras.layers.Dropout(dropout_rate / 8),
                                            GELU()], name='final_pointwise'),
                                tf.keras.layers.Dense(units=73),
                                ], name='head_tf0')
        
        self._heads_tf1 = Sequential([
                                tf.keras.layers.Dense(channels),
                                transformer_1,
                                crop_final,
                                tf.keras.layers.Dense(units=channels),
                                Sequential([conv_block(channels * 2, 1),
                                            tf.keras.layers.Dropout(dropout_rate / 8),
                                            GELU(),
                                            conv_block(channels * 4, 1),
                                            tf.keras.layers.Dropout(dropout_rate / 8),
                                            GELU(),
                                            conv_block(channels * 2, 1),
                                            tf.keras.layers.Dropout(dropout_rate / 8),
                                            GELU()], name='final_pointwise'),
                                tf.keras.layers.Dense(units=8),
                                ], name='head_tf1')
        
        
        for i in range(self.num_corr_tf):
            transformer = Sequential([
                                Sequential([
                                    Residual(Sequential([
                                        tf.keras.layers.LayerNormalization(axis=-1,
                                                    scale=True, center=True,
                                                    gamma_initializer=tf.keras.initializers.Ones()),

                                        attention_model.MultiheadAttention(**whole_attention_kwargs,
                                                                            name=f'attention_{j}'),
                                        tf.keras.layers.Dropout(dropout_rate)], name='mha')),
                                    Residual(transformer_mlp())], name=f'transformer_block_{j}')
                                    for j in range(9)], name='transformer')
            setattr(
                self,
                f"_heads_tf_{i}",
                Sequential([
                            tf.keras.layers.Dense(channels),
                            transformer,
                            crop_final,
                            tf.keras.layers.Dense(units=channels),
                            Sequential([conv_block(channels * 2, 1),
                                        tf.keras.layers.Dropout(dropout_rate / 8),
                                        GELU(),
                                        conv_block(channels * 4, 1),
                                        tf.keras.layers.Dropout(dropout_rate / 8),
                                        GELU(),
                                        conv_block(channels * 2, 1),
                                        tf.keras.layers.Dropout(dropout_rate / 8),
                                        GELU()
                                        ], name='final_pointwise'),
                            tf.keras.layers.Dense(units=1),
                            ], name='head_tf_%d' % i)
            )
                                        
  @property
  def conv(self):
    return self._conv

  @property
  def conv_seq(self):
    return self._conv_seq


  def __call__(self, inputs, is_training) -> Dict[str, tf.Tensor]:
    input_dna_encoding = inputs[0]
    input_seq_feature = inputs[1]

    out_dna = self.conv(input_dna_encoding, training=is_training)
    out_seq = self.conv_seq(input_seq_feature, training=is_training)

    outputs = tf.concat([out_dna, out_seq], axis=-1)

    outputs_0 = self._heads_tf0(outputs, training=is_training)
    outputs_1 = self._heads_tf1(outputs, training=is_training)

    outputs_tf_list = []
    for i in range(self.num_corr_tf):
        setattr(
           self,
           f"outputs_tf_{i}",
            getattr(self, f"_heads_tf_{i}", None)(outputs, training=is_training)
        )
        outputs_tf_list.append(getattr(self, f"outputs_tf_{i}", None))

    outputs_2 = tf.concat(outputs_tf_list, axis=-1)

    outputs_TF = tf.concat([outputs_0, outputs_1, outputs_2], axis=-1)

    outputs_TF = tf.sigmoid(outputs_TF)

    return outputs_TF