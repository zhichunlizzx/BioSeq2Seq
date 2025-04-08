import tensorflow as tf
from typing import Dict
from model_function import attention_model
from model_function.tools import Residual, SoftPlus, Sequential, GELU, pooling_module, exponential_linspace_int, TargetLengthCrop1D

class HMModel(tf.keras.Model):
  """Main model."""

  def __init__(self,
               channels: int = 1536,
               num_transformer_layers: int = 11,
               num_heads: int = 8,
               output_channels = 1,
               target_length=896,
               pooling_type: str = 'attention',
               name: str = 'HMModel'):
    
    super(HMModel, self).__init__(name=name)
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

    stem_roseq = Sequential([
        tf.keras.layers.Conv1D(filters=channels // 2, kernel_size=15, padding="same"),
        Residual(conv_block(channels // 2, 1, name='pointwise_conv_block_roseq')),
        pooling_module(pooling_type, pool_size=2),
    ], name='stem_roseq')


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

    conv_tower_roseq = Sequential([
        Sequential([
            conv_block(num_filters, 5),
            Residual(conv_block(num_filters, 1, name='pointwise_conv_block_roseq')),
            pooling_module(pooling_type, pool_size=2),
            ],
                   name=f'conv_tower_block_minus_{i}')
        for i, num_filters in enumerate(filter_list)], name='conv_tower_roseq')

    self.concat_x_proseq = Sequential([tf.keras.layers.Dense(channels)], name='concat_x_proseq')

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

    final_pointwise = Sequential([
        conv_block(channels * 2, 1),
        tf.keras.layers.Dropout(dropout_rate / 8),
        GELU()], name='final_pointwise')
    
    self._conv = Sequential([stem,
                            conv_tower],
                            name = 'conv_dna')
    
    self._conv_roseq = Sequential([stem_roseq,
                            conv_tower_roseq],
                            name = 'conv_roseq')


    self._trunk = Sequential([transformer,
                            crop_final,
                            final_pointwise],
                             name='trunk')
    

    trunk_name_scope.__exit__(None, None, None)

    with tf.name_scope('heads'):  
      self._head = Sequential([tf.keras.layers.Dense(units=output_channels), SoftPlus()], name=f'head')
         

  @property
  def conv_dna(self):
    return self._conv

  @property
  def conv_roseq(self):
    return self._conv_roseq

  @property
  def trunk(self):
    return self._trunk

  @property
  def head(self):
    return self._head

  def __call__(self, inputs, is_training) -> Dict[str, tf.Tensor]:

    inputs_dna = inputs[0]
    inputs_roseq = inputs[1]

    outputs_dna = self.conv_dna(inputs_dna, training=is_training)
    out_roseq = self._conv_roseq(inputs_roseq, training=is_training)

    outputs = tf.concat([outputs_dna, out_roseq], axis=-1)

    outputs = self.concat_x_proseq(outputs, training=is_training)
    
    outputs = self.trunk(outputs, training=is_training)

    outputs = self.head(outputs, training=is_training)

    return outputs
