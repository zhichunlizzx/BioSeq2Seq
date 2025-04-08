import tensorflow as tf
from model_function import attention_model
from model_function.tools import Residual, SoftPlus, Sequential, GELU, pooling_module, exponential_linspace_int, TargetLengthCrop1D
from typing import Dict


class FEModel(tf.keras.Model):
  """Main model."""

  def __init__(self,
               channels: int = 1536,
               num_transformer_layers: int = 11,
               num_heads: int = 8,
               output_channels = 1,
               target_length=896,
               pooling_type: str = 'max',
               name: str = 'FEModel'):

    super(FEModel, self).__init__(name=name)

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

    self.concat_x_proseq_0 = Sequential([tf.keras.layers.Dense(channels)], name='concat_x_proseq_0')
    self.concat_x_proseq_1 = Sequential([tf.keras.layers.Dense(channels)], name='concat_x_proseq_1')

    # Transformer.
    def transformer_mlp():
      return Sequential([
          tf.keras.layers.LayerNormalization(axis=-1, scale=True, center=True, gamma_initializer=tf.keras.initializers.Ones()),
          tf.keras.layers.Dense(channels * 2),
          tf.keras.layers.Dropout(dropout_rate),
          tf.keras.layers.ReLU(),
          tf.keras.layers.Dense(channels),
          tf.keras.layers.Dropout(dropout_rate)], name='mlp')

    transformer_act = Sequential([
        Sequential([
            Residual(Sequential([
                tf.keras.layers.LayerNormalization(axis=-1,
                              scale=True, center=True,
                              gamma_initializer=tf.keras.initializers.Ones()),

                attention_model.MultiheadAttention(**whole_attention_kwargs,
                                                    name=f'attention_{i}'),
                tf.keras.layers.Dropout(dropout_rate)], name='mha')),
            Residual(transformer_mlp())], name=f'transformer_block_{i}')
        for i in range(num_transformer_layers)], name='transformer_act')
    
    transformer_inact = Sequential([
        Sequential([
            Residual(Sequential([
                tf.keras.layers.LayerNormalization(axis=-1,
                              scale=True, center=True,
                              gamma_initializer=tf.keras.initializers.Ones()),

                attention_model.MultiheadAttention(**whole_attention_kwargs,
                                                    name=f'attention_{i}'),
                tf.keras.layers.Dropout(dropout_rate)], name='mha')),
            Residual(transformer_mlp())], name=f'transformer_block_{i}')
        for i in range(num_transformer_layers)], name='transformer_inact')
    
    crop_final = TargetLengthCrop1D(target_length, name='target_input')

    # self.crop_seq = TargetLengthCrop1D(target_length, name='target_input')

    # self.connect_parameter = tf.Variable(1, trainable=True, dtype=tf.float32)

    final_pointwise_act = Sequential([
        conv_block(channels * 2, 1),
        tf.keras.layers.Dropout(dropout_rate / 8),
        GELU()], name='final_pointwise_act')
    
    final_pointwise_inact = Sequential([
        conv_block(channels * 2, 1),
        tf.keras.layers.Dropout(dropout_rate / 8),
        GELU()], name='final_pointwise_inact')

    # self._initializer = tf.keras.initializers.VarianceScaling(scale=2.0)
    # self.var = tf.Variable(self._initializer([1], dtype=tf.float32))
    
    self._conv = Sequential([stem,
                            conv_tower],
                            name = 'conv_block_conv')
    
    self._conv_seq = Sequential([stem_seq,
                            conv_tower_seq],
                            name = 'conv_block_seq')

    self._trunk_act = Sequential([transformer_act,
                            crop_final,
                            ],
                             name='_trunk_act')
    
    self._trunk_inact = Sequential([transformer_inact,
                            crop_final,
                            ],
                             name='_trunk_inact')


    trunk_name_scope.__exit__(None, None, None)

    with tf.name_scope('heads'):
      self._heads_act = Sequential([final_pointwise_act,
                                tf.keras.layers.Dense(units=2), SoftPlus()
                                ], name=f'head_act')
      
      self._heads_inact = Sequential([final_pointwise_inact,
                                tf.keras.layers.Dense(units=2), SoftPlus()
                                ], name=f'head_inact')

  
  @property
  def conv(self):
    return self._conv

  @property
  def conv_seq(self):
    return self._conv_seq

  @property
  def trunk_act(self):
    return self._trunk_act
  
  @property
  def trunk_inact(self):
    return self._trunk_inact

  @property
  def heads_act(self):
    return self._heads_act

  @property
  def heads_inact(self):
    return self._heads_inact

  def __call__(self, inputs, is_training) -> Dict[str, tf.Tensor]:
    input_dna_encoding = inputs[0]
    input_seq_feature = inputs[1]

    out_dna = self.conv(input_dna_encoding, training=is_training)

    out_seq = self.conv_seq(input_seq_feature, training=is_training)

    outputs = tf.concat([out_dna, out_seq], axis=-1)
    outputs_0 = self.concat_x_proseq_0(outputs, training=is_training)
    outputs_1 = self.concat_x_proseq_1(outputs, training=is_training)
    
    outputs_act = self.trunk_act(outputs_0, training=is_training)
    outputs_act = self.heads_act(outputs_act, training=is_training)

    outputs_inact = self.trunk_inact(outputs_1, training=is_training)
    outputs_inact = self.heads_inact(outputs_inact, training=is_training)

    outputs = tf.concat([outputs_act, outputs_inact], axis=-1)

    return outputs