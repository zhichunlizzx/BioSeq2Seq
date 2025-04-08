import tensorflow as tf
from typing import Dict
from model_function import attention_model
from model_function.tools import Residual, Sequential, GELU, pooling_module, exponential_linspace_int, TargetLengthCrop1D


class TFModel(tf.keras.Model):
  """Main model."""

  def __init__(self,
               channels: int = 1536,
               num_transformer_layers: int = 9,
               num_heads: int = 8,
               target_length=896,
               pooling_type: str = 'max',
               output_channels = None,
               name: str = 'TFModel'):

    super(TFModel, self).__init__(name=name)
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
                                    for j in range(num_transformer_layers)], name='transformer')
        
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
                                    for j in range(num_transformer_layers)], name='transformer')
        
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
                                    for j in range(num_transformer_layers)], name='transformer')
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