The weight file uses the ori model, and you need to change the input format in the model to make it work.

like:

def __call__(self, inputs, is_training) -> Dict[str, tf.Tensor]:
    
    proseq_minus = inputs[1][0]
    proseq_plus = inputs[1][1]
    proseq_minus_plus = inputs[1][2]
    inputs = inputs[0]

    outputs = self.conv(inputs, training=is_training)

    out_proseq_minus = self.conv_minus(proseq_minus, training=is_training)

    out_proseq_plus = self.conv_plus(proseq_plus, training=is_training)

    out_proseq_minus_plus = self._conv_minus_plus(proseq_minus_plus, training=is_training)

    outputs = tf.concat([outputs, out_proseq_minus, out_proseq_plus, out_proseq_minus_plus], axis=-1)

    outputs = self.concat_x_proseq(outputs, training=is_training)
    
    outputs = self.trunk(outputs, training=is_training)

    outputs = self.heads['human'](outputs)

    return outputs
