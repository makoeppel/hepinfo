import tensorflow as tf


def quantize(x, bits):
    """Quantize a tensor to a given number of bits."""
    scale = tf.pow(2.0, bits - 1)
    quantized = tf.round(x * scale) / scale
    return x + tf.stop_gradient(quantized - x)  # STE approximation

def quantized_sigmoid(x, bits):
    """
    Quantized version of the sigmoid activation function.
    
    Args:
        x (tf.Tensor): Input tensor.
        bits (int): Number of bits to quantize the output to.
        
    Returns:
        tf.Tensor: Quantized sigmoid output.
    """
    scale = tf.pow(2.0, bits) - 1  # Number of discrete levels
    sigmoid = tf.math.sigmoid(x)
    quantized = tf.round(sigmoid * scale) / scale  # Quantize to discrete levels
    return sigmoid + tf.stop_gradient(quantized - sigmoid)  # STE approximation

def quantized_relu(x, bits, max_value):
    """
    Quantized version of the ReLU activation function.
    
    Args:
        x (tf.Tensor): Input tensor.
        bits (int): Number of bits to quantize the output to.
        max_value (float): Maximum value for clipping (e.g., 6.0 for ReLU6).

    Returns:
        tf.Tensor: Quantized ReLU output.
    """
    scale = tf.pow(2.0, bits) - 1  # Number of discrete levels
    relu = tf.clip_by_value(x, 0.0, max_value)  # Clip to [0, max_value]
    quantized = tf.round(relu * scale / max_value) * max_value / scale  # Quantize to discrete levels
    return relu + tf.stop_gradient(quantized - relu)  # STE approximation

class TQActivation(tf.keras.layers.Layer):
    def __init__(self, input_shape, bits, min_bits=1, max_bits=8, clip_min=-1.0, clip_max=1.0, **kwargs):
        """
        Trainable Quantized Activation Layer.

        Args:
            min_bits (int): Minimum bit width.
            max_bits (int): Maximum bit width.
            clip_min (float): Minimum value for clipping activations.
            clip_max (float): Maximum value for clipping activations.
        """
        super(TQActivation, self).__init__(**kwargs)
        self.input_shape = input_shape
        self.bits = bits
        self.min_bits = min_bits
        self.max_bits = max_bits
        self.clip_min = clip_min
        self.clip_max = clip_max

    def build(self, input_shape):
        # Trainable bit width
        self.activation_bits = self.add_weight(
            name="bits",
            shape=(),
            initializer=tf.keras.initializers.Constant(self.bits),
            trainable=True,
        )

    def call(self, inputs):
        # Quantize the activations using the current bit width
        inputs = tf.clip_by_value(inputs, self.clip_min, self.clip_max)
        quantized_output = quantize(inputs, self.activation_bits)
        return quantized_output

    def compute_output_shape(self, input_shape):
        # The output shape is the same as the input shape
        return input_shape

    def compute_bops(self):
        """
        Compute the BOPs for this layer.
        Layer BOPs = (num_activations x activation_bits)

        Args:
            num_activations (int): Number of activations in the layer output.

        Returns:
            float: BOPs for this layer.
        """
        num_activations = tf.size(self.input_shape)

        # Compute BOPs contributions
        activation_bops = tf.cast(num_activations, tf.float32) * self.activation_bits

        # Total BOPs for the layer
        total_bops = activation_bops
        return total_bops

    def get_config(self):
        config = super(TQActivation, self).get_config()
        config.update({
            "min_bits": self.min_bits,
            "max_bits": self.max_bits,
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
        })
        return config

class TQDense(tf.keras.layers.Layer):
  def __init__(self, units, activation="linear", init_bits=16, min_bits=1, max_bits=32, max_value_relu=32, alpha=0.01, **kwargs):
      """
      A quantized Dense layer with trainable bit widths for weights, activations, and biases.
      
      Args:
          units (int): Number of output units.
          min_bits (int): Minimum bit width allowed.
          max_bits (int): Maximum bit width allowed.
          kwargs: Additional arguments for the Layer superclass.
      """
      super(TQDense, self).__init__(**kwargs)
      self.units = units
      self.activation = activation
      self.init_bits = init_bits
      self.min_bits = min_bits
      self.max_bits = max_bits
      self.max_value_relu = max_value_relu
      self.alpha = alpha

  def build(self, input_shape):
      # Initialize weights, biases, and their bit widths
      self.kernel = self.add_weight(
          name="kernel",
          shape=(input_shape[-1], self.units),
          initializer="he_normal",
          trainable=True,
      )
      self.bias = self.add_weight(
          name="bias",
          shape=(self.units,),
          initializer="zeros",
          trainable=True,
      )

      # Trainable bit widths
      self.weight_bits = self.add_weight(
          name="weight_bits",
          shape=(),
          initializer=tf.keras.initializers.Constant(self.init_bits),
          trainable=True,
          dtype=tf.float32,
      )
      self.activation_bits = self.add_weight(
          name="activation_bits",
          shape=(),
          initializer=tf.keras.initializers.Constant(self.init_bits),
          trainable=True,
          dtype=tf.float32,
      )
      self.bias_bits = self.add_weight(
          name="bias_bits",
          shape=(),
          initializer=tf.keras.initializers.Constant(self.init_bits),
          trainable=True,
          dtype=tf.float32,
      )

      # # Add regularization to the loss
      # self.add_loss(self.alpha * (
      #     tf.reduce_sum(self.weight_bits) +
      #     tf.reduce_sum(self.activation_bits) +
      #     tf.reduce_sum(self.bias_bits)
      # ))

  def compute_bops(self):
      """
      Compute the BOPs for this layer.
      Layer BOPs = (num_weights x weight_bits) + (num_activations x activation_bits) + (num_biases x bias_bits)

      Args:
          num_activations (int): Number of activations in the layer output.

      Returns:
          float: BOPs for this layer.
      """
      num_weights = tf.size(self.kernel)  # Total weights
      num_biases = tf.size(self.bias)  # Total biases
      num_activations = tf.size(self.units)

      # Compute BOPs contributions
      weight_bops = tf.cast(num_weights, tf.float32) * self.weight_bits
      activation_bops = tf.cast(num_activations, tf.float32) * self.activation_bits
      bias_bops = tf.cast(num_biases, tf.float32) * self.bias_bits

      # Total BOPs for the layer
      total_bops = weight_bops + activation_bops + bias_bops
      return total_bops

  def call(self, inputs):
      # Quantize kernel (weights)
      unsigned_weight_bits = tf.clip_by_value(self.weight_bits, self.min_bits, self.max_bits)
      quantized_kernel = quantize(self.kernel, unsigned_weight_bits)

      # Compute the output
      output = tf.matmul(inputs, quantized_kernel)

      # Quantize bias
      unsigned_bias_bits = tf.clip_by_value(self.bias_bits, self.min_bits, self.max_bits)
      quantized_bias = quantize(self.bias, unsigned_bias_bits)
      output = tf.nn.bias_add(output, quantized_bias)

      # Quantize activations
      unsigned_activation_bits = tf.clip_by_value(self.activation_bits, self.min_bits, self.max_bits)
      if self.activation == "sigmoid":
        output = quantized_sigmoid(output, unsigned_activation_bits)

      if self.activation == "relu":
        output = quantized_relu(output, unsigned_activation_bits, self.max_value_relu)

      if self.activation == "linear":
        output = quantize(output, unsigned_activation_bits)

      return tf.convert_to_tensor(output)

  def get_config(self):
      config = super(TQDense, self).get_config()
      config.update({
          "units": self.units,
          "min_bits": self.min_bits,
          "max_bits": self.max_bits,
          "max_bits": self.max_bits,
      })
      return config
