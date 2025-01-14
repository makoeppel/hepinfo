import random
import tensorflow as tf


def round(x, round_type):
    if round_type == 0:  # standard round
        return tf.floor(x + 0.5)
    if round_type == 1:  # stochastic round
        _floor = tf.floor(x)
        noise = tf.random.uniform(tf.shape(x))
        return tf.where(noise < x - _floor, _floor + 1, _floor)

def quantize(x, bits, round_type):
    """Quantize a tensor to a given number of bits."""
    scale = tf.pow(2.0, bits) - 1
    quantized = round(x * scale, round_type) / scale
    return x + tf.stop_gradient(quantized - x)  # STE approximation

def quantized_sigmoid(x, bits, round_type):
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
    quantized = round(sigmoid * scale, round_type) / scale  # Quantize to discrete levels
    return sigmoid + tf.stop_gradient(quantized - sigmoid)  # STE approximation

def quantized_relu(x, bits, max_value, round_type):
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
    quantized = round(relu * scale / max_value, round_type) * max_value / scale  # Quantize to discrete levels
    return relu + tf.stop_gradient(quantized - relu)  # STE approximation

class TQActivation(tf.keras.layers.Layer):
    def __init__(self, input_shape, bits, min_bits=0, max_bits=32, clip_min=-1.0, clip_max=1.0, round_type=1, alpha="random", **kwargs):
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
        if type(bits) == str and bits == "random":
            self.bits = random.randrange(16, 32)
        self.min_bits = min_bits
        self.max_bits = max_bits
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.round_type = round_type
        self.alpha = alpha
        if type(alpha) == str and alpha == "random":
            self.alpha = random.uniform(0, 1)

    def build(self, input_shape):
        # Trainable bit width
        self.activation_bits = self.add_weight(
            name="bits",
            shape=(input_shape[-1],),
            initializer=tf.keras.initializers.Constant(self.bits),
            trainable=True,
        )

    def call(self, inputs):
        # Quantize the activations using the current bit width
        inputs = tf.clip_by_value(inputs, self.clip_min, self.clip_max)
        quantized_output = quantize(inputs, self.activation_bits, self.round_type)
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

        activation_bops = tf.reduce_sum(self.activation_bits)

        # Total BOPs for the layer
        total_bops = activation_bops
        return self.alpha * total_bops

    def compute_bops_std(self):
        activation_bops_std = tf.math.reduce_std(self.activation_bits)
        total_bops_std = activation_bops_std
        return total_bops_std

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
  def __init__(self, units, activation="linear", init_bits=16, min_bits=0, max_bits=32, max_value_relu=32, alpha="random", round_type=1, **kwargs):
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
      self.init_bits_weight = init_bits
      self.init_bits_activation = init_bits
      self.init_bits_bias = init_bits
      if type(init_bits) == str and init_bits == "random":
        # TODO: set range of bits
        self.init_bits_weight = random.randrange(8, 32)
        self.init_bits_activation = random.randrange(8, 32)
        self.init_bits_bias = random.randrange(8, 32)
      self.min_bits = min_bits
      self.max_bits = max_bits
      self.max_value_relu = max_value_relu
      self.alpha = alpha
      if type(alpha) == str and alpha == "random":
        self.alpha = random.uniform(0, 1)
      self.round_type = round_type

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
          shape=(input_shape[-1], self.units),
          initializer=tf.keras.initializers.Constant(self.init_bits_weight),
          trainable=True,
          dtype=tf.float32,
      )
      self.activation_bits = self.add_weight(
          name="activation_bits",
          shape=(self.units,),
          initializer=tf.keras.initializers.Constant(self.init_bits_activation),
          trainable=True,
          dtype=tf.float32,
      )
      self.bias_bits = self.add_weight(
          name="bias_bits",
          shape=(self.units,),
          initializer=tf.keras.initializers.Constant(self.init_bits_bias),
          trainable=True,
          dtype=tf.float32,
      )

  def compute_bops(self):
    """
    Compute the BOPs for this layer.
    Layer BOPs = (num_weights x weight_bits) + (num_activations x activation_bits) + (num_biases x bias_bits)

    Args:
        num_activations (int): Number of activations in the layer output.

    Returns:
        float: BOPs for this layer.
    """

    # Compute individual weight BOPs
    weight_bops = tf.reduce_sum(self.weight_bits)  # Sum trainable bit-widths per weight
    bias_bops = tf.reduce_sum(self.bias_bits)
    activation_bops = tf.reduce_sum(self.activation_bits)

    # Total BOPs for the layer
    total_bops = weight_bops + bias_bops + activation_bops
    return total_bops

  def compute_bops_std(self):
    weight_bops_std = tf.math.reduce_std(self.weight_bits)
    bias_bops_std = tf.math.reduce_std(self.bias_bits)
    activation_bops_std = tf.math.reduce_std(self.activation_bits)
    total_bops_std = [weight_bops_std, bias_bops_std, activation_bops_std]
    return tf.reduce_mean(total_bops_std)

  def call(self, inputs):
      # Quantize kernel (weights)
      unsigned_weight_bits = tf.clip_by_value(self.weight_bits, self.min_bits, self.max_bits)
      quantized_kernel = quantize(self.kernel, unsigned_weight_bits, self.round_type)

      # Compute the output
      output = tf.matmul(inputs, quantized_kernel)

      # Quantize bias
      unsigned_bias_bits = tf.clip_by_value(self.bias_bits, self.min_bits, self.max_bits)
      quantized_bias = quantize(self.bias, unsigned_bias_bits, self.round_type)
      output = tf.nn.bias_add(output, quantized_bias)

      # Quantize activations
      unsigned_activation_bits = tf.clip_by_value(self.activation_bits, self.min_bits, self.max_bits)
      if self.activation == "sigmoid":
        output = quantized_sigmoid(output, unsigned_activation_bits, self.round_type)

      if self.activation == "relu":
        output = quantized_relu(output, unsigned_activation_bits, self.max_value_relu, self.round_type)

      if self.activation == "linear":
        output = quantize(output, unsigned_activation_bits, self.round_type)

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
