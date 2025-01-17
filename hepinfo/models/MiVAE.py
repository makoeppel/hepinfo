import keras
import numpy as np
import tensorflow as tf
from keras.api import layers, ops
from keras.api.regularizers import L2
from sklearn.base import BaseEstimator
from squark.config import QuantizerConfigScope, global_config
from squark.layers import QDense as SQDense
from squark.utils.sugar import FreeEBOPs

from hepinfo.models.qkerasV3 import QActivation, QDense, quantized_sigmoid, BernoulliSampling
from hepinfo.models.QuantFlow import TQActivation, TQDense
from hepinfo.util import mutual_information_bernoulli_loss


# @keras.utils.register_keras_serializable()
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def __init__(self, latent_dims, **kwargs):
        super().__init__(**kwargs)
        self.latent_dims = latent_dims

    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal([self.latent_dims], dtype=tf.float32, name='epsilon0')
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# @keras.utils.register_keras_serializable()
class MiVAE(BaseEstimator, keras.Model):
    """
    Stochastically Quantized Variational Auto Endcoder which has a bernoulli
    activation at the latent layer to max/min the mutual information.
    """

    __module__ = 'Custom>MiVAE'

    def __init__(
        self,
        hidden_layers=None,
        activation='relu',
        use_s_quark=False,
        use_qkeras=False,
        use_quantflow=False,
        init_quantized_bits=32,
        input_quantized_bits='quantized_bits(16, 6, 0)',
        quantized_bits='quantized_bits(16, 6, 0, use_stochastic_rounding=True)',
        quantized_activation='quantized_relu(10, 6, use_stochastic_rounding=True, negative_slope=0.0)',
        latent_dims=64,
        kernel_regularizer=0.01,
        num_samples=10,
        bits_bernoulli_sigmoid=8,
        use_quantized_sigmoid=False,
        drop_out=0.2,
        use_batchnorm=False,
        beta_param=1,
        alpha=0.01,
        gamma=1,
        beta0=1e-7,
        batch_size=256,
        learning_rate=0.0001,
        learning_rate_decay_rate=1,
        learning_rate_decay_steps=1000,
        optimizer='Adam',
        epoch=60,
        verbose=0,
        patience=3,
        monitor='kl_loss',
        validation_size=0,
        run_eagerly=False,
        mi_loss=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # loss trackers
        self.total_loss_tracker = keras.metrics.Mean(name='total_loss')
        self.reconstruction_loss_tracker = keras.metrics.Mean(name='reconstruction_loss')
        self.kl_loss_tracker = keras.metrics.Mean(name='kl_loss')
        self.mi_loss_tracker = keras.metrics.Mean(name='mi_loss')
        self.bops_loss_tracker = keras.metrics.Mean(name='bops')
        self.ebops_loss_tracker = keras.metrics.Mean(name='ebops')

        # HP of the model
        self.hidden_layers = hidden_layers
        self.num_samples = num_samples
        self.bits_bernoulli_sigmoid = bits_bernoulli_sigmoid
        self.use_quantized_sigmoid = use_quantized_sigmoid
        self.latent_dims = latent_dims
        self.activation = activation
        self.use_qkeras = use_qkeras
        self.use_s_quark = use_s_quark
        self.use_quantflow = use_quantflow
        self.init_quantized_bits = init_quantized_bits
        self.input_quantized_bits = input_quantized_bits
        self.quantized_bits = quantized_bits
        self.quantized_activation = quantized_activation
        if drop_out > 0:
            self.kernel_regularizer = 0
        else:
            self.kernel_regularizer = kernel_regularizer
        self.drop_out = drop_out
        self.use_batchnorm = use_batchnorm
        self.beta_param = beta_param
        self.alpha = alpha
        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_decay_rate = learning_rate_decay_rate
        self.learning_rate_decay_steps = learning_rate_decay_steps
        self.optimizer = optimizer
        if optimizer == 'Adam':
            self.optimizer = keras.optimizers.Adam
        if optimizer == 'SGD':
            self.optimizer = keras.optimizers.SGD
        if optimizer == 'Nadam':
            raise NotImplementedError
        self.optimizer_name = optimizer
        self.epoch = epoch
        self.verbose = verbose
        self.patience = patience
        self.monitor = monitor
        self.validation_size = validation_size
        self.run_eagerly = run_eagerly
        self.inputshape = None
        self.mi_loss = mi_loss
        self.beta0 = beta0

    def train_step(self, data):
        # data = [x, s]
        x, s = data
        with tf.GradientTape() as tape:
            if self.mi_loss:
                z_mean, z_log_var, z, z_sample = self.encoder(x, training=True)
                reconstruction = self.decoder(z_sample, training=True)
            else:
                z_mean, z_log_var, z = self.encoder(x, training=True)
                reconstruction = self.decoder(z, training=True)

            reconstruction_loss = keras.losses.MeanSquaredError()(x, reconstruction)

            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = self.beta_param * tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            mi_loss = 0

            if self.mi_loss:
                mi_loss = self.gamma * mutual_information_bernoulli_loss(s, z)

            total_bops = 0
            total_bops_std = []
            if self.use_quantflow:
                for layer in self.encoder.layers:
                    if hasattr(layer, 'compute_bops'):
                        total_bops += layer.compute_bops()
                    if hasattr(layer, 'compute_bops_std'):
                        total_bops_std.append(layer.compute_bops_std())
                if type(self.alpha) is not str:
                    total_bops *= self.alpha

            total_ebops = sum([layer.ebops for layer in self.encoder.layers if hasattr(layer, 'ebops')])

            total_loss = reconstruction_loss + kl_loss + mi_loss + sum(self.encoder.losses)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.mi_loss_tracker.update_state(mi_loss)
        self.bops_loss_tracker.update_state(total_bops)
        self.ebops_loss_tracker.update_state(total_ebops)

        return {
            'loss': self.total_loss_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
            'kl_loss': self.kl_loss_tracker.result(),
            'mi_loss': self.mi_loss_tracker.result(),
            'bops': self.bops_loss_tracker.result(),
            'ebops': self.ebops_loss_tracker.result(),
        }

    def test_step(self, data):
        # data = [x, s]
        x, s = data
        if self.mi_loss:
            z_mean, z_log_var, z, z_sample = self.encoder(x, training=False)
            reconstruction = self.decoder(z_sample)
        else:
            z_mean, z_log_var, z = self.encoder(x, training=False)
            reconstruction = self.decoder(z)
        reconstruction_loss = keras.losses.MeanSquaredError()(x, reconstruction)

        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = self.beta_param * tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        mi_loss = 0

        if self.mi_loss:
            mi_loss = self.gamma * mutual_information_bernoulli_loss(s, z)

        total_loss = reconstruction_loss + kl_loss + mi_loss + sum(self.encoder.losses)

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.mi_loss_tracker.update_state(mi_loss)

        return {
            'loss': self.total_loss_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
            'kl_loss': self.kl_loss_tracker.result(),
            'mi_loss': self.mi_loss_tracker.result(),
        }

    def call(self, x, training=True):
        if self.mi_loss:
            z_mean, z_log_var, z, z_sample = self.encoder(x, training)
            reconstruction = self.decoder(z_sample)
        else:
            z_mean, z_log_var, z = self.encoder(x, training)
            reconstruction = self.decoder(z)
        return reconstruction

    def get_encoder(self):
        encoder_inputs = keras.Input(shape=self.inputshape)

        if self.use_qkeras:
            x = QActivation(self.input_quantized_bits)(encoder_inputs)
        if self.use_quantflow:
            x = TQActivation(self.inputshape, bits=self.init_quantized_bits, alpha=self.alpha)(encoder_inputs)
        if self.use_s_quark:
            x = encoder_inputs
            # iq_conf = QuantizerConfig(place='datalane', k0=1, fr=MonoL1(2e-6))
            # oq_conf = QuantizerConfig(place='datalane', k0=1, fr=MonoL1(1000))
            global_config['beta0'] = self.beta0
            scope0 = QuantizerConfigScope(place='all', k0=1, b0=4, i0=2, default_q_type='kbi', overflow_mode='sat_sym')
            scope1 = QuantizerConfigScope(place='datalane', k0=1, default_q_type='kif', overflow_mode='sat_sym', f0=2, i0=2)

        if not self.use_qkeras and not self.use_quantflow and not self.use_s_quark:
            x = encoder_inputs

        for i, layer in enumerate(self.hidden_layers):
            if self.use_qkeras:
                x = QDense(
                    layer,
                    kernel_initializer='glorot_uniform',
                    kernel_quantizer=self.quantized_bits,
                    bias_quantizer=self.quantized_bits,
                    activation=self.quantized_activation,
                )(x)
            elif self.use_quantflow:
                x = TQDense(layer, init_bits=self.init_quantized_bits, activation=self.activation, alpha=self.alpha)(x)
            elif self.use_s_quark:
                with scope0, scope1:
                    if i == 0:
                        x = SQDense(layer, activation=self.activation, name='input')(x)
                    elif i == len(self.hidden_layers) - 1:
                        x = SQDense(layer, activation=self.activation, name='output')(x)
                    else:
                        x = SQDense(layer, activation=self.activation)(x)
            else:
                x = layers.Dense(
                    layer,
                    activation=self.activation,
                    kernel_regularizer=L2(self.kernel_regularizer),
                    activity_regularizer=L2(self.kernel_regularizer),
                )(x)
            if self.use_batchnorm:
                x = layers.BatchNormalization()(x)
            if self.drop_out > 0:
                x = layers.Dropout(self.drop_out)(x)

        # setup latent layers
        if self.use_qkeras:
            z_mean = QDense(
                self.latent_dims,
                name='z_mean',
                kernel_initializer='glorot_uniform',
                kernel_quantizer=self.quantized_bits,
                bias_quantizer=self.quantized_bits,
            )(x)
            z_log_var = QDense(
                self.latent_dims,
                name='z_log_var',
                kernel_initializer='glorot_uniform',
                kernel_quantizer=self.quantized_bits,
                bias_quantizer=self.quantized_bits,
            )(x)
        elif self.use_s_quark:
            with scope0, scope1:
                z_mean = SQDense(self.latent_dims, activation='linear')(x)
                z_log_var = SQDense(self.latent_dims, activation='linear')(x)
        elif self.use_quantflow:
            z_mean = TQDense(
                self.latent_dims, name='z_mean', init_bits=self.init_quantized_bits, activation='linear', alpha=self.alpha
            )(x)
            z_log_var = TQDense(
                self.latent_dims, name='z_log_var', init_bits=self.init_quantized_bits, activation='linear', alpha=self.alpha
            )(x)
        else:
            z_mean = layers.Dense(
                self.latent_dims,
                name='z_mean',
                kernel_regularizer=L2(self.kernel_regularizer),
                activity_regularizer=L2(self.kernel_regularizer),
            )(x)
            z_log_var = layers.Dense(
                self.latent_dims,
                name='z_log_var',
                kernel_regularizer=L2(self.kernel_regularizer),
                activity_regularizer=L2(self.kernel_regularizer),
            )(x)

        z = Sampling(self.latent_dims)([z_mean, z_log_var])
        if self.mi_loss:
            z_sample = BernoulliSampling(
                self.num_samples,
                use_quantized=self.use_quantized_sigmoid,
                bits_bernoulli_sigmoid=self.bits_bernoulli_sigmoid,
                name='bernoulli',
            )(z)

            # build encoder
            encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z, z_sample], name='encoder')
        else:
            # build encoder
            encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
        if self.verbose > 0:
            encoder.summary()
        return encoder

    def get_decoder(self):
        latent_inputs = keras.Input(shape=(self.latent_dims,))

        # setup dense layers
        x = latent_inputs
        for layer in reversed(self.hidden_layers):
            x = layers.Dense(
                layer,
                activation=self.activation,
                kernel_regularizer=L2(self.kernel_regularizer),
                activity_regularizer=L2(self.kernel_regularizer),
            )(x)
            if self.use_batchnorm:
                x = layers.BatchNormalization()(x)
            if self.drop_out > 0:
                x = layers.Dropout(self.drop_out)(x)

        # last layer
        x = layers.Dense(
            self.inputshape[0],
            activation='sigmoid',
            kernel_regularizer=L2(self.kernel_regularizer),
            activity_regularizer=L2(self.kernel_regularizer),
        )(x)

        # build decoder
        decoder = keras.Model(latent_inputs, x, name='decoder')
        if self.verbose > 0:
            decoder.summary()
        return decoder

    def fit(self, x, s):
        # get input shape
        self.inputshape = [x.shape[1]]

        # fill up batch size NOTE: loss can have NaN values so batch_size should be not to small
        if len(x) % self.batch_size != 0:
            x = np.concatenate([x, x[: self.batch_size - len(x) % self.batch_size]])
            s = np.concatenate([s, s[: self.batch_size - len(s) % self.batch_size]])

        self.encoder = self.get_encoder()
        self.decoder = self.get_decoder()

        lr_schedule = keras.optimizers.schedules.InverseTimeDecay(
            self.learning_rate,
            decay_steps=self.learning_rate_decay_steps,
            decay_rate=self.learning_rate_decay_rate,
            staircase=False,
        )

        if type(self.optimizer) is str:
            if self.optimizer == 'Adam':
                self.optimizer = keras.optimizers.Adam
            if self.optimizer == 'SGD':
                self.optimizer = keras.optimizers.SGD
            if self.optimizer == 'Nadam':
                self.optimizer = keras.optimizers.Nadam

        self.compile(
            optimizer=self.optimizer(lr_schedule),
            run_eagerly=self.run_eagerly,
            jit_compile=False,
        )

        callback = [keras.callbacks.EarlyStopping(monitor=self.monitor, mode='min', patience=self.patience)]

        if self.use_s_quark:
            nan_terminate = keras.callbacks.TerminateOnNaN()
            ebops = FreeEBOPs()
            callback.append(nan_terminate)
            callback.append(ebops)

        history = super().fit(x, s, epochs=self.epoch, batch_size=self.batch_size, verbose=self.verbose, callbacks=callback)
        return history

    def score(self, x, y):
        if self.mi_loss:
            z_mean, z_log_var, z, z_sample = self.encoder(x, training=False)
            reconstruction = self.decoder(z_sample)
        else:
            z_mean, z_log_var, z = self.encoder(x, training=False)
            reconstruction = self.decoder(z)
        reconstruction_loss = keras.losses.MeanSquaredError()(x, reconstruction)
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = self.beta_param * tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        mi_loss = 0
        if self.mi_loss:
            mi_loss = self.gamma * mutual_information_bernoulli_loss(y, z)
        total_loss = reconstruction_loss + kl_loss + mi_loss
        return total_loss.numpy()

    def score_vector(self, x):
        if self.mi_loss:
            z_mean, z_log_var, z, _ = self.encoder(x, training=False)
        else:
            z_mean, z_log_var, z = self.encoder(x, training=False)
        reconstruction = self.decoder(z)
        reconstruction_loss = keras.losses.mse(x, reconstruction)
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = self.beta_param * tf.reduce_sum(kl_loss, axis=1)
        total_loss = reconstruction_loss + kl_loss
        losses = [reconstruction_loss.numpy(), kl_loss.numpy(), total_loss.numpy()]
        return losses

    def get_mean(self, x):
        return self.encoder(x, training=False)[0]

    def get_sigma(self, x):
        return self.encoder(x, training=False)[1]

    def get_latentspace(self, x):
        return self.encoder(x, training=False)[2]

    def get_params(self, deep=True):
        out = dict()
        for key in self._get_param_names():
            cur_key = key
            if key == 'optimizer':
                cur_key = 'optimizer_name'
            out[key] = getattr(self, cur_key)
        return out

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def to_dict(self):
        """
        Return a dictionary representation of the object while dropping the tensorflow stuff.
        Useful to keep track of hyperparameters at the experiment level.
        """
        d = dict(vars(self))

        for key in dict(vars(keras.Model)).keys():
            try:
                d.pop(key)
            except KeyError:
                pass

        drop_key = []
        for key in d.keys():
            if key.startswith('_'):
                drop_key.append(key)

        for key in drop_key:
            try:
                d.pop(key)
            except KeyError:
                pass

        return d
