import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.base import BaseEstimator


class BernoulliSampling(tf.keras.layers.Layer):

    def __init__(self, num_samples, name=None, std=1, temperature=6.0, **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_samples = num_samples
        self.std = std
        self.temperature = temperature

    def call(self, inputs):

        # convert inputs to sigmoid to get probablity for bernoulli
        p = tf.keras.backend.sigmoid(self.temperature * inputs / self.std)

        # sample num_samples times from a bernoulli
        out = tf.zeros(tf.shape(inputs))
        for i in range(self.num_samples):
            r = tf.random.uniform(tf.shape(inputs))
            q = tf.sign(p - r)
            q += (1.0 - tf.abs(q))
            q = (q + 1.0) / 2.0
            out += q

        # output is mean of stochastic sampling with straight through gradient
        out = inputs + tf.stop_gradient(-inputs + out / self.num_samples)

        return out

#@tf.keras.utils.register_keras_serializable()
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def __init__(self, latent_dims, **kwargs):
        super().__init__(**kwargs)
        self.latent_dims = latent_dims

    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal([self.latent_dims], dtype=tf.float32, name='epsilon0')
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


#@tf.keras.utils.register_keras_serializable()
class MiVAE(BaseEstimator, keras.Model):
    """
        Stochastically Quantized Variational Auto Endcoder which has a bernoulli
        activation at the latent layer to max/min the mutual infromation.
    """
    __module__ = "Custom>BinaryVAE"

    def __init__(self,
        hidden_layers=[64, 32],
        activation='relu',
        latent_dims=64,
        #kernel_regularizer=0.00001,
        kernel_regularizer=0.01,
        num_samples=10,
        #drop_out=0,
        drop_out=0.2,
        #Baixo beta PARA QUE A KL LOSS IMPORTE MENOS!
        #beta_param=0.0001,
        beta_param=0.01,
        gamma=1,
        #necesito este batch size para que se minimice ben a KL, se o subo comeza a subir a KL a épocas altas
        batch_size=2048,
        #learning_rate=0.0001,
        learning_rate=0.0001,
        learning_rate_decay_rate=1,
        learning_rate_decay_steps=1000,
        optimizer="Adam",
        epoch=60,
        #de normal son 20 épocas
        verbose=0,
        validation_size=0,
        run_eagerly=False,
        quantize=False,
        #quantize=True, probo a ver que onda ao non cuantizar
        **kwargs
    ):
        

        #PROBABLY A GOOD SET OF HPS:
        #optimizer': 'Adam', 'learning_rate': 0.0001, 'latent_dims': 16, 'kernel_regularizer': 0.001, 'hidden_layers': [128, 64, 32], 'gamma': 1, 'epoch': 20, 'beta_param': 0.001, 'batch_size': 2048, 'activation': 'relu'}

        super().__init__(**kwargs)

        # loss trackers
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.mi_loss_tracker = keras.metrics.Mean(name="mi_loss")

        # HP of the model
        self.hidden_layers = hidden_layers
        self.num_samples = num_samples
        self.latent_dims = latent_dims
        self.activation = activation
        self.kernel_regularizer = kernel_regularizer
        self.drop_out = drop_out
        self.beta_param = beta_param
        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_decay_rate = learning_rate_decay_rate
        self.learning_rate_decay_steps = learning_rate_decay_steps
        self.optimizer = optimizer
        if optimizer == "Adam":
            self.optimizer = tf.keras.optimizers.Adam
        if optimizer == "SGD":
            self.optimizer = tf.keras.optimizers.SGD
        if optimizer == "Nadam":
            raise NotImplementedError
        self.optimizer_name = optimizer
        self.epoch = epoch
        self.verbose = verbose
        self.validation_size = validation_size
        self.run_eagerly = run_eagerly
        self.inputshape = None
        self.quantize = quantize

    def mutual_information_bernoulli_loss(self, y_true, y_pred):
        """
        I(x;y)  = H(x)   - H(x|y)
                = H(L_n) - H(L_n|s)
                = H(L_n) - (H(L_n|s=0) + H(L_n|s=1))
        H_bernoulli(x) = -(1-theta) x ln(1-theta) - theta x ln(theta)
        here theta => probability for 1 and 1-theta => probability for 0

        pseudocode:
        def get_h_bernoulli(l):
            theta = np.mean(l, axis=0)
            return -(1-theta) * np.log2(1-theta) - theta * np.log2(theta)

        y_pred = np.random.binomial(n=1, p=0.6, size=[2000, 5])
        y_true = np.random.binomial(n=1, p=0.6, size=[2000])

        y_pred[y_true == 0] = np.random.binomial(n=1, p=0.5, size=[len(y_true[y_true == 0]), 5])
        y_pred[y_true == 1] = np.random.binomial(n=1, p=0.8, size=[len(y_true[y_true == 1]), 5])

        H_L_n = get_h_bernoulli(y_pred)
        H_L_n_s0 = get_h_bernoulli(y_pred[y_true == 0])
        H_L_n_s1 = get_h_bernoulli(y_pred[y_true == 1])

        counts = np.bincount(y_true)

        MI = H_L_n - ((counts[0] / 2000 * H_L_n_s0) + (counts[1] / 2000 * H_L_n_s1))

        return np.sum(MI)

        :param y_pred: output of the layer
        :param y_true: sensitive attribute
        :return: The loss
        """

        def get_h_bernoulli(tensor):
            def get_theta(x):
                alpha = None
                temperature = 6.0
                use_real_sigmoid = True
                # hard_sigmoid
                _sigmoid = tf.keras.backend.clip(0.5 * x + 0.5, 0.0, 1.0)
                if isinstance(alpha, six.string_types):
                    assert self.alpha in ["auto", "auto_po2"]

                if isinstance(alpha, six.string_types):
                    len_axis = len(x.shape)

                    if len_axis > 1:
                        if K.image_data_format() == "channels_last":
                            axis = list(range(len_axis - 1))
                        else:
                            axis = list(range(1, len_axis))
                    else:
                        axis = [0]

                    std = K.std(x, axis=axis, keepdims=True) + K.epsilon()
                else:
                    std = 1.0

                if use_real_sigmoid:
                    p = tf.keras.backend.sigmoid(temperature * x / std)
                else:
                    p = _sigmoid(temperature * x / std)

                return p

            def log2(x):
                numerator = tf.math.log(x + 1e-20)
                denominator = tf.math.log(tf.constant(2, dtype=numerator.dtype))
                return numerator / denominator

            theta = tf.reduce_mean(get_theta(tensor), axis=0)

            return tf.reduce_sum(-(1 - theta) * log2(1 - theta) - theta * log2(theta))

        def compute_for_value(value, y_true, y_pred, get_h_bernoulli):
            if tf.shape(y_true).shape[0] == 1:
                y_filter = tf.where(y_true == value)
            else:
                y_filter = tf.where(y_true[:, 0] == value)[:, 0]

            y_i = tf.gather(y_pred, indices=y_filter)
            H_L_n_si = get_h_bernoulli(y_i)
            cnt_i = tf.shape(y_i)[0] + tf.cast(1e-16, dtype=tf.int32)  # number of repr with index i
            norm_si = cnt_i / tf.shape(y_pred)[0]
            return H_L_n_si, norm_si

        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.float64)
        H_L_n = get_h_bernoulli(y_pred)

        unique_y_true, _ = tf.unique(y_true)

        H_L_n_s = []
        norm_s = []

        def compute_per_value(v):
            return compute_for_value(v, y_true, y_pred, get_h_bernoulli)

        results = tf.map_fn(compute_per_value, unique_y_true, fn_output_signature=(tf.float64, tf.float64))

        H_L_n_s = tf.convert_to_tensor(results[0])
        norm_s = tf.convert_to_tensor(results[1])

        MI = H_L_n - tf.reduce_sum(tf.math.multiply(norm_s, H_L_n_s))

        # NOTE: this is a hotfix when we dont have all classes
        MI = tf.where(tf.math.is_nan(MI), tf.convert_to_tensor([0.0], dtype=tf.float64), MI)

        return tf.cast(MI, dtype=tf.float32)

    def train_step(self, data):
        # data = [x, s]
        x, s = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z, z_sample = self.encoder(x)
            reconstruction = self.decoder(z)
            reconstruction_loss = keras.losses.MeanSquaredError()(x, reconstruction)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = self.beta_param * tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            mi_loss = self.mutual_information_bernoulli_loss(s, z_sample)
            total_loss = (1-self.beta_param)*reconstruction_loss + kl_loss + self.gamma * mi_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.mi_loss_tracker.update_state(mi_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "mi_loss": self.mi_loss_tracker.result(),
        }

    def test_step(self, data):
        # data = [x, s]
        x, s = data

        z_mean, z_log_var, z, z_sample = self.encoder(x)
        reconstruction = self.decoder(z)
        reconstruction_loss = keras.losses.MeanSquaredError()(x, reconstruction)
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = self.beta_param * tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        mi_loss = self.mutual_information_bernoulli_loss(s, z_sample)
        total_loss = (1-self.beta_param)*reconstruction_loss + kl_loss + self.gamma * mi_loss
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.mi_loss_tracker.update_state(mi_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "mi_loss": self.mi_loss_tracker.result(),
        }

    def call(self, x):
        z_mean, z_log_var, z, z_sample = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction

    def get_encoder(self):
        encoder_inputs = keras.Input(shape=self.inputshape)

        # layers
        x = encoder_inputs
        for layer in self.hidden_layers:
            
            #ORIGINAL CODE THAT I COULD REPLACE FOR THE GLOROTUNIFORM
            x = layers.Dense(
                layer,
                activation=self.activation,
                kernel_regularizer=tf.keras.regularizers.l2(self.kernel_regularizer),
                activity_regularizer=tf.keras.regularizers.l2(self.kernel_regularizer)
            )(x)
            
            #Extra added (to delete if it does not work)
            #x=layers.Dense(
            #layer,
            #activation=self.activation,
            #kernel_initializer=initializers.initializers.GlorotUniform(),
            #bias_initializer=initializers.Zeros()
            #)(x)
            #More original code
            if self.drop_out > 0:
                x = layers.Dropout(self.drop_out)(x)

        # setup latent layers
        z_mean = layers.Dense(
            self.latent_dims,
            name="z_mean",
            kernel_regularizer=tf.keras.regularizers.l2(self.kernel_regularizer),
            activity_regularizer=tf.keras.regularizers.l2(self.kernel_regularizer)
        )(x)
        z_log_var = layers.Dense(
            self.latent_dims,
            name="z_log_var",
            kernel_regularizer=tf.keras.regularizers.l2(self.kernel_regularizer),
            activity_regularizer=tf.keras.regularizers.l2(self.kernel_regularizer)
        )(x)

        z = Sampling(self.latent_dims)([z_mean, z_log_var])
        z_sample = z
        if self.quantize:
            z_sample = BernoulliSampling(self.num_samples, name="bernoulli")(z_sample)
            # z = QActivation("bernoulli")(z)

        # build encoder
        encoder = keras.Model(
            encoder_inputs,
            [z_mean, z_log_var, z, z_sample],
            name="encoder"
        )
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
                kernel_regularizer=tf.keras.regularizers.l2(self.kernel_regularizer),
                activity_regularizer=tf.keras.regularizers.l2(self.kernel_regularizer)
            )(x)
            if self.drop_out > 0:
                x = layers.Dropout(self.drop_out)(x)

        # last layer
        x = layers.Dense(
            self.inputshape[0],
            activation=self.activation,
            kernel_regularizer=tf.keras.regularizers.l2(self.kernel_regularizer),
            activity_regularizer=tf.keras.regularizers.l2(self.kernel_regularizer)
        )(x)

        # build decoder
        decoder = keras.Model(
            latent_inputs,
            x,
            name="decoder"
        )
        decoder.summary()
        return decoder

    def fit(self, x, s):

        # get input shape
        self.inputshape = [x.shape[1]]

        # fill up batch size NOTE: MMD loss can have NaN values so batch_size should be not to small
        if len(x) % self.batch_size != 0:
            x = np.concatenate([x, x[:self.batch_size-len(x) % self.batch_size]])
            s = np.concatenate([s, s[:self.batch_size-len(s) % self.batch_size]])

        self.encoder = self.get_encoder()
        self.decoder = self.get_decoder()

        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            self.learning_rate,
            decay_steps=self.learning_rate_decay_steps,
            decay_rate=self.learning_rate_decay_rate,
            staircase=False
        )

        if type(self.optimizer) is str:
            if self.optimizer == "Adam":
                self.optimizer = tf.keras.optimizers.Adam
            if self.optimizer == "SGD":
                self.optimizer = tf.keras.optimizers.SGD
            if self.optimizer == "Nadam":
                raise NotImplementedError

        #CÓDIGO OG
        
        self.compile(
            optimizer=self.optimizer(lr_schedule),
            run_eagerly=self.run_eagerly
        )

        history = super().fit(
            x,
            s,
            epochs=self.epoch,
            batch_size=self.batch_size,
            verbose=self.verbose
        )
        #METO RETURN HISTORY PARA PLOTS
        return history

    def score(self, x, y):
        z_mean, z_log_var, z, z_sample = self.encoder(x)
        reconstruction = self.decoder(z)
        reconstruction_loss = keras.losses.MeanSquaredError()(x, reconstruction)
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = self.beta_param * tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        mi_loss = self.mutual_information_bernoulli_loss(y, z_sample)
        total_loss = (1-self.beta_param)*reconstruction_loss + kl_loss + self.gamma * mi_loss
        return +total_loss.numpy() #RETURNS SCALAR! (úsase para comprobar cal dá menor loss co grid/random search)

    def score_vector(self, x):
        z_mean, z_log_var, z, z_sample = self.encoder(x)
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.keras.losses.mse(x, reconstruction)
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = self.beta_param * tf.reduce_sum(kl_loss, axis=1)
        total_loss = (1-self.beta_param)*reconstruction_loss + kl_loss
        losses=[reconstruction_loss.numpy(), kl_loss.numpy(), total_loss.numpy()]
        return losses #RETURNS VECTOR! THERE IS NO MUTUAL INFORMATION LOSS (ONLY RELEVANT BEFORE TO GET SCORE FOR GRID/RANDOM SEARCH!)

    def get_mean(self, x):
        return self.encoder(x)[0]

    def get_sigma(self, x):
        return self.encoder(x)[1]

    def get_latentspace(self, x):
        return self.encoder(x)[2]

    def get_params(self, deep=True):
        out = dict()
        for key in self._get_param_names():
            cur_key = key
            if key == "optimizer":
                cur_key = "optimizer_name"
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
            if key.startswith("_"):
                drop_key.append(key)

        for key in drop_key:
            try:
                d.pop(key)
            except KeyError:
                pass

        for key in ['optimizer', 'nan_loss', 'built', 'inputs',
                    'input_names', 'output_names', 'stop_training'
                    'history', 'train_function', 'test_function',
                    'predict_function', 'loss_tracker_a', 'loss_tracker_b',
                    'loss_tracker_cls', 'model_a', 'model_b', 'ranker',
                    'compiled_loss', 'compiled_metrics', 'loss', 'history',
                    'outputs'
                    ]:
            try:
                d.pop(key)
            except KeyError:
                pass

        return d
