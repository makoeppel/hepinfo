from __future__ import annotations

import gc
from typing import Any
from functools import partial

import keras
from keras.api.saving import register_keras_serializable

import numpy as np
from squark.utils.sugar import FreeEBOPs

from hepinfo.models.BaseModel import BaseModel
from hepinfo.models.QuantFlow import FreeBOPs
from hepinfo.util import MILoss


class BinaryMI(BaseModel):
    """
    Stochastically Quantized Neural Network which has a bernoulli
    activation after each layer to exactly compute the mutual information.
    """

    def __init__(
        self,
        # BinaryMI HPs
        hidden_layers: list[int] = [1024, 512, 128],
        batch_normalisation_layers: list[int] = [1024, 512, 128],
        quantized_position: list[bool] = [False, True, False],
        batch_normalisation: bool = False,
        activation_binary: str = 'bernoulli',
        activation_nonbinary: str = 'tanh',
        acitvation_last_layer: str = 'sigmoid',
        kernel_regularizer: float = 0.01,
        drop_out: float = 0.0,
        gamma: float = 0.0,
        use_s_quark: bool = False,
        use_qkeras: bool = False,
        use_quantflow: bool = False,
        init_quantized_bits=32,
        input_quantized_bits='quantized_bits(16, 6, 0)',
        quantized_bits='quantized_bits(16, 6, 0, use_stochastic_rounding=True)',
        quantized_activation='quantized_relu(10, 6, use_stochastic_rounding=True, negative_slope=0.0)',
        alpha=1,
        beta0=1,
        num_samples=1,
        use_quantized_sigmoid=True,
        bits_bernoulli_sigmoid=8,
        # Common HPs
        batch_size: int = 200,
        learning_rate: float = 0.001,
        learning_rate_decay_rate: float = 1,
        learning_rate_decay_steps: int = 1000,
        optimizer: str = 'Adam',
        epoch: int = 10,
        loss: str = 'binary_crossentropy',
        run_eagerly: bool = False,
        # other variables
        verbose: int = 0,
        validation_size: float = 0.1,
        input_shape: tuple | int = 0,
        last_layer_size: int = 1,
        random_seed: int = 42,
        name: str = 'BinaryMI',
        dataset_name: str = 'Mnist',
        print_summary: bool = False,
        bits: int = 2,
        checkpoint_path: str = '',
        datetime: str = '',
        conv: bool = False,
    ) -> None:
        super().__init__(
            # DirectRankerAdv HPs
            hidden_layers=hidden_layers,
            batch_normalisation_layers=batch_normalisation_layers,
            quantized_position=quantized_position,
            batch_normalisation=batch_normalisation,
            activation_binary=activation_binary,
            activation_nonbinary=activation_nonbinary,
            acitvation_last_layer=acitvation_last_layer,
            kernel_regularizer=kernel_regularizer,
            drop_out=drop_out,
            use_s_quark=use_s_quark,
            use_qkeras=use_qkeras,
            use_quantflow=use_quantflow,
            init_quantized_bits=init_quantized_bits,
            input_quantized_bits=input_quantized_bits,
            quantized_bits=quantized_bits,
            quantized_activation=quantized_activation,
            alpha=alpha,
            beta0=beta0,
            num_samples=num_samples,
            use_quantized_sigmoid=use_quantized_sigmoid,
            bits_bernoulli_sigmoid=bits_bernoulli_sigmoid,
            # Common HPs
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_decay_rate=learning_rate_decay_rate,
            learning_rate_decay_steps=learning_rate_decay_steps,
            optimizer=optimizer,
            epoch=epoch,
            loss=loss,
            # other variables
            verbose=verbose,
            validation_size=validation_size,
            input_shape=input_shape,
            random_seed=random_seed,
            name=name,
            dataset_name=dataset_name,
            print_summary=print_summary,
            bits=bits,
            checkpoint_path=checkpoint_path,
            datetime=datetime,
            run_eagerly=run_eagerly,
            last_layer_size=last_layer_size,
            conv=conv,
        )

        self.x_input = Any
        self.out = Any
        self.gamma = gamma

    def _build_model(self) -> None:
        r"""
        Is building the model by sequential adding layers.

        1. The data (x) gets into an input layer which is activated
        with a bernoulli layer via a sigmoid:

        .. math::
            \hat{x} = \mathrm{Bern} (\sigma(6.0 \cdot x / 1.0))

        2. The size of the next layers are defined with self.hidden_layers.
        The basic architecture is build via:

        .. code-block:: python

            for i in self.hidden_layers:
                layer = Dense()(layer)
                layer = QActivation('bernoulli')(layer)

        3. To avoid overfitting it is often useful to lower the learning rate
        during the training. Therefore, the model is build using a schedule
        which applies the inverse decay function to an initial learning rate.

        4. Different optimizers can be used such as Adam, Nadam or SGD.

        Args:
            NoInput
        Returns:
            None
        """

        # placeholders for the inputs: shape depends on whether we have a convnet or not
        self.x_input = keras.layers.Input(shape=self.input_shape, name='x')

        # build layers layers
        self.out, self.last_quantized = self._get_hidden_qlayer(
            self.x_input,
            hidden_layer=self.hidden_layers,
            drop_out=self.drop_out,
            name='t',
            kernel_regularizer=keras.regularizers.l2(self.kernel_regularizer),
            conv=self.conv,
        )

        if sum(self.quantized_position) > 0:
            outputs = [self.out, self.last_quantized]
            self.index_qact = [i for i, x in enumerate(self.quantized_position) if x][-1]
            loss = {f't_{len(self.hidden_layers)}': self.loss, f't_{self.index_qact}': MILoss(self.use_quantized_sigmoid, self.bits_bernoulli_sigmoid)}
            lossWeights = {f't_{len(self.hidden_layers)}': 1, f't_{self.index_qact}': float(self.gamma)}
            metrics = {
                f't_{len(self.hidden_layers)}': 'AUC' if self.last_layer_size == 1 else 'acc',
                f't_{self.index_qact}': 'acc',
            }
        else:
            outputs = self.out
            loss = [self.loss]
            lossWeights = float(1)
            metrics = ['AUC'] if self.last_layer_size == 1 else ['acc']

        # create the model
        self.model = keras.models.Model(inputs=self.x_input, outputs=outputs, name=self.name)

        if self.print_summary:
            self.model.summary()  # type: ignore

        # setup learning rate schedule
        # TODO: maybe we have to go with a simple learning rate here for the experiments
        lr_schedule = keras.optimizers.schedules.InverseTimeDecay(
            self.learning_rate,
            decay_steps=self.learning_rate_decay_steps,
            decay_rate=self.learning_rate_decay_rate,
            staircase=False,
        )

        if sum(self.quantized_position) > 0:
            self.model.compile(  # type: ignore
                optimizer=self.optimizer(lr_schedule),
                loss=loss,
                loss_weights=lossWeights,
                metrics=metrics,
                run_eagerly=self.run_eagerly,
            )
        else:
            self.model.compile(  # type: ignore
                optimizer=self.optimizer(lr_schedule), loss=loss, metrics=metrics, run_eagerly=self.run_eagerly
            )

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, s_train: np.ndarray, **fit_params: dict[Any, Any]) -> None:
        """
        Fit function which first build the model and than
        fits it using x_train and y_train. A special callback
        class is used to compute the mutual information at the
        beginning of each epoch and at the end of the training.

        Args:
            NDArray: x_train
            NDArray: y_train
            NDArray: s_train
            dict[Any]: fit_params

        Returns:
            None
        """

        # get the correct numpy type
        x_train = self.convert_array_to_float(x_train)
        y_train = self.convert_array_to_float(y_train)
        s_train = self.convert_array_to_float(s_train)

        # input size
        self.input_shape = (x_train.shape[1],)

        # convert for classifier output
        if self.last_layer_size > 1:
            y_train = keras.utils.to_categorical(y_train, self.last_layer_size)

        self._build_model()

        callback = []
        if self.use_s_quark:
            nan_terminate = keras.callbacks.TerminateOnNaN()
            ebops = FreeEBOPs()
            callback.append(nan_terminate)
            callback.append(ebops)

        if self.use_quantflow:
            nan_terminate = keras.callbacks.TerminateOnNaN()
            bops = FreeBOPs()
            callback.append(nan_terminate)
            callback.append(bops)

        if sum(self.quantized_position) > 0:
            y_train = {f't_{len(self.hidden_layers)}': y_train, f't_{self.index_qact}': s_train}

        history = self.model.fit(  # type: ignore
            x=x_train,
            y=y_train,
            batch_size=self.batch_size,
            epochs=self.epoch,
            verbose=self.verbose,
            shuffle=True,
            validation_split=self.validation_size,
            callbacks=callback,
        )
        # https://github.com/tensorflow/tensorflow/issues/14181
        # https://github.com/tensorflow/tensorflow/issues/30324
        gc.collect()

        return history

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Get the class probablities.

        Args:
            NDArray: features
        Returns:
            NDArray: class probablities
        """

        if len(features.shape) == 1:
            features = [features]  # type: ignore

        res = self.model.predict(  # type: ignore
            features, batch_size=self.batch_size, verbose=str(self.verbose)
        )[0]

        return res
