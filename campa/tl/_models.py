# TODO go through functions and comment
from copy import deepcopy
from enum import Enum
from typing import Any, Dict, Tuple, Optional
from functools import partial
import json
import logging

import tensorflow as tf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class ModelEnum(str, Enum):
    """
    Neural network models.

    Possible values are:

    - ``ModelEnum.BaseAEModel``: "BaseAEModel" (:class:`BaseAEModel`)
    - ``ModelEnum.VAEModel``: "VAEModel" (:class:`VAEModel`)
    """

    BaseAEModel = "BaseAEModel"
    VAEModel = "VAEModel"
    CatVAEModel = "CatVAEModel"
    GMMVAEModel = "GMMVAEModel"
    CondCatVAEModel = "CondCatVAEModel"

    def get_cls(self):
        """
        Get model class from enum string.
        """
        cls = self.__class__
        if self == cls.BaseAEModel:
            return BaseAEModel
        elif self == cls.VAEModel:
            return VAEModel
        elif self == cls.CatVAEModel:
            return CatVAEModel
        elif self == cls.GMMVAEModel:
            return GMMVAEModel
        elif self == cls.CondCatVAEModel:
            return CondCatVAEModel
        else:
            raise NotImplementedError


# --- tf functions needed for model definition ---
def expand_and_broadcast(x, s=1):
    """Expand tensor x with shape (batches,n) to shape (batches,s,s,n)."""
    C = tf.expand_dims(tf.expand_dims(x, 1), 1)
    C = tf.broadcast_to(C, [tf.shape(C)[0], s, s, tf.shape(C)[-1]])
    return C


def reparameterize_gumbel_softmax(latent, temperature=0.1):
    """Draw a sample from the Gumbel-Softmax distribution."""

    def sample_gumbel(shape, eps=1e-20):
        """Sample from Gumbel(0, 1)."""
        U = tf.random.uniform(shape, minval=0, maxval=1)
        return -tf.math.log(-tf.math.log(U + eps) + eps)

    y = latent + sample_gumbel(tf.shape(latent))
    return tf.nn.softmax(y / temperature)


def reparameterize_gaussian(latent):
    """Draw a sample from Gaussian distribution."""
    z_mean, z_log_var = tf.split(latent, 2, axis=-1)
    eps = tf.random.normal(shape=tf.shape(z_mean))
    return eps * tf.exp(z_log_var * 0.5) + z_mean


@tf.custom_gradient
def grad_reverse(x):
    """Reverse gradients for adversarial loss."""
    y = tf.identity(x)

    def custom_grad(dy):
        return -dy

    return y, custom_grad


class GradReverse(tf.keras.layers.Layer):
    """Reverse gradients class."""

    def __init__(self):
        super().__init__()

    def call(self, x):
        """Apply gradient reversal."""
        return grad_reverse(x)


# --- Model classes ---

BASE_MODEL_CONFIG: Dict[str, Any] = {
    "name": None,
    # input definition
    "num_neighbors": 1,
    "num_channels": 35,  # can be split up in num_input_channels and num_output_channels
    "num_output_channels": None,
    "num_input_channels": None,
    # conditions are appended to the input and the latent representation. They are assumed to be 1d
    "num_conditions": 0,  # if > 0, the model is conditional and an additional input w/ conditions is assumed.
    # if number or list, the condition is encoded using dense layers with this number of nodes
    "encode_condition": None,
    # which layers of encoder and decoder to apply condition to.
    # Give index of layer in encoder and decoder
    "condition_injection_layers": [0],
    # encoder architecture
    "input_noise": None,  # 'gaussian', 'dropout', adds noise to encoder input
    "noise_scale": 0,
    "encoder_conv_layers": [32],
    "encoder_conv_kernel_size": [1],
    "encoder_fc_layers": [32, 16],
    # from last encoder layer, a linear fcl to latent_dim is applied
    "latent_dim": 16,  # number of nodes in latent space (for some models == number of classes)
    # decoder architecture
    # from last decoder layer, a linear fcl to num_output_channels is applied
    "decoder_fc_layers": [],
    # decoder regularizer
    "decoder_regularizer": None,  # 'l1' or 'l2'
    "decoder_regularizer_weight": 0,
    # for adversarial models, add adversarial layers
    "adversarial_layers": None,  # only works with categorical conditions
}


class BaseAEModel:
    """
    Base class for AE and VAE models.

    This model can have neighbours, conditions (concatenated to input + decoder), and
    and adversarial head.
    The class defines initialisation functions for setting up AE with encoder and decoder.
    In addition, encoder and decoder input layers are defined (can be overwritten in subclassed functions).

    Subclassed models can define:

    - :attr:`BaseAEModel.default_config` (as class variable).
    - :meth:`BaseAEModel.create_encoder` function, returning encoder model and latent (for KL loss).
    - :meth:`BaseAEModel.create_decoder` function, returning decoder model.
    - :meth:`BaseAEModel.create_model` function, creating overall model (put encoder and decoder together).

    Default architecture of this model:

    - Encoder: ``(noise) - conv layers - fc layers - linear layer to latent_dim``
    - Decoder: ``fc_layers - linear (regularized) layer to num_output_channels``

    Conditional models additionally output `latent`, the latent space (for KL loss computation).

    Adversarial models additionally output `adv_head`, the output of the adversarial head
    (for adversarial loss computation).
    ``(adv_latent - reverse_gradients - adversarial_layers - linear layer to num_conditions)``.

    TODO adversarial models are not tested in this version of the code.

    Parameters
    ----------
    name: Optional[str]
        Model name.
    num_neighbors: str
        Number of neighbours used in input data.
    num_channels: int
        Number of channels for input and output.
        Can be split up in ``num_input_channels`` and ``num_output_channels``.
    num_output_channels: Optional[int]
        Number of output channels
    num_input_channels: Optional[int]
        Number of input channels
    num_conditions: int
        Number of conditions. If > 0, the model is conditional and an additional input w/ conditions is assumed.
        Conditions are appended to the input and the latent representation. They are assumed to be 1d.
    encode_condition: Optional[Union[int, Iterable[int]]
        If number or list, the condition is encoded using dense layers with this number of nodes.
    condition_injection_layers: Iterable[int]
        Which layers of encoder and decoder to apply condition to.
        Give index of layer in encoder and decoder.
    input_noise: Optional[str]
        One of `gaussian`, `dropout`, adds noise to encoder input.
    noise_scale: int
        Scale of Gaussian noise.
    encoder_conv_layers: Iterable[int]
        Size of convolutional layers for encoder.
    encoder_conv_kernel_size: Iterable[int]
        Kernel size for each encoder convolutional layer.
    encoder_fc_layers: Iterable[int]
        Size of fully connected encoder layers.
        From last encoder layer, a linear fully connected layer to ``latent_dim`` is applied.
    latent_dim: int
        Number of nodes in latent space (for some models == number of classes).
    decoder_fc_layers: Iterable[int]
        Decoder architecture.
        From last decoder later, a linear fully connected layer to ``num_output_channels`` is applied.
    decoder_regularizer: Optional[str]
        Regularizer for decoder, `l1` or `l2`.
    decoder_regularizer_weight: float
        Weight of regularizer.
    adversarial_layers
        For adversarial models, add adversarial layers.
        Only works with categorical conditions.
    """

    default_config: Dict[str, Any] = {"name": "BaseAEModel"}
    """
    Default config used in every model.
    """

    def __init__(self, **kwargs):
        # set up log and config
        self.log = logging.getLogger(self.__class__.__name__)
        self.config = deepcopy(BASE_MODEL_CONFIG)
        self.config.update(self.default_config)
        self.config.update(kwargs)
        if self.config["num_output_channels"] is None:
            self.config["num_output_channels"] = self.config["num_channels"]
        if self.config["num_input_channels"] is None:
            self.config["num_input_channels"] = self.config["num_channels"]
        if isinstance(self.config["encoder_conv_kernel_size"], int):
            self.config["encoder_conv_kernel_size"] = [
                self.config["encoder_conv_kernel_size"] for _ in self.config["encoder_conv_layers"]
            ]
        self.log.info("Creating model")
        self.log.debug(f"Creating model with config: {json.dumps(self.config, indent=4)}")

        # set up model
        # input layers for encoder and decoder
        self.encoder_input = tf.keras.layers.Input(
            (
                self.config["num_neighbors"],
                self.config["num_neighbors"],
                self.config["num_channels"],
            )
        )
        self.decoder_input = tf.keras.layers.Input((self.config["latent_dim"],))
        if self.is_conditional:
            self.encoder_input = [
                self.encoder_input,
                tf.keras.layers.Input((self.config["num_conditions"],)),
            ]
            self.decoder_input = [
                self.decoder_input,
                tf.keras.layers.Input((self.config["num_conditions"],)),
            ]

        # set self.encoder, self.latent, self.decoder, self.model_output
        self.model = self.create_model()

        # expose layers and summary here
        self.layers = self.model.layers
        summary = []
        self.model.summary(print_fn=lambda x: summary.append(x))
        self.encoder.summary(print_fn=lambda x: summary.append(x))
        self.decoder.summary(print_fn=lambda x: summary.append(x))
        if self.is_adversarial:
            self.adv_head.summary(print_fn=lambda x: summary.append(x))
        self.summary = "\n".join(summary)

    def create_model(self) -> tf.keras.Model:
        """
        Create :class:`tf.keras.Model`.

        Use :meth:`BaseAEModel.create_encoder` and :meth:`BaseAEModel.create_decoder` functions to
        set ``self.encoder``, ``self.latent``, ``self.decoder``, ``self.model_output`` attributes.

        Returns
        -------
        Neural network model.
        """
        # encoder and decoder
        self.encoder, self.latent = self.create_encoder()
        self.decoder = self.create_decoder()
        if self.is_adversarial:
            self.adv_head = self.create_adversarial_head()

        # create model
        if self.is_conditional:
            # self.model_output = self.decoder([self.encoder.output, self.encoder.input[1]])
            self.model_output = self.decoder([self.encoder(self.encoder_input), self.encoder_input[1]])
        else:
            self.model_output = self.decoder([self.encoder(self.encoder_input)])
        if self.latent is not None:
            # model should return both output + latent (for KL loss)
            self.model_output = [self.model_output, self.latent]
        if self.is_adversarial:
            if isinstance(self.model_output, list):
                self.model_output = self.model_output + [self.adv_head(self.encoder(self.encoder_input))]
            else:
                self.model_output = [
                    self.model_output,
                    self.adv_head(self.encoder(self.encoder_input)),
                ]
        model = tf.keras.Model(self.encoder_input, self.model_output, name=self.config["name"])
        return model

    @property
    def is_conditional(self):
        """Flag set based on ``num_conditions``."""
        return self.config["num_conditions"] > 0

    @property
    def is_adversarial(self):
        """Model is adversarial if is is conditional and adversarial layers are defined."""
        return self.config["adversarial_layers"] is not None and self.is_conditional

    def encode_condition(self, C):
        """
        Apply condition encoder to C.

        Parameters
        ----------
        C
            condition (maybe one-hot encoded)
        Returns
        -------
        Encoded condition.
        """
        if self.config["encode_condition"] is None:
            return C
        if not hasattr(self, "condition_encoder"):
            enc_l = self.config["encode_condition"]
            if isinstance(enc_l, int):
                enc_l = [enc_l]
            inpt = tf.keras.layers.Input((self.config["num_conditions"],))
            x = inpt
            for layer in enc_l:
                x = tf.keras.layers.Dense(layer, activation=tf.nn.relu)(x)
            self.condition_encoder = tf.keras.Model(inpt, x, name="condition_encoder")
        return self.condition_encoder(C)

    def _create_base_encoder(self):
        """
        Create base encoder structure with convolutional layers and fully connected layers.

        Does not apply the last (linear) layer to latent_dim - useful for VAE which does this differently.
        """
        if self.is_conditional:
            X, C = self.encoder_input
            C = self.encode_condition(C)
            # broadcast C to fit to X
            # fn = partial(expand_and_broadcast, s=self.config['num_neighbors'])
            # C = tf.keras.layers.Lambda(fn)(C)
        else:
            X = self.encoder_input
        if self.config["input_noise"] is not None:
            # add noise
            X = self.add_noise(X)
        # if self.is_conditional:
        #    # concatenate input and conditions
        #    X = tf.keras.layers.concatenate([X, C], axis=-1)

        # conv layers
        cond_layers = self.config["condition_injection_layers"]
        for i, l in enumerate(self.config["encoder_conv_layers"]):
            # check if need to concatenate current X with C
            if self.is_conditional and i in cond_layers:
                # need to broadcast C to fit to X
                fn = partial(expand_and_broadcast, s=self.config["num_neighbors"])
                C_bcast = tf.keras.layers.Lambda(fn)(C)
                X = tf.keras.layers.concatenate([X, C_bcast], axis=-1)
            k = self.config["encoder_conv_kernel_size"][i]
            X = tf.keras.layers.Conv2D(l, kernel_size=(k, k), activation=tf.nn.relu)(X)
        X = tf.keras.layers.Flatten()(X)
        # fully connected layers
        for j, l in enumerate(self.config["encoder_fc_layers"]):
            # check if need to concatenate current X with C
            if self.is_conditional and i + 1 + j in cond_layers:
                X = tf.keras.layers.concatenate([X, C], axis=-1)
            X = tf.keras.layers.Dense(l, activation=tf.nn.relu)(X)
        return X

    def create_encoder(self) -> Tuple[tf.keras.Model, Optional[tf.Tensor]]:
        """
        Create encoder.

        Encoder outputs reparameterized latent.
        Latent is potentially returned by overall model for loss calculation, e.g. for VAE.

        Returns
        -------
        Encoder and latent (None for BaseAEModel).
        """
        X = self._create_base_encoder()
        # linear layer to latent
        X = tf.keras.layers.Dense(self.config["latent_dim"], activation=None, name="latent")(X)

        # define encoder model
        encoder = tf.keras.Model(self.encoder_input, X, name="encoder")
        return encoder, None

    def create_decoder(self):
        """
        Create decoder.

        Returns
        -------
        tf.keras.Model
            Decoder.
        """
        X = self.decoder_input
        if self.is_conditional:
            X, C = self.decoder_input
            C = self.encode_condition(C)
            # concatenate latent + conditions
            # X = tf.keras.layers.concatenate([X, C])

        # fully-connected layers
        cond_layers = self.config["condition_injection_layers"]
        for i, l in enumerate(self.config["decoder_fc_layers"]):
            # check if need to concatenate current X with C
            if self.is_conditional and i in cond_layers:
                X = tf.keras.layers.concatenate([X, C], axis=-1)
            X = tf.keras.layers.Dense(l, activation=tf.nn.relu)(X)
            if i == 0 and self.is_conditional:
                self.entangled_latent = X  # might need this later on
        # if no fully-connected layers are build, need to still concatenate current X with C
        if len(self.config["decoder_fc_layers"]) == 0 and self.is_conditional:
            X = tf.keras.layers.concatenate([X, C], axis=-1)

        # linear layer to num_output_channels (optionally regularized)
        if self.config["decoder_regularizer"] == "l1":
            tf.keras.regularizers.l1(self.config["decoder_regularizer_weight"])
        else:
            tf.keras.regularizers.l2(self.config["decoder_regularizer_weight"])
        decoder_output = tf.keras.layers.Dense(self.config["num_output_channels"], activation=None)(X)

        # define decoder model
        decoder = tf.keras.Model(self.decoder_input, decoder_output, name="decoder")
        return decoder

    def create_adversarial_head(self) -> tf.keras.Model:
        """
        Create adversarial head: ``reverse_gradient - adversarial_layers - num_conditions``.

        Returns
        -------
        adversarial head.
        """
        assert self.is_conditional
        assert self.is_adversarial

        inpt = tf.keras.layers.Input((self.config["latent_dim"],))
        X = inpt
        X = GradReverse()(X)
        for layer in self.config["adversarial_layers"]:
            X = tf.keras.layers.Dense(layer, activation=tf.nn.relu)(X)
        # linear layer to num_conditions
        adv_head_output = tf.keras.layers.Dense(self.config["num_conditions"], activation=None)(X)

        # define adv_head model
        adv_head = tf.keras.Model(inpt, adv_head_output, name="adv_head")
        return adv_head

    def add_noise(self, X):
        """
        Add noise to X.

        Parameters
        ----------
        X
            inputs.

        Returns
        -------
        X with noise applied.
        """
        if self.config["input_noise"] == "dropout":
            X = tf.keras.layers.Dropout(self.config["noise_scale"])(X)
        elif self.config["input_noise"] == "gaussian":
            X = tf.keras.layers.GaussianNoise(self.config["noise_scale"])(X)
        else:
            raise NotImplementedError
        return X


class VAEModel(BaseAEModel):
    """
    VAE with simple Gaussian prior (trainable with KL loss).

    Inherits from :class:`BaseAEModel`.

    Model architecture:

    - Encoder: ``(noise) - conv layers - fc layers - linear layer to latent_dim * 2``
    - Latent: split ``latent_dim`` in half, re-sample using Gaussian prior
    - Decoder: ``fc_layers - linear (regularized) layer to num_output_channels``
    """

    default_config = {"name": "VAEModel"}

    def create_encoder(self) -> Tuple[tf.keras.Model, Optional[tf.Tensor]]:
        """
        Create encoder.

        Encoder outputs reparameterized latent.
        Latent is potentially returned by overall model for loss calculation, e.g. for VAE.

        Returns
        -------
        Encoder and latent (None for ``BaseAEModel``).
        """
        X = self._create_base_encoder()
        # linear layer to latent
        latent = tf.keras.layers.Dense(self.config["latent_dim"] * 2, activation=None, name="latent")(X)
        # reparameterise
        reparam_latent = reparameterize_gaussian(latent)
        # define encoder
        encoder = tf.keras.Model(self.encoder_input, reparam_latent, name="encoder")
        return encoder, latent


class CatVAEModel(BaseAEModel):
    """
    Categorical VAE Model.

    VAE with categorical prior (softmax gumbel) (trainable with categorical loss)
    Encoder: (noise) - conv layers - fc layers - linear layer to latent_dim * 2
    Latent: split latent_dim in half, resample using Gaussian prior
    Decoder: fc_layers - linear (regularized) layer to num_output_channels
    """

    default_config: Dict[str, Any] = {
        "name": "CatVAEModel",
        # temperature for scaling gumbel_softmax. values close to 0 are close to true categorical distribution
        "temperature": 0.1,
        "initial_temperature": 10,
        "anneal_epochs": 0,
    }

    def __init__(self, **kwargs):
        self.temperature = tf.Variable(
            initial_value=kwargs.get("initial_temperature", self.default_config["initial_temperature"]),
            trainable=False,
            dtype=tf.float32,
        )
        super().__init__(**kwargs)

    def create_encoder(self):
        """
        Create encoder.

        Encoder outputs reparameterised latent.
        Latent is potentially returned by overall model for loss calculation, e.g. for VAE.

        Returns
        -------
        tf.keras.model, tf.Tensor
            Encoder and latent (None for BaseAEModel).
        """
        X = self._create_base_encoder()
        # linear layer to latent
        latent = tf.keras.layers.Dense(self.config["latent_dim"], activation=None, name="latent")(X)
        # reparameterise
        reparam_latent = reparameterize_gumbel_softmax(latent, self.temperature)
        # define encoder
        encoder = tf.keras.Model(self.encoder_input, reparam_latent, name="encoder")
        return encoder, latent


class CondCatVAEModel(CatVAEModel):
    """
    Conditional Categorical VAE model.

    Conditional Categorical VAE using another concatenation scheme when adding the condition
    to the latent space. This model first calculates a fully connected layer to a vector
    with length #output_channels x #conditions

    IGNORES decoder_fc_layers - only supports linear decoder!
    """

    def create_decoder(self):
        """
        Create decoder.
        """
        X, C = self.decoder_input

        # dense layer to num_output_channels x num_conditions
        X = tf.keras.layers.Dense(
            self.config["num_output_channels"] * self.config["encode_condition"],
            activation=None,
        )(X)
        X = tf.keras.layers.Reshape((self.config["num_output_channels"], self.config["encode_condition"]))(X)

        C = self.encode_condition(C)
        # multiply X by conditions
        decoder_output = tf.keras.layers.Dot(axes=[2, 1])([X, C])

        # define decoder model
        decoder = tf.keras.Model(self.decoder_input, decoder_output, name="decoder")
        return decoder


class GMMVAEModel(BaseAEModel):
    """
    Gaussian Mixture Model VAE.

    VAE with gmm prior (trainable with categorical loss for y and weighted kl loss for z)
    Encoder y: (noise) - conv layers y - fc layers y - linear layer to latent_dim
    Encoder: (noise) + y - conv layers - fc layers - linear layer to latent_dim * 2
    Latent: split latent_dim in half, resample using Gaussian prior
    Decoder: fc_layers - linear (regularized) layer to num_output_channels
    """

    default_config: Dict[str, Any] = {
        "name": "GMMVAEModel",
        # y encoder architecture
        "y_conv_layers": None,
        "y_conv_kernel_size": None,
        "y_fc_layers": None,
        # pz (gmm prior for zmean and zvar from categorical y)
        "pz_fc_layers": None,
        # number of different gaussians
        "k": 10,
        # temperature for categorical loss on y <- might not need to anneal!
        # temperature for scaling gumbel_softmax. values close to 0 are close to true categorical distribution
        "temperature": 0.1,
        "initial_temperature": 10,
        "anneal_epochs": 0,
    }

    def __init__(self, **kwargs):
        # make sure y_... is defined
        config = deepcopy(BASE_MODEL_CONFIG)
        config.update(self.default_config)
        config.update(kwargs)
        if config["y_conv_layers"] is None:
            config["y_conv_layers"] = config["encoder_conv_layers"]
        if config["y_conv_kernel_size"] is None:
            config["y_conv_kernel_size"] = config["encoder_conv_kernel_size"]
        if config["y_fc_layers"] is None:
            config["y_fc_layers"] = config["encoder_fc_layers"]
        if config["pz_fc_layers"] is None:
            config["pz_fc_layers"] = config["encoder_fc_layers"]
        super().__init__(**config)

    def qy_graph(self, input_shape):
        """
        Return Y calculated from X (convolutional layers + fully connected layers).
        """
        X_input = tf.keras.layers.Input(input_shape)
        X = X_input
        # conv layers
        for i, l in enumerate(self.config["y_conv_layers"]):
            k = self.config["y_conv_kernel_size"][i]
            X = tf.keras.layers.Conv2D(l, kernel_size=(k, k), activation=tf.nn.relu)(X)
        X = tf.keras.layers.Flatten()(X)
        # fully connected layers
        for layer in self.config["y_fc_layers"]:
            X = tf.keras.layers.Dense(layer, activation=tf.nn.relu)(X)
        # linear layer to latent
        Y = tf.keras.layers.Dense(self.config["k"], activation="softmax")(X)
        model = tf.keras.Model(X_input, Y)
        return model

    def create_y_encoder(self, X):
        """
        Return Y calculated from X (convolutional layers + fully connected layers).
        """
        # conv layers
        for i, l in enumerate(self.config["y_conv_layers"]):
            k = self.config["y_conv_kernel_size"][i]
            X = tf.keras.layers.Conv2D(l, kernel_size=(k, k), activation=tf.nn.relu)(X)
        X = tf.keras.layers.Flatten()(X)
        # fully connected layers
        for layer in self.config["y_fc_layers"]:
            X = tf.keras.layers.Dense(layer, activation=tf.nn.relu)(X)
        # linear layer to latent
        Y = tf.keras.layers.Dense(self.config["k"], activation="softmax", name="latent_y")(X)
        return Y

    def qz_graph(self, X_input_shape):
        """
        Return Z calculated from X and Y.
        """
        X_input = tf.keras.layers.Input(X_input_shape)
        Y_input = tf.keras.layers.Input((self.config["k"],))

        # concatenate Y with X
        fn = partial(expand_and_broadcast, s=self.config["num_neighbors"])
        Y = tf.keras.layers.Lambda(fn)(Y_input)
        X = tf.keras.layers.concatenate([X_input, Y], axis=-1)  # this is now the input for the normal encoder

        # conv layers
        for i, l in enumerate(self.config["encoder_conv_layers"]):
            k = self.config["encoder_conv_kernel_size"][i]
            X = tf.keras.layers.Conv2D(l, kernel_size=(k, k), activation=tf.nn.relu)(X)
        X = tf.keras.layers.Flatten()(X)
        # fully connected layers
        for layer in self.config["encoder_fc_layers"]:
            X = tf.keras.layers.Dense(layer, activation=tf.nn.relu)(X)

        # linear layer to latent
        Z = tf.keras.layers.Dense(self.config["latent_dim"] * 2, activation=None)(X)
        model = tf.keras.Model((X_input, Y_input), Z, name="qz_model")
        return model

    def pz_graph(self):
        """Prior distibution of Z for different categories (Y should be 1-hot encoded vector)."""
        Y_input = tf.keras.layers.Input((self.config["k"],))
        X = Y_input
        # fully connected layers
        for layer in self.config["pz_fc_layers"]:
            X = tf.keras.layers.Dense(layer, activation=tf.nn.relu)(X)
        Z = tf.keras.layers.Dense(self.config["latent_dim"] * 2, activation=None)(X)
        model = tf.keras.Model(Y_input, Z, name="pz_model")
        return model

    def create_encoder(self):
        """
        Create encoder.

        Encoder outputs reparameterised latent.
        Latent is potentially returned by overall model for loss calculation, e.g. for VAE.

        Returns
        -------
        tf.keras.model, tf.Tensor
            Encoder and latent (None for BaseAEModel).
        """
        if self.is_conditional:
            X, C = self.encoder_input
            # broadcast C to fit to X
            fn = partial(expand_and_broadcast, s=self.config["num_neighbors"])
            C = tf.keras.layers.Lambda(fn)(C)
        else:
            X = self.encoder_input
        if self.config["input_noise"] is not None:
            X = self.add_noise(X)
        if self.is_conditional:
            # concatenate input and conditions
            X = tf.keras.layers.concatenate([X, C], axis=-1)

        # define qz and pz subgraphs
        shape_X = X.shape.as_list()[1:]
        qz_model = self.qz_graph(shape_X)
        pz_model = self.pz_graph()

        # get qy
        latent_y = self.create_y_encoder(X)

        # get qz for qy value (for reconstruction)
        Z = qz_model([X, latent_y])
        # reparameterise
        reparam_Z = reparameterize_gaussian(Z)
        # define encoder
        encoder = tf.keras.Model(self.encoder_input, reparam_Z, name="encoder")

        # get pz, qz for different values of y
        pZ = []
        qZ = []

        # functions for expanding Y to have batch_size dim
        def expand_and_broadcastY(Y, s):
            Y = tf.expand_dims(Y, 0)
            Y = tf.broadcast_to(Y, [s, self.config["k"]])
            return Y

        exp_fn = partial(expand_and_broadcastY, s=tf.shape(X)[0])
        for i in range(0, self.config["k"]):
            Yi = tf.one_hot(i, depth=self.config["k"])
            Yi = tf.keras.layers.Lambda(exp_fn)(Yi)
            pZ.append(pz_model(Yi))
            qZ.append(qz_model([X, Yi]))
        # stack together (shape: None, k, latent_dim)
        pZ = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=1))(pZ)
        qZ = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=1))(qZ)

        # expand latent_y to have shape: None, k, latent_dim
        def expand_and_broadcast_qY(Y, s):
            Y = tf.expand_dims(Y, axis=-1)
            Y = tf.broadcast_to(Y, [tf.shape(Y)[0], tf.shape(Y)[1], s])
            return Y

        fn = partial(expand_and_broadcast_qY, s=self.config["latent_dim"] * 2)
        qY = tf.keras.layers.Lambda(fn)(latent_y)
        # stack pZ, qZ, qY to one output
        latent_zy = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=1), name="latent_zy")([pZ, qZ, qY])

        return encoder, latent_y, latent_zy

    def create_model(self):
        """
        Create keras model using create_encoder and create_decoder functions.

        Set self.encoder, self.latent, self.decoder, self.model_output attributes.
        """
        # encoder and decoder
        self.encoder, self.latent_y, self.latent_zy = self.create_encoder()
        self.decoder = self.create_decoder()

        # add encoder_y model
        self.encoder_y = tf.keras.Model(self.encoder_input, self.latent_y, name="encoder_y")

        # create model
        if self.is_conditional:
            self.model_output = self.decoder([self.encoder(self.encoder_input), self.encoder_input[1]])
        else:
            self.model_output = self.decoder([self.encoder(self.encoder_input)])
        # model should return both output + latent_y (for cat loss) + latent_zy (for KL loss)
        self.model_output = [self.model_output, self.latent_y, self.latent_zy]
        model = tf.keras.Model(self.encoder_input, self.model_output, name=self.config["name"])
        return model


## PyTorch

class ModelEnumTorch(str, Enum):
    """
    Neural network models.

    Possible values are:

    - ``ModelEnum.BaseAEModelTorch``: "BaseAEModelTorch" (:class:`BaseAEModelTorch`)
    - ``ModelEnum.VAEModel``: "VAEModel" (:class:`VAEModel`)
    """

    BaseAEModelTorch = "BaseAEModelTorch"
    VAEModelTorch = "VAEModelTorch"
    CatVAEModelTorch = "CatVAEModelTorch"
    GMMVAEModelTorch = "GMMVAEModelTorch"
    CondCatVAEModelTorch = "CondCatVAEModelTorch"

    def get_cls(self):
        """
        Get model class from enum string.
        """
        cls = self.__class__
        if self == cls.BaseAEModelTorch:
            return BaseAEModelTorch
        elif self == cls.VAEModelTorch:
            return VAEModelTorch
        elif self == cls.CatVAEModelTorch:
            return CatVAEModelTorch
        elif self == cls.GMMVAEModelTorch:
            return GMMVAEModelTorch
        elif self == cls.CondCatVAEModelTorch:
            return CondCatVAEModelTorch
        else:
            raise NotImplementedError
        
def expand_and_broadcast_torch(x, s=1):
    """Expand tensor x with shape (batches,n) to shape (batches,s,s,n)."""
    C = x.unsqueeze(1).unsqueeze(1)
    C = C.expand(-1, s, s, -1)
    return C

def reparameterize_gumbel_softmax_torch(latent, temperature=0.1):
    """Draw a sample from the Gumbel-Softmax distribution."""

    def sample_gumbel(shape, eps=1e-20):
        """Sample from Gumbel(0, 1)."""
        U = torch.rand(shape)
        return -torch.log(-torch.log(U + eps) + eps)

    y = latent + sample_gumbel(latent.shape)
    return F.softmax(y / temperature, dim=-1)

def reparameterize_gaussian_torch(latent):
    """Draw a sample from Gaussian distribution."""
    z_mean, z_log_var = torch.chunk(latent, 2, dim=-1)
    eps = torch.randn_like(z_mean)
    return eps * torch.exp(z_log_var * 0.5) + z_mean

class GradReverseTorch(Function):
    """Reverse gradients for adversarial loss."""
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

class GradReverseTorchLayer(nn.Module):
    """Reverse gradients class."""

    def __init__(self):
        super(GradReverseTorchLayer, self).__init__()

    def forward(self, x):
        """Apply gradient reversal."""
        return GradReverseTorch.apply(x)

class BaseAEModelTorch(nn.Module):
    """
    Base class for AE and VAE models.
    """

    default_config = {"name": "BaseAEModelTorch"}

    def __init__(self, **kwargs):
        super(BaseAEModelTorch, self).__init__()
        self.log = logging.getLogger(self.__class__.__name__)
        self.config = deepcopy(BASE_MODEL_CONFIG)
        self.config.update(self.default_config)
        self.config.update(kwargs)
        if self.config["num_output_channels"] is None:
            self.config["num_output_channels"] = self.config["num_channels"]
        if self.config["num_input_channels"] is None:
            self.config["num_input_channels"] = self.config["num_channels"]
        if isinstance(self.config["encoder_conv_kernel_size"], int):
            self.config["encoder_conv_kernel_size"] = [
                self.config["encoder_conv_kernel_size"] for _ in self.config["encoder_conv_layers"]
            ]
        self.log.info("Creating model")
        self.log.debug(f"Creating model with config: {json.dumps(self.config, indent=4)}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder_input = nn.ModuleList()
        self.decoder_input = nn.ModuleList()
        if self.is_conditional:
            self.encoder_input.append(nn.Linear(self.config["num_conditions"], self.config["num_conditions"]))
            self.decoder_input.append(nn.Linear(self.config["num_conditions"], self.config["num_conditions"]))

        self.encoder, self.latent = self.create_encoder()
        self.decoder = self.create_decoder()
        if self.is_adversarial:
            self.adv_head = self.create_adversarial_head()

    def create_encoder(self):
        layers = []
        in_channels = self.config["num_channels"]
        for out_channels, kernel_size in zip(self.config["encoder_conv_layers"], self.config["encoder_conv_kernel_size"]):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.Flatten())
        for fc_layer in self.config["encoder_fc_layers"]:
            layers.append(nn.Linear(in_channels, fc_layer))
            layers.append(nn.ReLU())
            in_channels = fc_layer
        layers.append(nn.Linear(in_channels, self.config["latent_dim"]))
        encoder = nn.Sequential(*layers)
        return encoder, None

    def create_decoder(self):
        layers = []
        in_features = self.config["latent_dim"]
        for fc_layer in self.config["decoder_fc_layers"]:
            layers.append(nn.Linear(fc_layer, in_features))
            layers.append(nn.ReLU())
            in_features = fc_layer
        layers.append(nn.Linear(self.config["num_output_channels"], in_features))
        decoder = nn.Sequential(*layers)
        return decoder

    def create_adversarial_head(self):
        layers = []
        in_features = self.config["latent_dim"]
        layers.append(GradReverseTorch())
        for adv_layer in self.config["adversarial_layers"]:
            layers.append(nn.Linear(in_features, adv_layer))
            layers.append(nn.ReLU())
            in_features = adv_layer
        layers.append(nn.Linear(in_features, self.config["num_conditions"]))
        adv_head = nn.Sequential(*layers)
        return adv_head

    def encode_condition(self, C):
        if self.config["encode_condition"] is None:
            return C
        if not hasattr(self, "condition_encoder"):
            enc_l = self.config["encode_condition"]
            if isinstance(enc_l, int):
                enc_l = [enc_l]
            layers = []
            in_features = self.config["num_conditions"]
            for layer in enc_l:
                layers.append(nn.Linear(in_features, layer))
                layers.append(nn.ReLU())
                in_features = layer
            self.condition_encoder = nn.Sequential(*layers)
        return self.condition_encoder(C)

    def add_noise(self, X):
        if self.config["input_noise"] == "dropout":
            return F.dropout(X, p=self.config["noise_scale"])
        elif self.config["input_noise"] == "gaussian":
            return X + torch.randn_like(X) * self.config["noise_scale"]
        else:
            raise NotImplementedError

    @property
    def is_conditional(self):
        return self.config["num_conditions"] > 0

    @property
    def is_adversarial(self):
        return self.config["adversarial_layers"] is not None and self.is_conditional

    def forward(self, x, c=None):
        if self.is_conditional:
            x = [x, c]
        x = self.encoder(x)
        if self.is_adversarial:
            adv_output = self.adv_head(x)
            return x, adv_output
        return x
    

class VAEModelTorch(BaseAEModelTorch):
    """
    VAE with simple Gaussian prior (trainable with KL loss).

    Inherits from :class:`BaseAEModelTorch`.

    Model architecture:

    - Encoder: ``(noise) - conv layers - fc layers - linear layer to latent_dim * 2``
    - Latent: split ``latent_dim`` in half, re-sample using Gaussian prior
    - Decoder: ``fc_layers - linear (regularized) layer to num_output_channels``
    """

    default_config = {"name": "VAEModelTorch"}

    def create_encoder(self) -> Tuple[nn.Module, Optional[torch.Tensor]]:
        """
        Create encoder.

        Encoder outputs reparameterized latent.
        Latent is potentially returned by overall model for loss calculation, e.g. for VAE.

        Returns
        -------
        Encoder and latent (None for ``BaseAEModelTorch``).
        """
        layers = []
        in_channels = self.config["num_channels"]
        for out_channels, kernel_size in zip(self.config["encoder_conv_layers"], self.config["encoder_conv_kernel_size"]):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.Flatten())
        for i, fc_layer in enumerate(self.config["encoder_fc_layers"]):
            if self.config["num_neighbors"] != 0 and i==0: # only for the first layer after flattening
                layers.append(nn.Linear(in_channels*self.config["num_neighbors"]*self.config["num_neighbors"], fc_layer))
            else:
                layers.append(nn.Linear(in_channels, fc_layer))
            layers.append(nn.ReLU())
            in_channels = fc_layer
        latent_layer = nn.Linear(in_channels, self.config["latent_dim"])
        layers.append(latent_layer)
        encoder = nn.Sequential(*layers)
        return encoder, latent_layer
    
    def create_decoder(self):
        layers = []
        in_features = self.config["latent_dim"]
        for fc_layer in self.config["decoder_fc_layers"]:
            layers.append(nn.Linear(in_features, fc_layer))
            layers.append(nn.ReLU())
            in_features = fc_layer
        if not self.config["decoder_fc_layers"]: # if empty
            layers.append(nn.Linear(self.config["latent_dim"], self.config["num_output_channels"]))
        else:
            layers.append(nn.Linear(in_features, self.config["num_output_channels"]))
        decoder = nn.Sequential(*layers)
        return decoder

    def forward(self, x, c=None):
        if self.is_conditional:
            x = torch.cat([x, c], dim=-1)
        x = self.encoder(x)
        reparam_latent = reparameterize_gaussian_torch(x)
        if self.is_adversarial:
            adv_output = self.adv_head(reparam_latent)
            return reparam_latent, adv_output
        x = self.decoder(x)
        return reparam_latent, x
    

class CatVAEModelTorch(BaseAEModelTorch):
    """
    Categorical VAE Model.

    VAE with categorical prior (softmax gumbel) (trainable with categorical loss)
    Encoder: (noise) - conv layers - fc layers - linear layer to latent_dim * 2
    Latent: split latent_dim in half, resample using Gaussian prior
    Decoder: fc_layers - linear (regularized) layer to num_output_channels
    """

    default_config = {
        "name": "CatVAEModelTorch",
        # temperature for scaling gumbel_softmax. values close to 0 are close to true categorical distribution
        "temperature": 0.1,
        "initial_temperature": 10,
        "anneal_epochs": 0,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.temperature = nn.Parameter(
            torch.tensor(kwargs.get("initial_temperature", self.default_config["initial_temperature"])),
            requires_grad=False
        )

    def create_encoder(self) -> Tuple[nn.Module, Optional[torch.Tensor]]:
        """
        Create encoder.

        Encoder outputs reparameterised latent.
        Latent is potentially returned by overall model for loss calculation, e.g. for VAE.

        Returns
        -------
        Encoder and latent (None for BaseAEModel).
        """
        layers = []
        in_channels = self.config["num_channels"]
        for out_channels, kernel_size in zip(self.config["encoder_conv_layers"], self.config["encoder_conv_kernel_size"]):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.Flatten())
        for fc_layer in self.config["encoder_fc_layers"]:
            layers.append(nn.Linear(in_channels, fc_layer))
            layers.append(nn.ReLU())
            in_channels = fc_layer
        latent_layer = nn.Linear(in_channels, self.config["latent_dim"])
        layers.append(latent_layer)
        encoder = nn.Sequential(*layers)
        return encoder, latent_layer

    def forward(self, x, c=None):
        if self.is_conditional:
            x = torch.cat([x, c], dim=-1)
        x = self.encoder(x)
        reparam_latent = reparameterize_gumbel_softmax_torch(x, self.temperature)
        if self.is_adversarial:
            adv_output = self.adv_head(reparam_latent)
            return reparam_latent, adv_output
        return reparam_latent, x
    

class CondCatVAEModelTorch(CatVAEModelTorch):
    """
    Conditional Categorical VAE model.

    Conditional Categorical VAE using another concatenation scheme when adding the condition
    to the latent space. This model first calculates a fully connected layer to a vector
    with length #output_channels x #conditions

    IGNORES decoder_fc_layers - only supports linear decoder!
    """

    def create_decoder(self):
        """
        Create decoder.
        """
        class Decoder(nn.Module):
            def __init__(self, config):
                super(Decoder, self).__init__()
                self.config = config
                self.fc = nn.Linear(config["latent_dim"], config["num_output_channels"] * config["encode_condition"])
                self.reshape = (config["num_output_channels"], config["encode_condition"])

            def forward(self, x, c):
                x = self.fc(x)
                x = x.view(-1, *self.reshape)
                c = self.encode_condition(c)
                decoder_output = torch.einsum('boc,bc->bo', x, c)
                return decoder_output

        return Decoder(self.config)

    def forward(self, x, c=None):
        if self.is_conditional:
            x = torch.cat([x, c], dim=-1)
        x = self.encoder(x)
        reparam_latent = reparameterize_gumbel_softmax_torch(x, self.temperature)
        if self.is_adversarial:
            adv_output = self.adv_head(reparam_latent)
            return reparam_latent, adv_output
        return reparam_latent, x
    

class GMMVAEModelTorch(BaseAEModelTorch):
    """
    Gaussian Mixture Model VAE.

    VAE with gmm prior (trainable with categorical loss for y and weighted kl loss for z)
    Encoder y: (noise) - conv layers y - fc layers y - linear layer to latent_dim
    Encoder: (noise) + y - conv layers - fc layers - linear layer to latent_dim * 2
    Latent: split latent_dim in half, resample using Gaussian prior
    Decoder: fc_layers - linear (regularized) layer to num_output_channels
    """

    default_config = {
        "name": "GMMVAEModelTorch",
        # y encoder architecture
        "y_conv_layers": None,
        "y_conv_kernel_size": None,
        "y_fc_layers": None,
        # pz (gmm prior for zmean and zvar from categorical y)
        "pz_fc_layers": None,
        # number of different gaussians
        "k": 10,
        # temperature for categorical loss on y <- might not need to anneal!
        # temperature for scaling gumbel_softmax. values close to 0 are close to true categorical distribution
        "temperature": 0.1,
        "initial_temperature": 10,
        "anneal_epochs": 0,
    }

    def __init__(self, **kwargs):
        config = deepcopy(BASE_MODEL_CONFIG)
        config.update(self.default_config)
        config.update(kwargs)
        if config["y_conv_layers"] is None:
            config["y_conv_layers"] = config["encoder_conv_layers"]
        if config["y_conv_kernel_size"] is None:
            config["y_conv_kernel_size"] = config["encoder_conv_kernel_size"]
        if config["y_fc_layers"] is None:
            config["y_fc_layers"] = config["encoder_fc_layers"]
        if config["pz_fc_layers"] is None:
            config["pz_fc_layers"] = config["encoder_fc_layers"]
        super().__init__(**config)
        self.temperature = nn.Parameter(
            torch.tensor(kwargs.get("initial_temperature", self.default_config["initial_temperature"])),
            requires_grad=False
        )

    def qy_graph(self, input_shape):
        """
        Return Y calculated from X (convolutional layers + fully connected layers).
        """
        layers = []
        in_channels = input_shape[0]
        for out_channels, kernel_size in zip(self.config["y_conv_layers"], self.config["y_conv_kernel_size"]):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.Flatten())
        for fc_layer in self.config["y_fc_layers"]:
            layers.append(nn.Linear(in_channels, fc_layer))
            layers.append(nn.ReLU())
            in_channels = fc_layer
        layers.append(nn.Linear(in_channels, self.config["k"]))
        layers.append(nn.Softmax(dim=-1))
        return nn.Sequential(*layers)

    def create_y_encoder(self, X):
        """
        Return Y calculated from X (convolutional layers + fully connected layers).
        """
        layers = []
        in_channels = X.shape[1]
        for out_channels, kernel_size in zip(self.config["y_conv_layers"], self.config["y_conv_kernel_size"]):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.Flatten())
        for fc_layer in self.config["y_fc_layers"]:
            layers.append(nn.Linear(in_channels, fc_layer))
            layers.append(nn.ReLU())
            in_channels = fc_layer
        layers.append(nn.Linear(in_channels, self.config["k"]))
        layers.append(nn.Softmax(dim=-1))
        return nn.Sequential(*layers)

    def qz_graph(self, X_input_shape):
        """
        Return Z calculated from X and Y.
        """
        class QZModel(nn.Module):
            def __init__(self, config):
                super(QZModel, self).__init__()
                self.config = config
                self.conv_layers = nn.ModuleList()
                in_channels = X_input_shape[0]
                for out_channels, kernel_size in zip(config["encoder_conv_layers"], config["encoder_conv_kernel_size"]):
                    self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size))
                    self.conv_layers.append(nn.ReLU())
                    in_channels = out_channels
                self.flatten = nn.Flatten()
                self.fc_layers = nn.ModuleList()
                for fc_layer in config["encoder_fc_layers"]:
                    self.fc_layers.append(nn.Linear(in_channels, fc_layer))
                    self.fc_layers.append(nn.ReLU())
                    in_channels = fc_layer
                self.linear = nn.Linear(in_channels, config["latent_dim"] * 2)

            def forward(self, x, y):
                y = expand_and_broadcast_torch(y, s=self.config["num_neighbors"])
                x = torch.cat([x, y], dim=1)
                for layer in self.conv_layers:
                    x = layer(x)
                x = self.flatten(x)
                for layer in self.fc_layers:
                    x = layer(x)
                z = self.linear(x)
                return z

        return QZModel(self.config)

    def pz_graph(self):
        """Prior distribution of Z for different categories (Y should be 1-hot encoded vector)."""
        class PZModel(nn.Module):
            def __init__(self, config):
                super(PZModel, self).__init__()
                self.config = config
                self.fc_layers = nn.ModuleList()
                in_features = config["k"]
                for fc_layer in config["pz_fc_layers"]:
                    self.fc_layers.append(nn.Linear(in_features, fc_layer))
                    self.fc_layers.append(nn.ReLU())
                    in_features = fc_layer
                self.linear = nn.Linear(in_features, config["latent_dim"] * 2)

            def forward(self, y):
                x = y
                for layer in self.fc_layers:
                    x = layer(x)
                z = self.linear(x)
                return z

        return PZModel(self.config)

    def create_encoder(self):
        """
        Create encoder.

        Encoder outputs reparameterised latent.
        Latent is potentially returned by overall model for loss calculation, e.g. for VAE.

        Returns
        -------
        nn.Module, torch.Tensor, torch.Tensor
            Encoder and latent (None for BaseAEModel).
        """
        if self.is_conditional:
            X, C = self.encoder_input
            C = expand_and_broadcast_torch(C, s=self.config["num_neighbors"])
        else:
            X = self.encoder_input
        if self.config["input_noise"] is not None:
            X = self.add_noise(X)
        if self.is_conditional:
            X = torch.cat([X, C], dim=-1)

        shape_X = X.shape[1:]
        qz_model = self.qz_graph(shape_X)
        pz_model = self.pz_graph()

        latent_y = self.create_y_encoder(X)

        Z = qz_model(X, latent_y)
        reparam_Z = reparameterize_gaussian_torch(Z)

        encoder = nn.Sequential(self.encoder_input, reparam_Z)

        pZ = []
        qZ = []

        def expand_and_broadcastY(Y, s):
            Y = Y.unsqueeze(0)
            Y = Y.expand(s, -1)
            return Y

        for i in range(self.config["k"]):
            Yi = F.one_hot(torch.tensor(i), num_classes=self.config["k"]).float()
            Yi = expand_and_broadcastY(Yi, s=X.shape[0])
            pZ.append(pz_model(Yi))
            qZ.append(qz_model(X, Yi))

        pZ = torch.stack(pZ, dim=1)
        qZ = torch.stack(qZ, dim=1)

        def expand_and_broadcast_qY(Y, s):
            Y = Y.unsqueeze(-1)
            Y = Y.expand(-1, -1, s)
            return Y

        qY = expand_and_broadcast_qY(latent_y, s=self.config["latent_dim"] * 2)
        latent_zy = torch.stack([pZ, qZ, qY], dim=1)

        return encoder, latent_y, latent_zy

    def create_model(self):
        """
        Create PyTorch model using create_encoder and create_decoder functions.

        Set self.encoder, self.latent, self.decoder, self.model_output attributes.
        """
        self.encoder, self.latent_y, self.latent_zy = self.create_encoder()
        self.decoder = self.create_decoder()

        self.encoder_y = nn.Sequential(self.encoder_input, self.latent_y)

        if self.is_conditional:
            self.model_output = self.decoder(self.encoder(self.encoder_input), self.encoder_input[1])
        else:
            self.model_output = self.decoder(self.encoder(self.encoder_input))

        self.model_output = [self.model_output, self.latent_y, self.latent_zy]
        model = nn.Sequential(self.encoder_input, self.model_output)
        return model