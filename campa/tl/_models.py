# TODO go through functions and comment
from copy import deepcopy
from enum import Enum
from typing import Any, Dict, Tuple, Optional
from functools import partial
import json
import logging


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function




# # --- Model classes ---

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

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

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
        print(f"Device: {self.device}")
        # self.encoder_input = nn.ModuleList()
        # self.decoder_input = nn.ModuleList()
        # if self.is_conditional:
            # self.encoder_input.append(nn.Linear(self.config["num_conditions"], self.config["num_conditions"]))
            # self.decoder_input.append(nn.Linear(self.config["num_conditions"], self.config["num_conditions"]))

        self.encoder, self.latent = self.create_encoder()
        self.decoder = self.create_decoder()
        if self.is_adversarial:
            self.adv_head = self.create_adversarial_head()

    def _create_base_encoder(self):
        layers = []
        if self.is_conditional:
            self.create_condition_encoder(encoding_level="encoder")
            in_channels = self.config["num_channels"] + self.config["encode_condition"][-1]
        else:
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
        encoder = nn.Sequential(*layers)
        return encoder

    def create_encoder(self):
        base_encoder = self._create_base_encoder()
        encoder = nn.Sequential(
            base_encoder,
            nn.Linear(self.config["encoder_fc_layers"][-1], self.config["latent_dim"]))
        return encoder, None

    def create_decoder(self):
        layers = []
        in_features = self.config["latent_dim"]
        for fc_layer in self.config["decoder_fc_layers"]:
            layers.append(nn.Linear(fc_layer, in_features))
            layers.append(nn.ReLU())
            in_features = fc_layer
        layers.append(nn.Linear(in_features, self.config["num_output_channels"]))
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

    def create_condition_encoder(self, encoding_level="encoder"):
        if self.config["encode_condition"] is None:
            return nn.Identity()
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
            condition_encoder = nn.Sequential(*layers)
        if encoding_level == "encoder":
            self.condition_encoder_latent = condition_encoder
        elif encoding_level == "decoder":
            self.condition_encoder_decoder = condition_encoder
        print("Condition encoder created")

    def encode_condition(self, encoding_level="encoder"):
        if encoding_level == "encoder":
            return self.condition_encoder_latent
        elif encoding_level == "decoder":
            return self.condition_encoder_decoder

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
    
    def total_trainable_params(self):
        print(sum(p.numel() for p in self.parameters() if p.requires_grad))


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
        encoder = self._create_base_encoder()
        latent = nn.Linear(self.config["encoder_fc_layers"][-1], self.config["latent_dim"] * 2)
        # encoder.apply(init_weights)
        # latent.apply(init_weights)
        return encoder, latent
    
    def create_decoder(self):
        if self.is_conditional:
            # C = self.encode_condition(encoding_level="decoder")
            in_features = self.config["latent_dim"] + self.config["encode_condition"][-1]
        else:
            in_features = self.config["latent_dim"]
        layers = []
        for fc_layer in self.config["decoder_fc_layers"]:
            layers.append(nn.Linear(fc_layer, in_features))
            layers.append(nn.ReLU())
            in_features = fc_layer
        layers.append(nn.Linear(in_features, self.config["num_output_channels"]))
        decoder = nn.Sequential(*layers)
        # decoder.apply(init_weights)
        return decoder

    def forward(self, x, c=None):
        if self.is_conditional:
            assert c!=None, "Cannot be None, must be a tensor"
            cond = self.encode_condition(encoding_level="encoder")(c)
            x = torch.cat([x, cond[:,:,None,None].expand(-1,-1,3,3)], dim=1) # dim=1 is the channel dimension.
        x = self.encoder(x)
        latent = self.latent(x)
        reparam_latent = reparameterize_gaussian_torch(latent)
        if self.is_adversarial:
            adv_output = self.adv_head(reparam_latent)
            return reparam_latent, adv_output
        if self.is_conditional:
            reparam_latent = torch.cat([reparam_latent, cond], dim=1) # dim=1 is the channel dimension.
        reparam_latent = self.decoder(reparam_latent)
        return {"decoder": reparam_latent, "latent":x}
    

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
    

# class CondCatVAEModelTorch(CatVAEModelTorch):
#     """
#     Conditional Categorical VAE model.

#     Conditional Categorical VAE using another concatenation scheme when adding the condition
#     to the latent space. This model first calculates a fully connected layer to a vector
#     with length #output_channels x #conditions

#     IGNORES decoder_fc_layers - only supports linear decoder!
#     """

#     def create_decoder(self):
#         """
#         Create decoder.
#         """
#         class Decoder(nn.Module):
#             def __init__(self, config):
#                 super(Decoder, self).__init__()
#                 self.config = config
#                 self.fc = nn.Linear(config["latent_dim"], config["num_output_channels"] * config["encode_condition"])
#                 self.reshape = (config["num_output_channels"], config["encode_condition"])

#             def forward(self, x, c):
#                 x = self.fc(x)
#                 x = x.view(-1, *self.reshape)
#                 c = self.encode_condition(c)
#                 decoder_output = torch.einsum('boc,bc->bo', x, c)
#                 return decoder_output

#         return Decoder(self.config)

#     def forward(self, x, c=None):
#         if self.is_conditional:
#             x = torch.cat([x, c], dim=-1)
#         x = self.encoder(x)
#         reparam_latent = reparameterize_gumbel_softmax_torch(x, self.temperature)
#         if self.is_adversarial:
#             adv_output = self.adv_head(reparam_latent)
#             return reparam_latent, adv_output
#         return reparam_latent, x
    

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