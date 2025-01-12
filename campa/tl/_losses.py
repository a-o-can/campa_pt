# TODO go though functions and comment
from enum import Enum

import numpy as np
import tensorflow as tf

import torch
import torch.nn.functional as F


class LossEnum(str, Enum):
    """
    Loss functions.

    Possible values are:

    - ``LossEnum.MSE``: "mean_squared_error" (:func:`tf.losses.mean_squared_error`)
    - ``LossEnum.SIGMA_MSE``: "sigma_vae_mse", MSE loss for sigma VAE
    - ``LossEnum.KL``: "kl_divergence"
    - ``LossEnum.MSE_METRIC``: "mean_squared_error_metric" (:class:`tf.keras.metrics.MeanSquaredError`)
    """

    MSE = "mean_squared_error"
    KL = "kl_divergence"
    SIGMA_MSE = "sigma_vae_mse"
    ENT = "entropy"
    CAT_KL = "categorical_kl"
    GMM_KL = "gmm_kl_divergence"
    SOFTMAX = "softmax"

    MSE_metric = "mean_squared_error_metric"

    ACC_metric = "accuracy_metric"

    def get_fn(self):
        """Return loss function."""
        cls = self.__class__
        if self == cls.MSE:
            return tf.losses.mean_squared_error
        elif self == cls.KL:
            return kl_loss
        elif self == cls.SIGMA_MSE:
            return sigma_vae_mse
        elif self == cls.ENT:
            return min_entropy
        elif self == cls.CAT_KL:
            return categorical_kl_loss
        elif self == cls.GMM_KL:
            return gmm_kl_loss
        elif self == cls.SOFTMAX:
            return tf.nn.softmax_cross_entropy_with_logits
        elif self == cls.MSE_metric:
            return tf.keras.metrics.MeanSquaredError()
        elif self == cls.ACC_metric:
            return tf.keras.metrics.CategoricalAccuracy()
        else:
            raise NotImplementedError


@tf.function
def kl_loss(y_true, y_pred, logs=None):
    """KL divergence."""
    mean, var = tf.split(y_pred, 2, axis=-1)
    l_kl = -0.5 * tf.reduce_mean(1 + var - tf.square(mean) - tf.exp(var))
    return l_kl


@tf.function
def categorical_kl_loss(y_true, y_pred):
    """KL loss for categorical VAE."""
    # kl divergence between y_pred and bernulli distribution with p=0.5
    logits_y = y_pred
    q_y = tf.nn.softmax(logits_y)
    log_q_y = tf.math.log(q_y + 1e-20)
    kl = q_y * (log_q_y - tf.math.log(0.5))
    kl = tf.reduce_mean(kl)
    return kl


def gmm_kl_loss(y_true, y_pred):
    """KL loss for GMM.

    From: https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    """
    p, q, latent_y = tf.unstack(y_pred, axis=1)
    p_mu, p_log_var = tf.split(p, 2, axis=-1)
    q_mu, q_log_var = tf.split(q, 2, axis=-1)
    # weighted sum of kl divergence between qZ and pZ
    tf.keras.backend.epsilon()

    kl = p_log_var - q_log_var - 0.5 * (1.0 - (tf.exp(q_log_var) ** 2 + (q_mu - p_mu) ** 2) / tf.exp(p_log_var) ** 2)

    # kl = tf.math.log(qZ_var / (pZ_var + eps) + eps) + (pZ_var + (pZ_mu - qZ_mu)**2) / (2*qZ_var + eps) - 0.5
    # mean over batch and latent_dim
    kl = tf.math.reduce_mean(kl, axis=[0, 2])
    # sum over k
    kl = tf.math.reduce_sum(kl)
    return kl


def gaussian_nll(mu, log_sigma, x):
    """Gaussian negative log-likelihood.

    From: https://github.com/orybkin/sigma-vae-tensorflow/blob/master/model.py
    """
    return 0.5 * ((x - mu) / tf.math.exp(log_sigma)) ** 2 + log_sigma + 0.5 * np.log(2 * np.pi)


@tf.function
def sigma_vae_mse(y_true, y_pred):
    """
    MSE loss for sigma-VAE (calibrated decoder).
    """
    log_sigma = tf.math.log(tf.math.sqrt(tf.reduce_mean((y_true - y_pred) ** 2, [0, 1], keepdims=True)))
    return tf.reduce_sum(gaussian_nll(y_pred, log_sigma, y_true))


@tf.function
def min_entropy(y_true, y_pred):
    """
    Entropy.
    """
    l_ent = -1 * tf.reduce_mean(tf.math.log(y_pred + tf.keras.backend.epsilon()) * y_pred)
    return l_ent


## PyTorch

class LossEnumTorch(str, Enum):
    """
    Loss functions for PyTorch.

    Possible values are:

    - ``LossEnumTorch.MSE``: "mean_squared_error" (:func:`torch.nn.functional.mse_loss`)
    - ``LossEnumTorch.SIGMA_MSE``: "sigma_vae_mse_torch", MSE loss for sigma VAE
    - ``LossEnumTorch.KL``: "kl_divergence"
    - ``LossEnumTorch.MSE_METRIC``: "mean_squared_error_metric" (:class:`torch.nn.MSELoss`)
    """

    MSE_Torch = "mean_squared_error_torch"
    KL_Torch = "kl_divergence_torch"
    SIGMA_MSE_Torch = "sigma_vae_mse_torch"
    ENT_Torch = "entropy_torch"
    CAT_KL_Torch = "categorical_kl_torch"
    GMM_KL_Torch = "gmm_kl_divergence_torch"
    SOFTMAX_Torch = "softmax_torch"

    MSE_metric_Torch = "mean_squared_error_metric_torch"

    ACC_metric_Torch = "accuracy_metric_torch"

    def get_fn(self):
        """Return loss function."""
        cls = self.__class__
        if self == cls.MSE_Torch:
            return F.mse_loss
        elif self == cls.KL_Torch:
            return kl_loss_torch
        elif self == cls.SIGMA_MSE_Torch:
            return sigma_vae_mse_torch
        elif self == cls.ENT_Torch:
            return min_entropy_torch
        elif self == cls.CAT_KL_Torch:
            return categorical_kl_loss_torch
        elif self == cls.GMM_KL_Torch:
            return gmm_kl_loss_torch
        elif self == cls.SOFTMAX_Torch:
            return F.cross_entropy
        elif self == cls.MSE_metric_Torch:
            return torch.nn.MSELoss()
        elif self == cls.ACC_metric_Torch:
            return torch.nn.CrossEntropyLoss()
        else:
            raise NotImplementedError


def kl_loss_torch(y_true, y_pred):
    """KL divergence."""
    mean, var = torch.chunk(y_pred, 2, dim=-1)
    l_kl = -0.5 * torch.mean(1 + var - mean.pow(2) - var.exp())
    return l_kl


def categorical_kl_loss_torch(y_true, y_pred):
    """KL loss for categorical VAE."""
    logits_y = y_pred
    q_y = F.softmax(logits_y, dim=-1)
    log_q_y = torch.log(q_y + 1e-20)
    kl = q_y * (log_q_y - torch.log(torch.tensor(0.5)))
    kl = torch.mean(kl)
    return kl


def gmm_kl_loss_torch(y_true, y_pred):
    """KL loss for GMM.

    From: https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    """
    p, q, latent_y = torch.unbind(y_pred, dim=1)
    p_mu, p_log_var = torch.chunk(p, 2, dim=-1)
    q_mu, q_log_var = torch.chunk(q, 2, dim=-1)

    kl = p_log_var - q_log_var - 0.5 * (1.0 - (q_log_var.exp() ** 2 + (q_mu - p_mu) ** 2) / p_log_var.exp() ** 2)

    kl = torch.mean(kl, dim=[0, 2])
    kl = torch.sum(kl)
    return kl


def gaussian_nll_torch(mu, log_sigma, x):
    """Gaussian negative log-likelihood.

    From: https://github.com/orybkin/sigma-vae-tensorflow/blob/master/model.py
    """
    return 0.5 * ((x - mu) / log_sigma.exp()) ** 2 + log_sigma + 0.5 * np.log(2 * np.pi)


def sigma_vae_mse_torch(y_true, y_pred):
    """
    MSE loss for sigma-VAE (calibrated decoder).
    """
    log_sigma = torch.log(torch.sqrt(torch.mean((y_true - y_pred) ** 2, dim=[0, 1], keepdim=True)))
    return torch.sum(gaussian_nll_torch(y_pred, log_sigma, y_true))


def min_entropy_torch(y_true, y_pred):
    """
    Entropy.
    """
    l_ent = -1 * torch.mean(torch.log(y_pred + torch.finfo(torch.float32).eps) * y_pred)
    return l_ent