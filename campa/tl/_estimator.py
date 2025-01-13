from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from campa.tl import Experiment
    import numpy as np

import os
import logging

import pandas as pd
import tensorflow as tf

from campa.tl import LossEnum, ModelEnum, LossEnumTorch, ModelEnumTorch
from campa.data import NNDataset, NNTorchDataset
from campa.tl._layers import UpdateSparsityLevel, UpdateSparsityLevelTorch

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pdb


# --- Callbacks ---
class LossWarmup(tf.keras.callbacks.Callback):
    """Callback to warmup loss weights."""

    def __init__(self, weight_vars, to_weights, to_epochs):
        super().__init__()
        self.to_weights = to_weights
        self.to_epochs = to_epochs
        self.weight_vars = weight_vars

    def on_epoch_begin(self, epoch, logs=None):
        """Update loss weights."""
        for key in self.to_epochs.keys():
            to_epoch = self.to_epochs[key]
            to_weight = self.to_weights[key]
            if to_epoch == 0 or to_epoch <= epoch:
                tf.keras.backend.set_value(self.weight_vars[key], to_weight)
            else:
                tf.keras.backend.set_value(self.weight_vars[key], to_weight / to_epoch * epoch)
            print(f"set {key} loss weight to {tf.keras.backend.get_value(self.weight_vars[key])}")

        if "latent" in self.weight_vars.keys():
            print(f"set latent loss weight to {tf.keras.backend.get_value(self.weight_vars['latent'])}")


class AnnealTemperature(tf.keras.callbacks.Callback):
    """Callback to anneal learning rate."""

    def __init__(self, temperature, initial_temperature, final_temperature, to_epoch):
        super().__init__()
        self.temperature = temperature
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.to_epoch = to_epoch

    def on_epoch_begin(self, epoch, logs=None):
        """Update temperature."""
        if self.to_epoch == 0 or self.to_epoch <= epoch:
            tf.keras.backend.set_value(self.temperature, self.final_temperature)
        else:
            tf.keras.backend.set_value(
                self.temperature,
                self.initial_temperature + (self.final_temperature - self.initial_temperature) / self.to_epoch * epoch,
            )
        print(f"set temperature to {tf.keras.backend.get_value(self.temperature)}")


# --- Estimator class ---
class Estimator:
    """
    Neural network estimator.

    Handles training and evaluation of models.

    Parameters
    ----------
    exp
        Experiment with model config.
    """

    def __init__(self, exp: Experiment):
        self.log = logging.getLogger(self.__class__.__name__)
        self.exp = exp
        self.config = exp.estimator_config

        self.config["training"]["loss"] = {
            key: LossEnum(val).get_fn() for key, val in self.config["training"]["loss"].items()
        }
        self.config["training"]["metrics"] = {
            key: LossEnum(val).get_fn() for key, val in self.config["training"]["metrics"].items()
        }
        self.callbacks: list[object] = []

        # create model
        self.optimizer = None
        self.epoch = 0
        self.create_model()
        self.compiled_model = False

        # train and val datasets
        # config params impacting y
        self.output_channels = self.config["data"]["output_channels"]
        self.repeat_y = len(self.config["training"]["loss"].keys())
        if self.repeat_y == 1:
            self.repeat_y = False
        self.add_c_to_y = False
        if "adv_head" in self.config["training"]["loss"].keys():
            self.add_c_to_y = True
            self.repeat_y = self.repeat_y - 1
        self.ds = NNDataset(
            self.config["data"]["dataset_name"],
            data_config=self.config["data"]["data_config"],
        )
        self._train_dataset, self._val_dataset, self._test_dataset = None, None, None

        # set up model weights and history paths for saving/loading later
        self.weights_name = os.path.join(self.exp.full_path, "weights_epoch{:03d}")  # noqa: P103
        self.history_name = os.path.join(self.exp.full_path, "history.csv")

    @property
    def train_dataset(self) -> tf.data.Dataset:
        """
        Shuffled :class:`tf.data.Dataset` of train split.
        """
        if self._train_dataset is None:
            self._train_dataset = self._get_dataset("train", shuffled=True)
        return self._train_dataset

    @property
    def val_dataset(self) -> tf.data.Dataset:
        """
        :class:`tf.data.Dataset` of val split.
        """
        if self._val_dataset is None:
            self._val_dataset = self._get_dataset("val")
        return self._val_dataset

    @property
    def test_dataset(self) -> tf.data.Dataset:
        """
        :class:`tf.data.Dataset` of test split.
        """
        if self._test_dataset is None:
            self._test_dataset = self._get_dataset("test")
        return self._test_dataset

    def _get_dataset(self, split: str, shuffled: bool = False) -> tf.data.Dataset:
        return self.ds.get_tf_dataset(
            split=split,
            output_channels=self.output_channels,
            is_conditional=self.model.is_conditional,
            repeat_y=self.repeat_y,
            add_c_to_y=self.add_c_to_y,
            shuffled=shuffled,
        )

    def create_model(self):
        """
        Initialise neural network model.

        Adds ``self.model``.
        """
        ModelClass = ModelEnum(self.config["model"]["model_cls"]).get_cls()
        self.model = ModelClass(**self.config["model"]["model_kwargs"])
        weights_path = self.config["model"]["init_with_weights"]
        if weights_path is True:
            weights_path = tf.train.latest_checkpoint(self.exp.full_path)
            if weights_path is None:
                self.log.warning(
                    f"WARNING: weights_path set to true but no trained model found in {self.exp.full_path}"
                )
        if isinstance(weights_path, str):
            # first need to compile the model
            self._compile_model()
            self.log.info(f"Initializing model with weights from {weights_path}")
            w1 = self.model.model.layers[5].get_weights()
            self.model.model.load_weights(weights_path).assert_nontrivial_match().assert_existing_objects_matched()
            w2 = self.model.model.layers[5].get_weights()
            assert (w1[0] != w2[0]).any()
            assert (w1[1] != w2[1]).any()
            self.epoch = self.exp.epoch
            # TODO when fine-tuning need to reset self.epoch!

    def _compile_model(self):
        config = self.config["training"]
        # set loss weights
        self.loss_weights = {key: tf.keras.backend.variable(val) for key, val in config["loss_weights"].items()}
        # callback to update weights before each epoch
        self.callbacks.append(
            LossWarmup(
                self.loss_weights,
                config["loss_weights"],
                config["loss_warmup_to_epoch"],
            )
        )
        self.callbacks.append(UpdateSparsityLevel())
        if hasattr(self.model, "temperature"):
            self.callbacks.append(
                AnnealTemperature(
                    self.model.temperature,
                    self.model.config["initial_temperature"],
                    self.model.config["temperature"],
                    self.model.config["anneal_epochs"],
                )
            )
        # create optimizer
        if self.optimizer is None:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])
        self.model.model.compile(
            optimizer=self.optimizer,
            loss=config["loss"],
            loss_weights=self.loss_weights,
            metrics=config["metrics"],
        )
        self.compiled_model = True

    def train_model(self):
        """
        Train neural network model.

        Needs an initialised model in ``self.model``.
        """
        config = self.config["training"]
        if not self.compiled_model:
            self._compile_model()
        # reset epoch when overwriting history
        if config["overwrite_history"]:
            self.epoch = 0
        self.log.info(f"Training model for {config['epochs']} epochs")
        # pdb.set_trace()
        history = self.model.model.fit(
            # TODO this is only shuffling the first 10000 samples, but as data is shuffled already should be ok
            x=self.train_dataset.shuffle(10000).batch(config["batch_size"]).prefetch(1),
            validation_data=self.val_dataset.batch(config["batch_size"]).prefetch(1),
            epochs=config["epochs"],
            verbose=1,
            callbacks=self.callbacks,
        )
        self.epoch += config["epochs"]
        history = pd.DataFrame.from_dict(history.history)
        history["epoch"] = range(self.epoch - config["epochs"], self.epoch)
        history = history.set_index("epoch")
        if config["save_model_weights"]:
            weights_name = self.weights_name.format(self.epoch)
            self.log.info(f"Saving model to {weights_name}")
            self.model.model.save_weights(weights_name)
        if config["save_history"]:
            if os.path.exists(self.history_name) and not config["overwrite_history"]:
                # if there is a previous history, concatenate to this
                prev_history = pd.read_csv(self.history_name, index_col=0)
                history = pd.concat([prev_history, history])
            history.to_csv(self.history_name)
        return history

    def predict_model(self, data: tf.data.Dataset | np.ndarray, batch_size: int | None = None) -> Any:
        """
        Predict all elements in ``data``.

        Parameters
        ----------
        data
            Data to predict, with first dimension the number of elements.
        batch_size
            Batch size. If None, the training batch size is used.

        Returns
        -------
        ``Iterable``
            prediction
        """
        if isinstance(data, tf.data.Dataset):
            data = data.batch(self.config["training"]["batch_size"])
            batch_size = None
        elif batch_size is None:
            batch_size = self.config["training"]["batch_size"]

        pred = self.model.model.predict(data, batch_size=batch_size)
        if isinstance(pred, list):
            # multiple output model, but only care about first output
            pred = pred[0]
        return pred

    def evaluate_model(self, dataset: tf.data.Dataset | None = None) -> Any:
        """
        Evaluate model using :class:`tf.data.Dataset`.

        Parameters
        ----------
        dataset
            Dataset to evaluate.
            If None, :meth:`Estimator.val_dataset` is used.

        Returns
        -------
        ``Iterable[float]``
            Scores.
        """
        if not self.compiled_model:
            self._compile_model()
        if dataset is None:
            dataset = self.val_dataset
        self.model.model.reset_metrics()
        scores = self.model.model.evaluate(dataset.batch(self.config["training"]["batch_size"]))
        return scores
    

## PyTorch

class LossWarmupTorch:
    """Callback to warmup loss weights."""

    def __init__(self, weight_vars, to_weights, to_epochs):
        self.to_weights = to_weights
        self.to_epochs = to_epochs
        self.weight_vars = weight_vars

    def on_epoch_begin(self, epoch):
        """Update loss weights."""
        for key in self.to_epochs.keys():
            to_epoch = self.to_epochs[key]
            to_weight = self.to_weights[key]
            if to_epoch == 0 or to_epoch <= epoch:
                self.weight_vars[key].data = torch.tensor(to_weight, device=self.weight_vars[key].device)
            else:
                self.weight_vars[key].data = torch.tensor(to_weight / to_epoch * epoch, device=self.weight_vars[key].device)
            print(f"set {key} loss weight to {self.weight_vars[key].item()}")

        if "latent" in self.weight_vars.keys():
            print(f"set latent loss weight to {self.weight_vars['latent'].item()}")

class AnnealTemperatureTorch:
    """Callback to anneal learning rate."""

    def __init__(self, temperature, initial_temperature, final_temperature, to_epoch):
        self.temperature = temperature
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.to_epoch = to_epoch

    def on_epoch_begin(self, epoch):
        """Update temperature."""
        if self.to_epoch == 0 or self.to_epoch <= epoch:
            self.temperature.data = torch.tensor(self.final_temperature, device=self.temperature.device)
        else:
            new_temperature = self.initial_temperature + (self.final_temperature - self.initial_temperature) / self.to_epoch * epoch
            self.temperature.data = torch.tensor(new_temperature, device=self.temperature.device)
        print(f"set temperature to {self.temperature.item()}")
        

class TorchEstimator:
    def __init__(self, exp):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log = logging.getLogger(self.__class__.__name__)
        self.exp = exp
        self.config = exp.estimator_config

        self.config["training"]["loss"] = {
            key: LossEnumTorch(val).get_fn() for key, val in self.config["training"]["loss"].items()
        }
        self.config["training"]["metrics"] = {
            key: LossEnumTorch(val).get_fn() for key, val in self.config["training"]["metrics"].items()
        }
        self.callbacks: list[object] = []

        # create model
        self.criterion = self.config["training"]["loss"]
        self.optimizer = None
        self.epoch = 0
        self.create_model()
        self.compiled_model = False

        # train and val datasets
        # config params impacting y
        self.output_channels = self.config["data"]["output_channels"]
        self.repeat_y = len(self.config["training"]["loss"].keys())
        if self.repeat_y == 1:
            self.repeat_y = False
        self.add_c_to_y = False
        if "adv_head" in self.config["training"]["loss"].keys():
            self.add_c_to_y = True
            self.repeat_y = self.repeat_y - 1
        self.ds = NNTorchDataset(
            dataset_name=self.config["data"]["dataset_name"],
            data_config=self.config["data"]["data_config"],
        )
        self._train_dataset, self._val_dataset, self._test_dataset = None, None, None

        # set up model weights and history paths for saving/loading later
        self.weights_name = os.path.join(self.exp.full_path, "weights_epoch{:03d}")  # noqa: P103
        self.history_name = os.path.join(self.exp.full_path, "history.csv")
    
    @property
    def train_dataset(self) -> Dataset:
        """
        Shuffled :class:`tf.data.Dataset` of train split.
        """
        if self._train_dataset is None:
            self._train_dataset = self._get_dataset("train", shuffled=True)
        return self._train_dataset

    @property
    def val_dataset(self) -> Dataset:
        """
        :class:`tf.data.Dataset` of val split.
        """
        if self._val_dataset is None:
            self._val_dataset = self._get_dataset("val")
        return self._val_dataset

    @property
    def test_dataset(self) -> Dataset:
        """
        :class:`tf.data.Dataset` of test split.
        """
        if self._test_dataset is None:
            self._test_dataset = self._get_dataset("test")
        return self._test_dataset

    def _get_dataset(self, split: str, shuffled: bool = False) -> tf.data.Dataset:
        return self.ds.get_torch_dataset(
            split=split,
            output_channels=self.output_channels,
            is_conditional=self.model.is_conditional,
            repeat_y=self.repeat_y,
            add_c_to_y=self.add_c_to_y,
            shuffled=shuffled,
        )

    def latest_checkpoint(self, path):
        """
        Utility function for pytorch because tf.train.latest_checkpoint does not exist.
        Find the latest checkpoint in the given path.
        """
        checkpoints = [f for f in os.listdir(path) if f.endswith('.pt')]
        if not checkpoints:
            return None
        latest_checkpoint = max(checkpoints, key=lambda f: os.path.getctime(os.path.join(path, f)))
        return os.path.join(path, latest_checkpoint)
    
    def create_model(self):
        ModelClass = ModelEnumTorch(self.config["model"]["model_cls"]).get_cls()
        self.model = ModelClass(**self.config["model"]["model_kwargs"]).to(self.device)
        # To be done using maybe pytorchlightning 
        weights_path = self.config["model"]["init_with_weights"]
        if weights_path is True:
            weights_path = self.latest_checkpoint(self.exp.full_path)
            if weights_path is None:
                self.log.warning(
                    f"WARNING: weights_path set to true but no trained model found in {self.exp.full_path}"
                )
        if isinstance(weights_path, str):
            self.log.info(f"Initializing model with weights from {weights_path}")
            self.model.load_state_dict(torch.load(weights_path))
            self.epoch = self.exp.epoch

    def _compile_model(self):
        config = self.config["training"]
        self.loss_weights = {key: torch.tensor(val, device=self.device) for key, val in config["loss_weights"].items()}
        self.callbacks.append(
            LossWarmupTorch(
                self.loss_weights,
                config["loss_weights"],
                config["loss_warmup_to_epoch"],
            )
        )
        self.callbacks.append(UpdateSparsityLevelTorch(self.model))
        if hasattr(self.model, "temperature"):
            self.callbacks.append(
                AnnealTemperatureTorch(
                    self.model.temperature,
                    self.model.config["initial_temperature"],
                    self.model.config["temperature"],
                    self.model.config["anneal_epochs"],
                )
            )
        if self.optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=config["learning_rate"])
        self.compiled_model = True

    def train_model(self):
        config = self.config["training"]
        if not self.compiled_model:
            self._compile_model()
        if config["overwrite_history"]:
            self.epoch = 0
        self.log.info(f"Training model for {config['epochs']} epochs")
        self.model.train()
        train_loader = self.ds.get_torch_dataloader("train", batch_size=config["batch_size"], shuffled=True)
        val_loader = self.ds.get_torch_dataloader("val", batch_size=config["batch_size"], shuffled=False)
        history = []
        for epoch in range(config["epochs"]):
            epoch_loss = 0
            for batch in train_loader:
                self.optimizer.zero_grad()
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = sum(self.criterion[key](outputs[0], targets) * self.loss_weights[key] for key in self.criterion)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            val_loss = self.evaluate_model(val_loader)
            history.append({"epoch": epoch, "loss": epoch_loss, "val_loss": val_loss})
            self.log.info(f"Epoch {epoch}: loss = {epoch_loss}, val_loss = {val_loss}")
        self.epoch += config["epochs"]
        if config["save_model_weights"]:
            weights_name = self.weights_name.format(self.epoch)
            self.log.info(f"Saving model to {weights_name}")
            torch.save(self.model.state_dict(), weights_name)
        if config["save_history"]:
            history_df = pd.DataFrame(history)
            if os.path.exists(self.history_name) and not config["overwrite_history"]:
                prev_history = pd.read_csv(self.history_name, index_col=0)
                history_df = pd.concat([prev_history, history_df])
            history_df.to_csv(self.history_name)
        return history

    def predict_model(self, data, batch_size=None):
        self.model.eval()
        if isinstance(data, DataLoader):
            data_loader = data
        else:
            data_loader = DataLoader(data.transpose(0,3,1,2), batch_size=batch_size or self.config["training"]["batch_size"], shuffle=False)
        predictions = []
        with torch.no_grad():
            for batch in data_loader:
                inputs = batch.to(self.device)
                outputs = self.model(inputs)
                predictions.append(outputs[0].cpu().numpy())
        return np.concatenate(predictions, axis=0)

    def evaluate_model(self, dataset=None):
        self.model.eval()
        if dataset is None:
            dataset = self.val_dataset
            data_loader = self.ds.get_torch_dataloader("val", batch_size=self.config["training"]["batch_size"], shuffled=False)
        else:
            data_loader = dataset
        total_loss = 0
        with torch.no_grad():
            for batch in data_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = sum(self.criterion[key](outputs[0], targets) * self.loss_weights[key] for key in self.criterion)
                total_loss += loss.item()
        return total_loss / len(data_loader)