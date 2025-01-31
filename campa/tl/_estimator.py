from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from campa.tl import TorchExperiment
    import numpy as np

import os
import logging

import pandas as pd

from campa.tl import LossEnumTorch, ModelEnumTorch
from campa.data import NNTorchDataset
from campa.tl._layers import UpdateSparsityLevelTorch

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import re

import pytorch_warmup as warmup


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
        self.metrics = self.config["training"]["metrics"]
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

    def _get_dataset(self, split: str, shuffled: bool = False):
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
        # Regular expression to match the naming convention "weights_epoch<3 digit int>"
        pattern = re.compile(r'weights_epoch(\d{3})')

        max_epoch = -1
        latest_weights_path = None

        # Iterate over all files in the directory
        for filename in os.listdir(path):
            match = pattern.search(filename)
            if match:
                epoch = int(match.group(1))
                if epoch > max_epoch:
                    max_epoch = epoch
                    latest_weights_path = os.path.join(path, filename)

        return latest_weights_path
    
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
            print(f"Initializing model with weights from {weights_path}")
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

    def inference(self, batch):
        if self.model.is_conditional:
            inputs, targets = batch
            inputs, conditions = inputs
            inputs, conditions, targets = inputs.to(self.device), conditions.to(self.device), targets.to(self.device)
        else:
            inputs, targets = batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            conditions=None
        outputs = self.model(inputs, conditions)
        return outputs, targets

    def train_model(self):
        config = self.config["training"]
        if not self.compiled_model:
            self._compile_model()
        if config["overwrite_history"]:
            self.epoch = 0
        self.log.info(f"Training model for {config['epochs']} epochs")
        self.model.train()
        train_loader = self.ds.get_torch_dataloader("train", batch_size=config["batch_size"], shuffled=True, is_conditional=self.model.is_conditional)
        val_loader = self.ds.get_torch_dataloader("val", batch_size=config["batch_size"], shuffled=False, is_conditional=self.model.is_conditional)
        history_list = []

        # iters = len(train_loader)
        # num_steps = iters * config["epochs"]
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=num_steps)
        # warmup_scheduler = warmup.LinearWarmup(self.optimizer, warmup_period=1000)
        
        for epoch in tqdm(range(config["epochs"]), desc="Epochs"):
            epoch_loss = 0
            epoch_individual_loss = {key: 0 for key in self.criterion}
            epoch_individual_metrics = {key: 0 for key in self.metrics}
            n_batch = 0
            idx = 1
            for batch in tqdm(train_loader, desc="Batches", leave=False):
                self.optimizer.zero_grad()
                outputs, targets = self.inference(batch=batch)
                losses = {key: self.criterion[key](targets, outputs[key]) for key in self.criterion}
                metrics = {key: self.metrics[key](targets, outputs[key.split("_")[0]]).cpu().item() for key in self.metrics}
                loss = sum(losses[key] * self.loss_weights[key] for key in self.criterion)
                loss.backward()
                self.optimizer.step()

                # with warmup_scheduler.dampening():
                #     lr_scheduler.step(epoch + idx / iters)
                
                epoch_loss += loss.item()
                epoch_individual_loss = {key: epoch_individual_loss[key] + losses[key].item() for key in self.criterion}
                epoch_individual_metrics = {key: epoch_individual_metrics[key] + metrics[key] for key in self.metrics}
                n_batch += 1
                idx += 1
            epoch_loss = epoch_loss/n_batch
            epoch_individual_loss = {key+"_loss": value / n_batch for key, value in epoch_individual_loss.items()}
            epoch_individual_metrics = {key+"_metrics": value / n_batch for key, value in epoch_individual_metrics.items()}
            val_loss, val_losses, val_metrics = self.evaluate_model(val_loader)
            history = {"epoch": epoch, "loss": epoch_loss, "val_loss": val_loss}
            history.update(val_metrics)
            history.update(val_losses)
            history.update(epoch_individual_loss)
            history.update(epoch_individual_metrics)
            history_list.append(history)
            self.log.info(history)
            self.epoch += 1
        if config["save_model_weights"]:
            weights_name = self.weights_name.format(self.epoch)
            self.log.info(f"Saving model to {weights_name}")
            print(f"Saving model to {weights_name}")
            torch.save(self.model.state_dict(), weights_name)
        if config["save_history"]:
            history_df = pd.DataFrame(history_list)
            if os.path.exists(self.history_name) and not config["overwrite_history"]:
                prev_history = pd.read_csv(self.history_name, index_col=0)
                history_df = pd.concat([prev_history, history_df])
            history_df.to_csv(self.history_name)
        return history
    
    def predict_model(self, data) -> Any:
        """
        Predict all elements in ``data``.

        Parameters
        ----------
        data
            Data to predict, with first dimension the number of elements.

        Returns
        -------
        ``Iterable``
            prediction
        """
        if isinstance(data, NNTorchDataset):
            inputs, targets = next(iter(data))
            if self.model.is_conditional:
                x, c = inputs
            else:
                c = None
                x = inputs
            x = torch.tensor(x.transpose(0,3,1,2))
            c = torch.tensor(c)
        else:
            if self.model.is_conditional:
                x = torch.tensor(data[0].transpose(0,3,1,2), dtype=torch.float)
                c = torch.tensor(data[1], dtype=torch.float).to(self.device)
            else:
                x = torch.tensor(data.transpose(0,3,1,2), dtype=torch.float)
                c = None
        pred = self.model(x=x.to(self.device), c=c)["decoder"]
        # if isinstance(pred, list) or isinstance(pred, torch.Tensor):
        #     # multiple output model, but only care about first output
        #     pred = pred[0]
        return pred

    def evaluate_model(self, dataset=None):
        self.model.eval()
        if dataset is None:
            dataset = self.val_dataset
            data_loader = self.ds.get_torch_dataloader("val", batch_size=self.config["training"]["batch_size"], shuffled=False)
        else:
            data_loader = dataset
        total_loss = 0
        total_losses = {key: 0.0 for key in self.criterion} 
        total_metrics = {key: 0.0 for key in self.metrics} 
        with torch.no_grad():
            for batch in data_loader:
                outputs, targets = self.inference(batch=batch)
                losses = {key: self.criterion[key](targets, outputs[key]) for key in self.criterion}
                metrics = {key: self.metrics[key](targets, outputs[key.split("_")[0]]) for key in self.metrics}
                total_losses = {key: total_losses[key] + value.item() for key, value in losses.items()}
                total_metrics = {key: total_metrics[key] + value.item() for key, value in metrics.items()}
                loss = sum(losses[key] * self.loss_weights[key] for key in self.criterion)
                total_loss += loss.item()
            normalized_losses = {"val_"+key+"_loss": value / len(data_loader) for key, value in total_losses.items()}  # Average across batches
            normalized_metrics = {"val_"+key+"_metric": value / len(data_loader) for key, value in total_metrics.items()}  # Average across batches
        return total_loss / len(data_loader), normalized_losses, normalized_metrics