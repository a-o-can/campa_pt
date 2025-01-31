from typing import Any, Dict, List, Tuple, Union, Mapping, Iterable, Optional
import os
import json
import logging

import numpy as np

from campa.constants import campa_config
from campa.data._data import MPPData

import torch
from torch.utils.data import Dataset, DataLoader


def create_dataset(params: Mapping[str, Any]) -> None:
    """
    Create a :class:`NNDataset`.

    Parameters determine how the data should be selected and processed.
    The following keys in parameters are expected:

    - ``dataset_name``: name of the resulting dataset that is defined by these parameters
      (relative to ``DATA_DIR/datasets``)
    - ``data_config``: name of data configuration (registered in ``campa.ini``)
    - ``data_dirs``: where to read data from (relative to ``DATA_DIR`` defined in data config)
    - ``channels``: list of channel names to include in this dataset
    - ``condition``: list of conditions. Should be defined in data config.
      The suffix `_one_hot` will convert the condition in a one-hot encoded vector.
      Conditions are concatenated, except when they are defined as a list of lists.
      In this case the condition is defined as a pairwise combination of the conditions.
    - ``condition_kwargs``: kwargs to :meth:`MPPData.add_conditions`
    - ``split_kwargs``: kwargs to :meth:`MPPData.train_val_test_split`
    - ``test_img_size``: standard size of images in test set. Imaged are padded/truncated to this size
    - ``subset``: (bool) subset to objects with certain metadata.
    - ``subset_kwargs``: kwargs to :meth:`MPPData.subset` defining which object to subset to
    - ``subsample``: (bool) subsampling of pixels (only for train/val)
    - ``subsample_kwargs``: kwargs for :meth:`MPPData.subsample` defining the fraction of pixels to be sampled
    - ``neighborhood``: (bool) add local neighbourhood to samples in NNDataset
    - ``neighborhood_size``: size of neighbourhood
    - ``normalise``: (bool) Intensity normalisation
    - ``normalise_kwargs``: kwargs to :meth:`MPPData.normalise`
    - ``seed``: random seed to make subsampling reproducible

    Parameters
    ----------
    params
        parameter dict
    """
    log = logging.getLogger()
    log.info("Creating train/val/test datasets with params:")
    log.info(json.dumps(params, indent=4))
    p = params
    # prepare outdir
    data_config = campa_config.get_data_config(p["data_config"])
    outdir = os.path.join(data_config.DATASET_DIR, p["dataset_name"])
    os.makedirs(outdir, exist_ok=True)
    # prepare datasets
    mpp_datas: Mapping[str, List[MPPData]] = {"train": [], "val": [], "test": []}
    for data_dir in p["data_dirs"]:
        mpp_data = MPPData.from_data_dir(data_dir, seed=p["seed"], data_config=p["data_config"])
        train, val, test = mpp_data.train_val_test_split(**p["split_kwargs"])
        # subsample train data now
        if p["subsample"]:
            train = train.subsample(
                add_neighborhood=p["neighborhood"],
                neighborhood_size=p["neighborhood_size"],
                **p["subsample_kwargs"],
            )
        elif p["neighborhood"]:
            train.add_neighborhood(p["neighborhood_size"])
        mpp_datas["train"].append(train)
        mpp_datas["test"].append(test)
        mpp_datas["val"].append(val)
    # merge all datasets
    train = MPPData.concat(mpp_datas["train"])
    val = MPPData.concat(mpp_datas["val"])
    test = MPPData.concat(mpp_datas["test"])
    # prepare (channels, normalise, condition, subset)
    train.prepare(params)  # this has side-effects on params, st val + test use correct params
    val.prepare(params)
    test.prepare(params)
    # save test and val imgs
    val.write(os.path.join(outdir, "val_imgs"))
    test.write(os.path.join(outdir, "test_imgs"))
    # subsample and add neighbors to val and test (for prediction during training)
    if p["subsample"]:
        val = val.subsample(
            add_neighborhood=p["neighborhood"],
            neighborhood_size=p["neighborhood_size"],
            **p["subsample_kwargs"],
        )
        test = test.subsample(
            add_neighborhood=p["neighborhood"],
            neighborhood_size=p["neighborhood_size"],
            **p["subsample_kwargs"],
        )
    elif p["neighborhood"]:
        val.add_neighborhood(p["neighborhood_size"])
        test.add_neighborhood(p["neighborhood_size"])

    log.info("-------------------")
    log.info("created datasets:")
    log.info(f"train: {str(train)}")
    log.info(f"val: {str(val)}")
    log.info(f"test: {str(test)}")
    log.info("-------------------")

    # save datasets
    train.write(os.path.join(outdir, "train"))
    val.write(os.path.join(outdir, "val"))
    test.write(os.path.join(outdir, "test"))
    # save params
    json.dump(params, open(os.path.join(outdir, "params.json"), "w"), indent=4)


class NNTorchDataset(Dataset):
    """
    Dataset for training and evaluation of neural networks with PyTorch.

    A ``NNTorchDataset`` is stored within ``DATA_DIR/dataset_name``.
    This folder contains `train`/`val`/`test`/`val_img`/`test_img` folders with :class:`MPPData` objects.

    Parameters
    ----------
    dataset_name:
        name of the dataset, relative to ``DATA_DIR``
    data_config:
        name of the data config to use, should be registered in ``campa.ini``
    """
    def __init__(self, dataset_name: str, data_config: Optional[str] = None):
        super().__init__()
        self.log = logging.getLogger(self.__class__.__name__)
        if data_config is None:
            self.data_config_name = "NascentRNA"
            self.log.warning(f"Using default data_config {self.data_config_name}")
        else:
            self.data_config_name = data_config
        self.data_config = campa_config.get_data_config(self.data_config_name)
        self.dataset_folder = os.path.join(self.data_config.DATASET_DIR, dataset_name)

        # data
        self.data: Dict[str, MPPData] = {
            "train": MPPData.from_data_dir(
                os.path.join(self.dataset_folder, "train"), base_dir="", data_config=self.data_config_name
            ),
            "val": MPPData.from_data_dir(
                os.path.join(self.dataset_folder, "val"), base_dir="", data_config=self.data_config_name
            ),
            "test": MPPData.from_data_dir(
                os.path.join(self.dataset_folder, "test"), base_dir="", data_config=self.data_config_name
            ),
        }
        """
        Train, val, and test MPPDatas.
        """
        self.imgs: Dict[str, MPPData] = {
            "val": MPPData.from_data_dir(
                os.path.join(self.dataset_folder, "val_imgs"), base_dir="", data_config=self.data_config_name
            ),
            "test": MPPData.from_data_dir(
                os.path.join(self.dataset_folder, "test_imgs"), base_dir="", data_config=self.data_config_name
            ),
        }
        """
        Val and test MPPDatas containing entire images for visualisation.
        """
        self.channels = self.data["train"].channels.reset_index().set_index("name")
        self.params = json.load(open(os.path.join(self.dataset_folder, "params.json")))

    def __str__(self):
        s = f"NNDataset for {self.data_config_name} (shape {self.data['train'].mpp.shape[1:]})."
        s += f" train: {len(self.data['train'].mpp)}, val: {len(self.data['val'].mpp)},"
        s += f" test: {len(self.data['test'].mpp)}"
        return s

    def __len__(self):
        return len(self.data["train"].mpp)

    def __getitem__(self, idx):
        x = self.x("train")[idx]
        y = self.y("train")[idx]
        return x, y
    
    def x(self, split: str, is_conditional: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Neural network inputs.

        Parameters
        ----------
        split
            One of `train`, `val`, `test`.
        is_conditional
            Whether to add condition information to x
        """
        x = self.data[split].mpp.astype(np.float32)
        if is_conditional:
            c = self.data[split].conditions.astype(np.float32)  # type: ignore[union-attr]
            x = (x, c)  # type: ignore[assignment]
        if isinstance(x, np.memmap):
            x = x.transpose(0,3,1,2) # pytorch expects the channel to be the second dimension
        else: # it is a tuple with x being the 0th and c being the 1st element
            x = (x[0].transpose(0,3,1,2), x[1]) # recreate because tuples are immutable
        return x

    def y(self, split: str, output_channels: Optional[Iterable[str]] = None) -> np.ndarray:
        """
        Groundtruth outputs.

        Parameters
        ----------
        split
            One of `train`, `val`, `test`.
        output_channels
            Channels that should be predicted by the neural network.
            Defaults to all input channels.
        """
        y = self.data[split].center_mpp
        if output_channels is not None:
            channel_ids = self.data[split].get_channel_ids(list(output_channels))
            y = y[:, channel_ids]
        return y

    def get_torch_dataset(
        self,
        split: str = "train",
        output_channels: Optional[Iterable[str]] = None,
        is_conditional: bool = False,
        repeat_y: Union[bool, int] = False,
        add_c_to_y: bool = False,
    ) -> Dataset:
        """
        PyTorch Dataset of the desired split.

        Parameters
        ----------
        split
            One of `train`, `val`, `test`.
        output_channels
            Channels that should be predicted by the neural network.
            Defaults to all input channels.
        is_conditional
            Whether to add condition information to x
        repeat_y:
            Match output length to number of losses
            (otherwise PyTorch will not work, even if its losses that do not need y).
        add_c_to_y
            Append condition to y. Needed for adversarial loss.
        shuffled
            Shuffle indices before generating data.
            Will produce same order every time.
        batch_size
            Number of samples per batch.

        Returns
        -------
        DataLoader
            The dataset loader.
        """
        class CustomDataset(Dataset):
            def __init__(self, x, y, is_conditional, repeat_y, add_c_to_y):
                self.x = x
                self.y = y
                self.is_conditional = is_conditional
                self.repeat_y = repeat_y
                self.add_c_to_y = add_c_to_y

            def __len__(self):
                return len(self.x[0]) if self.is_conditional else len(self.x)

            def __getitem__(self, idx):
                if self.is_conditional:
                    el_x = (self.x[0][idx], self.x[1][idx])
                else:
                    el_x = self.x[idx]

                if self.repeat_y is not False:
                    el_y = tuple(self.y[j][idx] for j in range(len(self.y)))
                else:
                    el_y = self.y[idx]

                if self.add_c_to_y:
                    el_y = (el_y, self.x[1][idx])

                return el_x, el_y

        x = self.x(split, is_conditional)
        y = self.y(split, output_channels)

        if repeat_y is not False:
            y = tuple(y for _ in range(repeat_y))

        dataset = CustomDataset(x, y, is_conditional, repeat_y, add_c_to_y)
        return dataset
        # return DataLoader(dataset, shuffle=shuffled, batch_size=batch_size)

    def get_torch_dataloader(self,
        split: str = "train",
        output_channels: Optional[Iterable[str]] = None,
        is_conditional: bool = False,
        repeat_y: Union[bool, int] = False,
        add_c_to_y: bool = False,
        shuffled: bool = False,
        batch_size: int = 32,
        ) -> DataLoader:
        return DataLoader(self.get_torch_dataset(split, output_channels, is_conditional, repeat_y, add_c_to_y), shuffle=shuffled, batch_size=batch_size)