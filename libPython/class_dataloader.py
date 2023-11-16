#
# Thomas Haubner, LMS, 2020
#
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
import libPython.class_dataset as class_dataset


class DataLoaderModule(pl.LightningDataModule):
    """Dataloader class for AEC datasets."""

    def __init__(self, train_data_path, test_data_path, num_workers, batch_size, train_val_ratio,
                 shuffle_train_data, shuffle_val_data, pin_memory, frameshift):
        super().__init__()

        # settings
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.batch_size = batch_size
        assert 0 < train_val_ratio < 1
        self.train_val_ratio = train_val_ratio
        self.num_workers = num_workers
        self.shuffle_train_data = shuffle_train_data
        self.shuffle_val_data = shuffle_val_data
        self.pin_memory = pin_memory

        # block processing parameters
        self.frameshift = frameshift

        # data sets
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def setup(self, stage=None):
        """Setup to create datasets."""

        # crate training and validation datasets
        if stage == 'fit' or stage is None:
            # create training dataset
            train_dataset_full = class_dataset.AecDataset(self.train_data_path)

            num_samples_train = int(self.train_val_ratio * len(train_dataset_full))
            num_samples_val = len(train_dataset_full) - num_samples_train

            self.train_data, self.val_data = random_split(train_dataset_full, [num_samples_train, num_samples_val], generator=torch.Generator().manual_seed(41))    # seed to ensure equivalent validation datasets across different runs

        # create test dataset
        if stage == 'test' or stage is None:
            self.test_data = class_dataset.AecDataset(self.test_data_path)

    # initialize data loaders
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle_train_data, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle_val_data, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory)
