#
# Thomas Haubner, LMS, 2022
#
import torch
from torch.utils.data import Dataset
import numpy as np
import h5py


class AecDataset(torch.utils.data.Dataset):
    """AEC dataset class."""

    def __init__(self, filename):

        # load hdf5 file
        with h5py.File(filename, 'r') as hf_file:
            hf_file = h5py.File(filename, 'r')

            # load sampling frequency
            self.fs = np.array(hf_file.get('fs'))

            # convert hdf5 to numpy
            self.u_td_tensor = np.array(hf_file.get('u_td_tensor'))                        # loudspeaker tensor
            self.y_td_tensor = np.array(hf_file.get('y_td_tensor'))                        # microphone tensor
            self.d_td_tensor = np.array(hf_file.get('d_td_tensor'))                        # ground-truth echo tensor
            assert (self.u_td_tensor.shape == self.y_td_tensor.shape == self.d_td_tensor.shape)

        # convert to torch tensor
        self.u_td_tensor = torch.from_numpy(self.u_td_tensor).float()
        self.y_td_tensor = torch.from_numpy(self.y_td_tensor).float()
        self.d_td_tensor = torch.from_numpy(self.d_td_tensor).float()

        # number of files
        self.num_files = self.u_td_tensor.shape[0]

    def __getitem__(self, sample_ind):
        """Get one sequence of dataset."""

        u_td_sample = self.u_td_tensor[sample_ind, :]
        y_td_sample = self.y_td_tensor[sample_ind, :]
        d_td_sample = self.d_td_tensor[sample_ind, :]

        return u_td_sample, y_td_sample, d_td_sample

    def __len__(self):
        """Number of sequences in dataset."""

        return self.num_files
