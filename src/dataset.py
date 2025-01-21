import h5py
import torch
import torchvision
import tqdm

import numpy as np
import torch.nn as nn

from pathlib import Path

from torch.utils.data import Dataset

from src.data_transform import MinMaxScaler, MinMaxNegScaler, StdScaler


def create_normalization(data, normalization):
    transform = nn.Identity()
    if normalization == 'MinMax':
        min_value = data.min().item()
        max_value = data.max().item()
        transform = MinMaxScaler(min_value, max_value)
    if normalization == 'MinMaxNeg':
        min_value = data.min().item()
        max_value = data.max().item()
        transform = MinMaxNegScaler(min_value, max_value)
    elif normalization == 'Std':
        mean = data.mean().item()
        std = data.std().item()
        transform = StdScaler(mean, std)
    return transform


class ParticleDataset(Dataset):
    """
    Dataset to hold particle event data
    :param signal_data (torch.Tensor): tensor of the signal data
    :param bg_data (torch.Tensor): tensor of the background data
    :param transform (list): list of transforms to apply
    """

    def __init__(self, signal_data, bg_data, transform=None):
        self.data = [bg_data, signal_data]
        self.transform = transform
        self.rng = np.random.default_rng(seed=42)

        self.indices = []
        for i, data_type in enumerate(self.data):
            self.indices += [(i, j) for j in range(len(data_type))]
        self.indices = np.array(self.indices)
        self.rng.shuffle(self.indices)

    def __getitem__(self, index):
        """
        Returns the (input, label) pair at an index
        :param index (int): index of the data to retrieve
        :return (Tuple[torch.Tensor]): (input, label) pair
        """
        label, data_index = self.indices[index]
        data = self.data[label][data_index]

        if self.transform:
            data = self.transform(data)

        return data, torch.tensor([label])

    def __len__(self):
        return len(self.indices)


class DatasetGenerator:
    def __init__(
            self,
            name,
            file_path,
            signal_groups,
            bg_groups,
            size,
            signal_ratio=0.5,
            splits=(0.8, 0.2),
            transforms=None,
            normalization=False,
            regenerate=False):

        self.file_path = Path(file_path)
        self.f = h5py.File(file_path, 'r')
        self.rng = np.random.default_rng(seed=42)
        self.signal_groups = signal_groups
        self.bg_groups = bg_groups
        self.size = size
        self.signal_ratio = signal_ratio
        self.splits = splits
        self.transforms = transforms[:] if transforms is not None else []
        self.normalization = normalization
        self.regenerate = regenerate

        self.signal_data_path = Path(self.file_path.parent / f'{name}_signal.pt')
        self.signal_data = torch.empty(0)
        self.bg_data_path = Path(self.file_path.parent / f'{name}_bg.pt')
        self.bg_data = torch.empty(0)
        self.datasets = []

    #@src.utils.timeit
    def generate(self):
        # get signal data or load from file
        self.signal_data = self._get_data(self.signal_data_path, self.signal_groups, self.signal_ratio)

        # get background data or load from file
        self.bg_data = self._get_data(self.bg_data_path, self.bg_groups, 1 - self.signal_ratio)

        #if (len(self.signal_data) + len(self.bg_data)) != self.size:
        #    raise ValueError("Specified size doesn't match size of data"
        #                     "read from disk. Please regenerate data.")

        # generate splits (train/validate etc.)
        from_signal = 0
        from_bg = 0
        current_split = 0
        normalization_transform = None
        for split in self.splits:
            current_split += split
            to_signal = int(len(self.signal_data) * current_split)
            to_bg = int(len(self.bg_data) * current_split)

            signal_split = self.signal_data[from_signal:to_signal]
            bg_split = self.bg_data[from_bg:to_bg]

            # assuming the first set is the training set: set up normalization
            if self.normalization and not normalization_transform:
                normalization_transform = create_normalization(
                    signal_split, self.normalization)
                self.transforms.append(normalization_transform)

            dataset = ParticleDataset(
                signal_split,
                bg_split,
                torchvision.transforms.Compose(self.transforms)
            )

            self.datasets.append(dataset)
            from_signal = to_signal
            from_bg = to_bg

        return self.datasets

    def _get_data(self, data_path, source_groups, ratio):
        # read existing data from disk
        if data_path.exists() and not self.regenerate:
            return torch.load(data_path, weights_only=False)

        # set up continuous array of indices
        indices = []
        groups = []
        data_size = 0

        if isinstance(source_groups, list):
            for i, group in enumerate(source_groups):
                indices += [(i, j) for j in range(len(self.f[f'{group}/jet']))]

            # shuffle for even distribution of groups
            indices = np.array(indices)
            self.rng.shuffle(indices)

            # calculate size of current data and select as many data examples
            data_size = int(self.size * ratio)
            indices = indices[:data_size]
            groups = source_groups

        elif isinstance(source_groups, dict):
            for i, (group, group_size) in enumerate(source_groups.items()):
                group_indices = np.array([(i, j) for j in range(len(self.f[f'{group}/jet']))])
                self.rng.shuffle(group_indices)
                indices += group_indices[:group_size].tolist()
            groups = list(source_groups.keys())
            data_size = len(indices)

        # read data from h5py file
        data = torch.empty((data_size, 7500), dtype=torch.float32)
        with tqdm.tqdm(total=data_size, miniters=data_size/1000) as progress:
            progress.set_description(f'Loading data')
            for i, j in enumerate(indices):
                data[i] = torch.from_numpy(self.f[f'{groups[j[0]]}/jet'][j[1]])
                progress.update()

        # store in file
        torch.save(data, data_path)

        return data
