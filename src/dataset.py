import h5py
import torch
import torchvision
import tqdm

import numpy as np
import torch.nn as nn

from pathlib import Path
from torch.utils.data import Dataset

from src.data_transform import MinMaxScaler, MinMaxNegScaler, StdScaler


def create_normalization(data: torch.Tensor, normalization: str) -> nn.Module:
    """Creates a normalization transform based on the provided data and normalization type.

    Args:
        data (torch.Tensor): Input data to be normalized.
        normalization (str): The type of normalization to apply. Supported values:
            - 'MinMax'
            - 'MinMaxNeg'
            - 'Std'

    Returns:
        nn.Module: The transformation object for normalizing the data.
    """
    transform = nn.Identity()
    if normalization == 'MinMax':
        min_value = data.min().item()  # Get the minimum value of the dataset
        max_value = data.max().item()  # Get the maximum value of the dataset
        transform = MinMaxScaler(min_value, max_value)
    elif normalization == 'MinMaxNeg':
        min_value = data.min().item()  # Get the minimum value of the dataset
        max_value = data.max().item()  # Get the maximum value of the dataset
        transform = MinMaxNegScaler(min_value, max_value)
    elif normalization == 'Std':
        mean = data.mean().item()  # Compute mean of the data
        std = data.std().item()    # Compute standard deviation of the data
        transform = StdScaler(mean, std)

    return transform


class ParticleDataset(Dataset):
    """Dataset to hold particle event data.

    This dataset contains both signal and background data, and provides
    transforms (if any) for each returned sample.

    Args:
        signal_data (torch.Tensor): Tensor of the signal data.
        bg_data (torch.Tensor): Tensor of the background data.
        transform (torch.nn.Module, optional): Transform or composition of transforms
            to apply to each sample. Defaults to None.
    """

    def __init__(
        self,
        signal_data: torch.Tensor,
        bg_data: torch.Tensor,
        transform: nn.Module = None
    ) -> None:
        # Store signal and background data in a list for unified access
        self.data = [bg_data, signal_data]
        # The transformation pipeline to apply to each sample
        self.transform = transform
        # Random number generator for shuffling
        self.rng = np.random.default_rng(seed=42)

        # Prepare a list of (type_index, sample_index) to keep track
        # of which data sample belongs to which class
        self.indices = []
        for i, data_type in enumerate(self.data):
            self.indices += [(i, j) for j in range(len(data_type))]
        self.indices = np.array(self.indices)

        # Shuffle the combined indices for training/randomization
        self.rng.shuffle(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the (input, label) pair at a given index.

        Args:
            index (int): Index of the data to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - The data tensor (possibly transformed)
                - A tensor containing the label
        """
        label, data_index = self.indices[index]
        data = self.data[label][data_index]

        # Apply transform if provided
        if self.transform:
            data = self.transform(data)

        # Convert label to a tensor
        return data, torch.tensor([label])

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset.

        Returns:
            int: Total number of samples.
        """
        return len(self.indices)


class DatasetGenerator:
    """Generates datasets for training, validation, and testing based on particle event data.

    This class reads data from an HDF5 file, splits it into the desired ratios,
    optionally normalizes it, and then produces `ParticleDataset` instances for
    further usage (e.g., training, validation, testing).

    Args:
        name (str): Name of the dataset.
        file_path (str): Path to the HDF5 file containing the data.
        signal_groups (list | dict): Groups of signal data in the HDF5 file.
        bg_groups (list | dict): Groups of background data in the HDF5 file.
        size (int): Total size of the dataset to generate.
        signal_ratio (float, optional): Proportion of signal data in the dataset. Defaults to 0.5.
        splits (tuple, optional): Splits for partitioning the dataset. Defaults to (0.8, 0.2).
        transforms (list, optional): List of transformations (callables) to apply. Defaults to None.
        normalization (bool | str, optional): Whether or which type of normalization to apply.
            Possible values: False, 'MinMax', 'MinMaxNeg', 'Std'. Defaults to False.
        regenerate (bool, optional): Whether to regenerate the data even if it exists on disk.
            Defaults to False.
    """

    def __init__(
        self,
        name: str,
        file_path: str,
        signal_groups: list | dict,
        bg_groups: list | dict,
        size: int,
        signal_ratio: float = 0.5,
        splits: tuple = (0.8, 0.2),
        transforms: list = None,
        normalization: bool | str = False,
        regenerate: bool = False
    ) -> None:
        # Convert file path to a Path object for easier manipulation
        self.file_path = Path(file_path)
        # Open the HDF5 file in read-only mode
        self.f = h5py.File(file_path, 'r')
        # Random number generator for consistent shuffling
        self.rng = np.random.default_rng(seed=42)

        # User-defined groups and other parameters
        self.signal_groups = signal_groups
        self.bg_groups = bg_groups
        self.size = size
        self.signal_ratio = signal_ratio
        self.splits = splits
        self.transforms = transforms[:] if transforms is not None else []
        self.normalization = normalization
        self.regenerate = regenerate

        # Paths to cached torch data for signal/background
        self.signal_data_path = self.file_path.parent / f'{name}_signal.pt'
        self.bg_data_path = self.file_path.parent / f'{name}_bg.pt'

        # Initialize empty Tensors for signal and background
        self.signal_data = torch.empty(0)
        self.bg_data = torch.empty(0)

        # List to hold ParticleDataset instances after generation
        self.datasets = []

    def generate(self) -> list[ParticleDataset]:
        """Generates the datasets by splitting the signal and background data
        according to the specified ratios in `self.splits`.

        Returns:
            list[ParticleDataset]: A list of ParticleDataset instances, each
            representing a data split (e.g., train/validation/test).
        """
        # Retrieve or load signal/background data
        self.signal_data = self._get_data(self.signal_data_path, self.signal_groups, self.signal_ratio)
        self.bg_data = self._get_data(self.bg_data_path, self.bg_groups, 1 - self.signal_ratio)

        # Track slice indices for splitting
        from_signal = 0
        from_bg = 0
        current_split = 0

        # If normalization is requested, we will create the transform
        # during the first split creation
        normalization_transform = None

        for split in self.splits:
            current_split += split
            # Calculate up-to indices for slicing each segment
            to_signal = int(len(self.signal_data) * current_split)
            to_bg = int(len(self.bg_data) * current_split)

            # Slice the data for this split
            signal_split = self.signal_data[from_signal:to_signal]
            bg_split = self.bg_data[from_bg:to_bg]

            # If normalization is requested and not yet created,
            # create a normalization transform based on signal data
            if self.normalization and not normalization_transform:
                normalization_transform = create_normalization(signal_split, self.normalization)
                self.transforms.append(normalization_transform)

            # Create the transform pipeline
            transform_pipeline = torchvision.transforms.Compose(self.transforms)

            # Instantiate the dataset for this particular split
            dataset = ParticleDataset(signal_split, bg_split, transform_pipeline)
            self.datasets.append(dataset)

            # Update the slicing indices for the next split
            from_signal = to_signal
            from_bg = to_bg

        return self.datasets

    def _get_data(
        self,
        data_path: Path,
        source_groups: list | dict,
        ratio: float
    ) -> torch.Tensor:
        """Retrieves data from the HDF5 file or loads it from disk if it already exists.

        Args:
            data_path (Path): Path to the stored (cached) data file.
            source_groups (list | dict): Groups of data to read from the HDF5 file.
            ratio (float): Proportion of total `self.size` that this data should occupy.

        Returns:
            torch.Tensor: Loaded data tensor.
        """
        # If a cached file exists and we do not want to regenerate, load from disk
        if data_path.exists() and not self.regenerate:
            return torch.load(data_path, weights_only=False)

        indices = []
        groups = []
        data_size = 0

        # If groups is a list, randomly select 'ratio * size' samples from the available data
        if isinstance(source_groups, list):
            for i, group in enumerate(source_groups):
                # Combine indices from each group
                num_samples_in_group = len(self.f[f'{group}/jet'])
                indices += [(i, j) for j in range(num_samples_in_group)]

            indices = np.array(indices)
            self.rng.shuffle(indices)

            data_size = int(self.size * ratio)
            # Take only the portion of indices needed for the requested ratio
            indices = indices[:data_size]
            groups = source_groups

        # If groups is a dict, we assume each key-value pair explicitly defines the number of samples
        elif isinstance(source_groups, dict):
            for i, (group, group_size) in enumerate(source_groups.items()):
                # For each group, shuffle the indices and select the specified group_size
                group_indices = np.array([(i, j) for j in range(len(self.f[f'{group}/jet']))])
                self.rng.shuffle(group_indices)
                indices += group_indices[:group_size].tolist()
            groups = list(source_groups.keys())
            data_size = len(indices)

        # Create an empty tensor to hold the data
        data = torch.empty((data_size, 7500), dtype=torch.float32)

        # Use a tqdm progress bar to monitor data loading
        with tqdm.tqdm(total=data_size, miniters=data_size/1000) as progress:
            progress.set_description('Loading data')
            for idx, (group_idx, sample_idx) in enumerate(indices):
                # Read data from the HDF5 file and place it into the pre-allocated tensor
                data[idx] = torch.from_numpy(self.f[f'{groups[group_idx]}/jet'][sample_idx])
                progress.update()

        # Save the data for future use
        torch.save(data, data_path)

        return data