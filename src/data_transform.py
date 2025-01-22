import torch
from typing import Callable, Tuple


class MinMaxScaler:
    """A transform that rescales input data to the [0, 1] range.

    This transform shifts the data by its global minimum and divides
    by the range (max - min), resulting in data scaled to [0, 1].

    Args:
        min_value (float): The minimum value in the dataset.
        max_value (float): The maximum value in the dataset.
    """

    def __init__(self, min_value: float, max_value: float) -> None:
        self.min = min_value
        self.max = max_value

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the MinMax scaling to the input data.

        Args:
            data (torch.Tensor): The input tensor to be scaled.

        Returns:
            torch.Tensor: A tensor scaled to the [0, 1] range.
        """
        return (data - self.min) / (self.max - self.min)


class MinMaxNegScaler:
    """A transform that rescales input data to the [-1, 1] range.

    This transform first applies standard MinMax scaling to [0, 1],
    then linearly shifts the result to [-1, 1].

    Args:
        min_value (float): The minimum value in the dataset.
        max_value (float): The maximum value in the dataset.
    """

    def __init__(self, min_value: float, max_value: float) -> None:
        self.min = min_value
        self.max = max_value

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the MinMax scaling to [-1, 1] range.

        Args:
            data (torch.Tensor): The input tensor to be scaled.

        Returns:
            torch.Tensor: A tensor scaled to the [-1, 1] range.
        """
        # First scale to [0, 1]
        data = (data - self.min) / (self.max - self.min)
        # Shift the result to [-1, 1]
        return data * 2 - 1


class StdScaler:
    """A transform that standardizes input data by subtracting its mean and dividing by its standard deviation.

    Args:
        mean (float): The mean of the dataset.
        std (float): The standard deviation of the dataset.
    """

    def __init__(self, mean: float, std: float) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply standardization to the input data.

        Args:
            data (torch.Tensor): The input tensor to be standardized.

        Returns:
            torch.Tensor: A tensor standardized to have mean 0 and std 1.
        """
        return (data - self.mean) / self.std


class RandomFlip:
    """A transform that randomly flips the input tensor along specified dimensions.

    Args:
        dims (Tuple[int, ...], optional): Dimensions along which the tensor can be flipped.
            Defaults to (0,).
        p (float, optional): Probability of performing the flip. Defaults to 0.5.
    """

    def __init__(self, dims: Tuple[int, ...] = (0,), p: float = 0.5) -> None:
        self.dims = dims
        self.p = p

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Randomly flip the input tensor with probability `p`.

        Args:
            data (torch.Tensor): The input tensor to possibly flip.

        Returns:
            torch.Tensor: Either the flipped or the original tensor.
        """
        rand = torch.rand(1).item()
        if rand <= self.p:
            return torch.flip(data, dims=self.dims)
        else:
            return data


class RandomRot90:
    """A transform that rotates the input tensor by a multiple of 90 degrees.

    Args:
        dims (Tuple[int, int], optional): The dimensions along which to rotate.
            Defaults to (0, 1).
    """

    def __init__(self, dims: Tuple[int, int] = (0, 1)) -> None:
        self.dims = dims

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Rotate the input tensor by k*90 degrees, where k is randomly chosen from [0..3].

        Args:
            data (torch.Tensor): The input tensor to be rotated.

        Returns:
            torch.Tensor: The rotated tensor.
        """
        rand = torch.randint(0, 4, (1,)).item()
        return torch.rot90(data, k=rand, dims=self.dims)


class ReshapeTransform:
    """A transform that reshapes the input tensor to a specified shape.

    Args:
        new_shape (Tuple[int, ...]): The desired new shape for the tensor.
    """

    def __init__(self, new_shape: Tuple[int, ...]) -> None:
        self.new_shape = new_shape

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Reshape the input tensor.

        Args:
            data (torch.Tensor): The input tensor to be reshaped.

        Returns:
            torch.Tensor: A tensor reshaped to `new_shape`.
        """
        return torch.reshape(data, self.new_shape)


class ReduceDimension:
    """A transform that applies a specified reduction operation along a given dimension.

    Args:
        op (Callable): A function such as torch.mean, torch.sum, etc.
            The function signature must be of the form `op(tensor, dim=...)`.
        dim (int, optional): The dimension along which to reduce. Defaults to 0.
    """

    def __init__(self, op: Callable, dim: int = 0) -> None:
        self.op = op
        self.dim = dim

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Reduce the tensor along the specified dimension.

        Args:
            data (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The reduced tensor.
        """
        return self.op(data, dim=self.dim)


class LogTransform:
    """A transform that applies the natural logarithm elementwise to the input tensor."""

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the natural logarithm to each element of the input tensor.

        Args:
            data (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: A tensor with log applied elementwise.
        """
        return torch.log(data)