import math
import numpy as np
import torch
from torch import nn


class FC(nn.Module):
    """A fully-connected feedforward neural network with a single output neuron.

    This model is intended for binary classification tasks. It supports different
    ways of defining hidden-layer widths (constant, logarithmic, or linear spacing),
    as well as optional dropout and batch normalization layers.

    Args:
        in_features (int, optional): Number of input features. Defaults to 7500.
        width (int | str, optional): Width of the hidden layers. This can be:
            - An integer (e.g., 32).
            - 'log': Logarithmically scale from `in_features` down to 1 over `depth` steps.
            - 'lin': Linearly scale from `in_features` down to 1 over `depth` steps.
          Defaults to 32.
        depth (int, optional): Number of layers, including the output layer. Defaults to 6.
        variances (tuple[float, float] | None, optional): A tuple containing
            the variances for weights and biases respectively. If provided,
            the weight variance is further scaled by the number of input features
            in each layer. Defaults to None.
        activation (nn.Module, optional): The activation function to use in the
            hidden layers (e.g., nn.ReLU). Defaults to `nn.ReLU`.
        dropout (float, optional): Dropout probability. If non-zero, dropout
            layers are added after each linear layer (except the last). Defaults to 0.
        batch_norm (bool, optional): If True, batch normalization is applied
            after each linear layer (except the last). Defaults to False.
    """

    def __init__(
        self,
        in_features: int = 7500,
        width: int | str = 32,
        depth: int = 6,
        variances: tuple[float, float] | None = None,
        activation: nn.Module = nn.ReLU,
        dropout: float = 0.0,
        batch_norm: bool = False
    ) -> None:
        super().__init__()
        self.layers = []

        # Determine layer widths based on `width` parameter
        if width == 'log':
            # Logarithmically spaced widths from in_features down to 1
            layer_widths = list(map(
                int,
                torch.round(torch.logspace(
                    math.log(in_features, 10), 0, depth + 1
                ))
            ))
        elif width == 'lin':
            # Linearly spaced widths from in_features down to 1
            layer_widths = list(map(
                int,
                torch.round(torch.linspace(in_features, 1, depth + 1))
            ))
        else:
            # Fixed or user-specified integer width for hidden layers
            layer_widths = [in_features] + (depth - 1) * [width] + [1]

        # Create each linear layer with optional initialization
        for i in range(depth):
            layer = nn.Linear(layer_widths[i], layer_widths[i + 1])

            # If variances are specified, initialize using a normal distribution
            if variances:
                nn.init.normal_(
                    layer.weight,
                    mean=0.0,
                    std=np.sqrt(variances[0] / layer.in_features)
                )
                nn.init.normal_(
                    layer.bias,
                    mean=0.0,
                    std=np.sqrt(variances[1])
                )

            self.layers.append(layer)

            # For all but the last layer, optionally add dropout, batch norm, and activation
            if i < (depth - 1):
                if dropout:
                    self.layers.append(nn.Dropout(p=dropout))
                if batch_norm:
                    self.layers.append(nn.BatchNorm1d(layer.out_features))

                self.layers.append(activation())

        # Wrap all layers into a Sequential module
        self.fc = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_features].

        Returns:
            torch.Tensor: The output tensor of shape [batch_size, 1].
        """
        # Pass input through the sequential model
        return self.fc(x)