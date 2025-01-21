import math
import numpy as np
import torch

from torch import nn


class FC(nn.Module):
    """
    A fully-connected feed forward neural network with a single output neuron for
    binary classification.

    :param in_features (int): Number of input features
    :param width (int or str): Width of the hidden layers; can be an int or 'log' or 'lin'.
        When 'log' or 'lin' the widths of the layers are gradually scaled to the final layer size of 1.
    :param depth: Number of hidden layers including the output layer
    :param variances (float, float): Tuple of the variances for weights and biases;
        the variance for the weights is scaled by the number of input features of each layer
    :param activation (nn.Module): Activation function to be used, e.g. torch.nn.ReLU, torch.nn.Tanh
    :param dropout (float): If non-zero, drop-out will be applied after each layer
    :param batch_norm (bool): If True, bach normalization will be applied after each layer
    """
    def __init__(self, in_features=7500, width=32, depth=6, variances=None,
                 activation=nn.ReLU, dropout=0, batch_norm=False):
        super().__init__()
        self.layers = []

        # generate a list with all layer widths
        if width == 'log':
            layer_widths = list(map(int, torch.round(torch.logspace(
                math.log(in_features, 10), 0, depth + 1))))
        elif width == 'lin':
            layer_widths = list(map(int, torch.round(torch.linspace(
                in_features, 1, depth + 1))))
        else:
            layer_widths = [in_features] + (depth - 1) * [width] + [1]

        # generate the linear layers according to the specified widths and
        # initialize the weights with a zero-mean Gaussian and the specified variance
        for i in range(depth):
            layer = nn.Linear(layer_widths[i], layer_widths[i + 1])
            if variances:
                torch.nn.init.normal_(layer.weight, mean=0.0,
                    std=(np.sqrt(variances[0] / layer.in_features)))
                torch.nn.init.normal_(layer.bias, mean=0.0,
                    std=(np.sqrt(variances[1])))
            self.layers.append(layer)

            # apply drop-out and bach normalization if required
            if i < (depth - 1):
                if dropout:
                    self.layers.append(nn.Dropout(p=dropout))
                if batch_norm:
                    self.layers.append(nn.BatchNorm1d(layer.out_features))

                self.layers.append(activation())

        # create the sequential model
        self.fc = nn.Sequential(*self.layers)

    def forward(self, x):
        """
        Forward pass through the network
        :param x (torch.Tensor): Input tensor
        :return (torch.Tensor): Output tensor
        """
        x = self.fc(x)
        return x
