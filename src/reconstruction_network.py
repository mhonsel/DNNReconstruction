import time

import numpy as np
import torch
import torch.nn as nn
import tqdm

from typing import List, Union


class InverseSequential(nn.Module):
    """
    Reconstruction layer class
    For each layer of the DNN an InverseSequential is instantiated automatically
    within the ReconstructionNetwork class

    :param forward_layer (nn.Module): Layer of the DNN to be reconstructed
    :param forward_activation (nn.Module): Activation of the layer of the DNN
    :param depth (int): Number of reconstruction layers
    :param lr (float): Learning rate of the reconstruction
    """
    def __init__(self, forward_layer: nn.Module, forward_activation: nn.Module,
                 depth: int = 1, lr: float = 1e-3):
        super().__init__()

        # for the forward pass through the layer of the DNN
        self.forward_stack = nn.Sequential(
            forward_layer,
            forward_activation
        )

        # linear layers trained to reconstruct the forward pass
        features = ([forward_layer.out_features] * depth + [forward_layer.in_features])
        inverse_layers = []
        for i in range(len(features) - 1):
            inverse_layers.append(nn.Linear(features[i], features[i + 1]))
            inverse_layers.append(nn.Tanh())

        self.inverse_stack = nn.Sequential(*inverse_layers)
        self.lr = lr
        self.optimizer = None
        self.loss_fn = nn.MSELoss()

    def register_optimizer(self):
        """ Helper method to add the optimizer after initializing the network """
        self.optimizer = torch.optim.Adam(self.inverse_stack.parameters(), lr=self.lr)

    def forward(self, x):
        """
        Forward pass through the reconstruction layer
        :param x (torch.Tensor): Input tensor
        :return (torch.Tensor): Output tensor, e.g. the reconstruction
        """
        return self.inverse_stack(x)

    def get_forward(self, x):
        """
        Forward pass through the layer to be reconstructed
        :param x (torch.Tensor): Input tensor
        :return (torch.Tensor): Output tensor
        """
        with torch.no_grad():
            x_input = self.forward_stack(x)
            x_input = x_input.detach()
            return x_input


class ReconstructionNetwork:
    """
    Network for reconstruction of a DNN
    This class holds all reconstruction layers to produce the cascades

    :param net (nn.Module): DNN to create reconstruction layers for
    :param device (torch.device or str): Union[str, torch.device]: Device to perform calculations on
    :param lr (float): Learning rate for the reconstruction layers
    :param reconstruction_depth (int): Number of reconstruction layers per layer of the DNN
    """
    def __init__(self, net, device: Union[str, torch.device] = 'cpu',
                 lr: float = 1e-3, reconstruction_depth=1):
        self.net = net
        self.lr = lr
        self.device = device
        self.reconstruction_depth = reconstruction_depth
        self.activation_fns = (nn.Tanh, nn.ReLU)

        # construct reconstruction layers
        self.inverse_sequentials = self._construct_inverse_sequentials()
        self.length = len(self.inverse_sequentials)

    def __len__(self) -> int:
        """
        Return the number of reconstruction layers
        :return (int): Number of reconstruction networks
        """
        return self.length

    def train(self):
        """ Set all reconstruction layers to training mode """
        for inverse_sequential in self.inverse_sequentials:
            inverse_sequential.train()

    def eval(self):
        """ Set all reconstruction networks to eval mode """
        for inverse_sequential in self.inverse_sequentials:
            inverse_sequential.eval()

    def save(self, path: str):
        """
        Save reconstruction layers at given location
        :param path (str): File path for file to store data in
        """
        reconstruction_params = [seq.state_dict() for seq in self.inverse_sequentials]
        model_params = self.net.state_dict()
        torch.save(
            {'reconstruction': reconstruction_params,
             'model': model_params},
            path
        )

    def load(self, path: str):
        """
        Load reconstruction networks from file at given path
        :param path (str): File path for file to load data from
        """
        params = torch.load(path, weights_only=True)
        for seq, seq_params in zip(self.inverse_sequentials, params['reconstruction']):
            seq.load_state_dict(seq_params)
        self.net.load_state_dict(params['model'])

    def _construct_inverse_sequentials(self) -> List[nn.Module]:
        """ Helper method to construct the reconstruction layers from the DNN"""
        inverse_sequentials = []
        modules = self.net.modules()
        # get all modules from the DNN
        for layer in modules:
            linear = None
            activation = None
            if isinstance(layer, nn.Linear):
                linear = layer
                layer = next(modules, None)
            if isinstance(layer, self.activation_fns):
                activation = layer
            # we've found a consecutive linear layer and activation function
            # add reconstruction layer with linear layer as input
            if linear is not None and activation is not None:
                inverse_sequential = InverseSequential(
                    linear, activation, self.reconstruction_depth, self.lr)
                inverse_sequential.to(self.device)
                inverse_sequential.register_optimizer()
                inverse_sequentials.append(inverse_sequential)
        return inverse_sequentials

    def train_network(self, train_data, epochs: int = 1, verbose: bool = False):
        """
        Train all reconstruction layers
        :param train_data (Iterable): Training data
        :param epochs (int): Number of epochs to train
        :param verbose (bool): Show progress bar
        :return (np.ndarray): Training losses per epoch per layer
        """
        t1 = time.time()

        self.train()  # training mode
        train_losses: np.ndarray = np.zeros((epochs, len(self)))
        for epoch in range(epochs):
            with tqdm.tqdm(
                    total=len(train_data), disable=(not verbose), miniters=len(train_data)/100
            ) as progress:
                progress.set_description(f'Training epoch {epoch + 1}/{epochs}')

                for X, _ in train_data:
                    X = X.to(self.device)
                    # step through reconstruction layers
                    for j, inverse_sequential in enumerate(self.inverse_sequentials):
                        # compute forward pass
                        with torch.no_grad():
                            X_forward = inverse_sequential.get_forward(X)
                        # compute reconstruction from forward pass
                        X_prediction = inverse_sequential(X_forward)
                        loss = inverse_sequential.loss_fn(X_prediction, X)
                        loss.backward()
                        inverse_sequential.optimizer.step()
                        inverse_sequential.optimizer.zero_grad()

                        # override old x
                        X = X_forward

                        train_losses[epoch][j] += loss.item()

                    progress.set_postfix_str(f'avg loss/layer: {train_losses[epoch].mean().item():.3f}')
                    progress.update()

        if verbose:
            print(f'\rContraNetwork training done in {(time.time() - t1):.3f}s')

        return train_losses

    def eval_network(self, test_data, verbose: bool = False) -> np.ndarray:
        """
        Evaluate the loss of each reconstruction network for the given test data
        :param test_data (Iterable): Validation data
        :return (np.ndarray): Loss of each reconstruction network
        """
        self.eval()
        losses: np.ndarray = np.zeros(self.length)
        with torch.no_grad():
            with tqdm.tqdm(total=len(test_data), disable=(not verbose)) as progress:
                progress.set_description(f'Evaluating...')
                for X, _ in test_data:
                    X = X.to(self.device)
                    # step through reconstruction layers
                    for i, inverse_sequential in enumerate(self.inverse_sequentials):
                        # compute forward pass and reconstruction prediction and generate loss
                        X_forward = inverse_sequential.get_forward(X)
                        X_prediction = inverse_sequential(X_forward)
                        loss = inverse_sequential.loss_fn(X_prediction, X)
                        losses[i] += loss.item()
                        # override old x
                        X = X_forward

                    progress.set_postfix_str(f'avg loss={losses.mean().item():.3f}')
                    progress.update()

        return losses

    def cascade(self, inp: torch.tensor, verbose: bool = False) -> List[torch.tensor]:
        """
        Calculate the reconstructions for each layer of network for the provided input data. This is done by
        combining the reconstruction networks into cascades for each depth in the forward network
        :param inp (torch.Tensor): Input to propagate through DNN and compute the reconstructions for
        :param verbose (bool): Print timing information
        :return (List[torch.Tensor]): Reconstructions for every layer
        """
        t1 = time.time()

        self.eval()  # evaluation mode
        x_forward = inp.to(self.device)
        images = [x_forward.detach().cpu()]  # input is first item of reconstruction array
        with torch.no_grad():
            # step through reconstruction layers
            for i, inverse_sequential in enumerate(self.inverse_sequentials):
                # compute the forward pass and the reconstruction thereof
                x_forward = inverse_sequential.get_forward(x_forward)
                x_inverse = x_forward.detach()
                # propagate the reconstruction all the way to the beginning
                for j in range(i, -1, -1):
                    x_inverse = self.inverse_sequentials[j](x_inverse)
                images.append(x_inverse.detach().cpu())

        if verbose:
            print(f'Cascade done in {(time.time() - t1):.3f}s')
        return images

    def get_perfect(self, index: int, full=0.0) -> torch.tensor:
        """
        Artificially create reconstruction for a maximal identification of a single class
        :param index: int = number of class to create reconstruction for
        :param full: float = activation for the not-wanted-classes
        :return: torch.tensor = artificially created reconstruction
        """
        dim = self.inverse_sequentials[-1].forward_stack[0].out_features
        if dim == 1:
            inp = torch.tensor([[index]])
        else:
            inp = torch.full((1, dim), full)
            inp[0, index] = 1
        inp = inp.to(self.device)
        for c in self.inverse_sequentials[::-1]:
            inp = c(inp)
        return inp.detach().cpu()
