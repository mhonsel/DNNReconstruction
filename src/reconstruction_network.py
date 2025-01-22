import time
import numpy as np
import torch
import torch.nn as nn
import tqdm

from typing import List, Union


class InverseSequential(nn.Module):
    """A reconstruction layer class used to invert a particular layer of a DNN.

    Each instance of this class corresponds to one layer of the original forward network,
    storing both the forward pass (original layer + its activation) and a learned
    inverse stack to reconstruct the input of that layer from its output.

    Args:
        forward_layer (nn.Module): The forward layer (e.g., nn.Linear) from the DNN.
        forward_activation (nn.Module): The activation function of the forward layer.
        depth (int, optional): Number of reconstruction layers (hidden layers) in the inverse stack. Defaults to 1.
        lr (float, optional): Learning rate for the reconstruction optimizer. Defaults to 1e-3.
    """

    def __init__(
        self,
        forward_layer: nn.Module,
        forward_activation: nn.Module,
        depth: int = 1,
        lr: float = 1e-3
    ) -> None:
        super().__init__()

        # Forward stack: the original layer and its activation
        self.forward_stack = nn.Sequential(
            forward_layer,
            forward_activation
        )

        # Build the inverse stack: a series of fully connected layers + Tanh
        features = ([forward_layer.out_features] * depth + [forward_layer.in_features])
        inverse_layers = []
        for i in range(len(features) - 1):
            inverse_layers.append(nn.Linear(features[i], features[i + 1]))
            inverse_layers.append(nn.Tanh())

        # Wrap the inverse layers into a Sequential module
        self.inverse_stack = nn.Sequential(*inverse_layers)

        # Learning rate and optimizer placeholder
        self.lr = lr
        self.optimizer = None
        # Mean Squared Error loss for reconstruction
        self.loss_fn = nn.MSELoss()

    def register_optimizer(self) -> None:
        """Register an optimizer for the inverse stack after the network is initialized."""
        self.optimizer = torch.optim.Adam(self.inverse_stack.parameters(), lr=self.lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the inverse stack.

        Args:
            x (torch.Tensor): The input tensor of shape [batch_size, out_features]
                from the forward layer.

        Returns:
            torch.Tensor: The reconstructed tensor of shape [batch_size, in_features].
        """
        return self.inverse_stack(x)

    def get_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Obtain the output of the corresponding forward layer from the DNN.

        Args:
            x (torch.Tensor): The input to the original forward layer (shape [batch_size, in_features]).

        Returns:
            torch.Tensor: The output of the forward layer (shape [batch_size, out_features]).
        """
        # We do not track gradients here, as we only need the forward output
        with torch.no_grad():
            x_input = self.forward_stack(x)
            # Detach from the current computation graph
            x_input = x_input.detach()
            return x_input


class ReconstructionNetwork:
    """A network that holds inverse reconstruction layers for each forward layer of a DNN.

    This class reconstructs the input of each layer given its output, effectively
    creating 'cascades' that can invert each layer step by step.

    Args:
        net (nn.Module): The original DNN from which layers will be inverted.
        device (Union[str, torch.device], optional): Device to perform calculations on.
            Defaults to 'cpu'.
        lr (float, optional): Learning rate for the reconstruction layers. Defaults to 1e-3.
        reconstruction_depth (int, optional): Number of hidden layers in each reconstruction
            stack of the inverse modules. Defaults to 1.
    """

    def __init__(
        self,
        net: nn.Module,
        device: Union[str, torch.device] = 'cpu',
        lr: float = 1e-3,
        reconstruction_depth: int = 1
    ) -> None:
        self.net = net
        self.lr = lr
        self.device = device
        self.reconstruction_depth = reconstruction_depth

        # Tuple of activation functions we expect to see in the forward net
        self.activation_fns = (nn.Tanh, nn.ReLU)

        # Construct an inverse module for each (linear+activation) pair in the net
        self.inverse_sequentials = self._construct_inverse_sequentials()
        self.length = len(self.inverse_sequentials)

    def __len__(self) -> int:
        """Get the number of reconstruction layers."""
        return self.length

    def train(self) -> None:
        """Set all reconstruction layers to training mode."""
        for inverse_sequential in self.inverse_sequentials:
            inverse_sequential.train()

    def eval(self) -> None:
        """Set all reconstruction networks to evaluation mode."""
        for inverse_sequential in self.inverse_sequentials:
            inverse_sequential.eval()

    def save(self, path: str) -> None:
        """Save reconstruction layers and the original model to a file.

        Args:
            path (str): File path to store the data.
        """
        # State dictionaries of each reconstruction layer
        reconstruction_params = [seq.state_dict() for seq in self.inverse_sequentials]
        # State dictionary of the original model
        model_params = self.net.state_dict()
        torch.save(
            {
                'reconstruction': reconstruction_params,
                'model': model_params
            },
            path
        )

    def load(self, path: str) -> None:
        """Load reconstruction networks and the original model from a file.

        Args:
            path (str): File path to load the data from.
        """
        params = torch.load(path, weights_only=True)
        # Load each inverse layer's state dictionary
        for seq, seq_params in zip(self.inverse_sequentials, params['reconstruction']):
            seq.load_state_dict(seq_params)
        # Load the original model's state dictionary
        self.net.load_state_dict(params['model'])

    def _construct_inverse_sequentials(self) -> List[nn.Module]:
        """Construct inverse sequential modules for each layer in the DNN."""
        inverse_sequentials = []
        # Get all modules from the DNN
        modules = self.net.modules()

        for layer in modules:
            linear = None
            activation = None

            # If the layer is a linear layer, store it temporarily
            if isinstance(layer, nn.Linear):
                linear = layer
                # Move to the next module to see if there's a matching activation
                layer = next(modules, None)

            # If the next layer is an expected activation, store it as well
            if isinstance(layer, self.activation_fns):
                activation = layer

            # If we found a linear+activation pair, create an InverseSequential
            if linear is not None and activation is not None:
                inverse_sequential = InverseSequential(
                    linear,
                    activation,
                    self.reconstruction_depth,
                    self.lr
                )
                # Move reconstruction layer to the designated device
                inverse_sequential.to(self.device)
                # Register optimizer after creation
                inverse_sequential.register_optimizer()
                inverse_sequentials.append(inverse_sequential)

        return inverse_sequentials

    def train_network(
        self,
        train_data: torch.utils.data.DataLoader,
        epochs: int = 1,
        verbose: bool = False
    ) -> np.ndarray:
        """Train all reconstruction layers on the given training data.

        Args:
            train_data (torch.utils.data.DataLoader): The training data loader
                providing (input, label) batches.
            epochs (int, optional): Number of epochs to train for. Defaults to 1.
            verbose (bool, optional): Whether to show a progress bar. Defaults to False.

        Returns:
            np.ndarray: A 2D array of shape [epochs, number_of_layers] containing the
                total training loss per layer for each epoch.
        """
        t1 = time.time()
        # Set reconstruction layers to training mode
        self.train()

        # Storage for training losses: one row per epoch, one column per layer
        train_losses: np.ndarray = np.zeros((epochs, len(self)))

        for epoch in range(epochs):
            # Create a progress bar
            with tqdm.tqdm(
                total=len(train_data), disable=(not verbose), miniters=len(train_data)/100
            ) as progress:
                progress.set_description(f'Training epoch {epoch + 1}/{epochs}')

                # Loop over the training data
                for X, _ in train_data:
                    X = X.to(self.device)

                    # Iterate through each inverse layer sequentially
                    for j, inverse_sequential in enumerate(self.inverse_sequentials):
                        # Forward pass through the corresponding original layer
                        with torch.no_grad():
                            X_forward = inverse_sequential.get_forward(X)

                        # Reconstruct the input of the forward layer
                        X_prediction = inverse_sequential(X_forward)

                        # Compute MSE loss between reconstruction and actual input
                        loss = inverse_sequential.loss_fn(X_prediction, X)

                        # Backpropagate
                        loss.backward()
                        inverse_sequential.optimizer.step()
                        inverse_sequential.optimizer.zero_grad()

                        # Update the total loss for this layer
                        train_losses[epoch][j] += loss.item()

                        # Override X with the forward pass for the next inverse layer
                        X = X_forward

                    # Display running average loss per layer in the progress bar
                    progress.set_postfix_str(f'avg loss/layer: {train_losses[epoch].mean():.3f}')
                    progress.update()

        if verbose:
            print(f'\rReconstructionNetwork training done in {(time.time() - t1):.3f}s')

        return train_losses

    def eval_network(
        self,
        test_data: torch.utils.data.DataLoader,
        verbose: bool = False
    ) -> np.ndarray:
        """Evaluate the reconstruction loss of each inverse layer on the test data.

        Args:
            test_data (torch.utils.data.DataLoader): The test data loader
                providing (input, label) batches.
            verbose (bool, optional): Whether to show a progress bar. Defaults to False.

        Returns:
            np.ndarray: A 1D array with the accumulated loss for each reconstruction layer.
        """
        # Set reconstruction layers to evaluation mode
        self.eval()

        # Store cumulative losses for each layer
        losses: np.ndarray = np.zeros(self.length)

        with torch.no_grad():
            with tqdm.tqdm(total=len(test_data), disable=(not verbose)) as progress:
                progress.set_description('Evaluating...')
                for X, _ in test_data:
                    X = X.to(self.device)

                    # Pass through each inverse layer
                    for i, inverse_sequential in enumerate(self.inverse_sequentials):
                        X_forward = inverse_sequential.get_forward(X)
                        X_prediction = inverse_sequential(X_forward)
                        loss = inverse_sequential.loss_fn(X_prediction, X)

                        # Accumulate loss
                        losses[i] += loss.item()

                        # Override X with the forward output
                        X = X_forward

                    progress.set_postfix_str(f'avg loss={losses.mean():.3f}')
                    progress.update()

        return losses

    def cascade(self, inp: torch.Tensor, verbose: bool = False) -> List[torch.Tensor]:
        """Compute the reconstruction cascades for each layer of the network.

        This method takes an input, feeds it forward through each layer of the DNN,
        and then reconstructs backwards through all previously visited layers.

        Args:
            inp (torch.Tensor): The input tensor to the entire DNN of shape
                [batch_size, in_features].
            verbose (bool, optional): Whether to print timing information. Defaults to False.

        Returns:
            List[torch.Tensor]: A list of reconstructed tensors, one for each layer
        """
        t1 = time.time()

        self.eval()  # Switch to evaluation mode
        x_forward = inp.to(self.device)
        # The first item in the list is the original input
        images = [x_forward.detach().cpu()]

        with torch.no_grad():
            # Step through each inverse layer in order
            for i, inverse_sequential in enumerate(self.inverse_sequentials):
                # Get the forward output for the current layer
                x_forward = inverse_sequential.get_forward(x_forward)
                x_inverse = x_forward.detach()

                # Propagate the reconstruction backwards through all layers up to this point
                for j in range(i, -1, -1):
                    x_inverse = self.inverse_sequentials[j](x_inverse)

                # Save the reconstructed version at this stage
                images.append(x_inverse.detach().cpu())

        if verbose:
            print(f'Cascade done in {(time.time() - t1):.3f}s')
        return images

    def get_perfect(self, index: int, full: float = 0.0) -> torch.Tensor:
        """Construct a synthetic reconstruction that maximally activates a single output class.

        This method creates a one-hot or pseudo one-hot vector at the final layer
        (depending on the dimension of the output), then inverts it back through all
        reconstruction layers to generate a synthetic "perfect" input for that class.

        Args:
            index (int): The class index to maximize.
            full (float, optional): The activation level for other classes. Defaults to 0.0.

        Returns:
            torch.Tensor: The artificially created reconstruction for the specified class.
        """
        # Determine the dimensionality of the final forward layer
        dim = self.inverse_sequentials[-1].forward_stack[0].out_features

        # If the final layer output dimension is 1, treat it as a single-value output
        if dim == 1:
            inp = torch.tensor([[index]], dtype=torch.float32)
        else:
            # Otherwise, fill a vector with `full` for all classes
            inp = torch.full((1, dim), full, dtype=torch.float32)
            # Set the target class to 1
            inp[0, index] = 1.0

        # Move this synthetic vector to the correct device
        inp = inp.to(self.device)

        # Invert backwards through all inverse layers
        for c in reversed(self.inverse_sequentials):
            inp = c(inp)

        return inp.detach().cpu()