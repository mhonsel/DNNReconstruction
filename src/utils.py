import copy
import time
import numpy as np
import torch
import torch.nn as nn
import tqdm

import matplotlib.pyplot as plt

from functools import wraps
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Union, List, Callable, Tuple, Optional

from sklearn.metrics import roc_curve, roc_auc_score

from src.dataset import DatasetGenerator
from src.entropies import diff_entropy, diff_entropy_clamped, shannon_entropy
from src.network import FC
from src.reconstruction_network import ReconstructionNetwork


def is_number(s: str) -> bool:
    """Check whether a given string can be interpreted as a float.

    Args:
        s (str): The input string to check.

    Returns:
        bool: True if the string can be converted to a float, False otherwise.
    """
    try:
        float(s)
        return True
    except ValueError:
        return False


def shannon_entropy_from_array(
    arr: np.ndarray, bins: int, normalize: bool = False
) -> float:
    """Compute the Shannon entropy of a given numpy array via histogram binning.

    Args:
        arr (np.ndarray): Input data as a NumPy array.
        bins (int): Number of bins to use for the histogram.
        normalize (bool, optional): Whether to normalize by array size. Defaults to True.

    Returns:
        float: The computed Shannon entropy.
    """
    # Create histogram probabilities
    p = np.histogram(arr.ravel(), bins=bins)[0] / arr.size
    # Retain only nonzero bins
    p = p[p > 0]
    # Compute the raw Shannon entropy, i.e. -\sum p * log(p)
    H = -np.sum(p * np.log2(p))

    # Optionally normalize by array size
    if normalize:
        return H / arr.size
    else:
        return H


def timeit(func: Callable) -> Callable:
    """Decorator to measure the execution time of a function.

    Args:
        func (Callable): The function to time.

    Returns:
        Callable: A wrapped function that prints its execution time.
    """
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(
            f'Function {func.__name__}{args} {kwargs} '
            f'Took {total_time:.4f} seconds'
        )
        return result
    return timeit_wrapper


def parse_model_path(file_path: Union[str, Path]) -> Tuple[int, int, int, nn.Module]:
    """Parse a model file path to extract architecture parameters and the activation function.

    The file name is expected to be in the format:
        `[something]_F[in_features]_W[width]_D[depth]_[activation]_[anything else].pt`

    Args:
        file_path (Union[str, Path]): Path to the model file.

    Returns:
        Tuple[int, int, int, nn.Module]: A tuple containing:
            - in_features (int)
            - width (int)
            - depth (int)
            - activation_fn (nn.Module)
    """
    file_name = Path(file_path).stem
    in_features, width, depth = [
        int(item[1:]) for item in file_name.split('_')[1:4]
    ]
    activation = file_name.split('_')[4]
    if activation == 'Tanh':
        activation_fn = nn.Tanh
    elif activation == 'ReLU':
        activation_fn = nn.ReLU
    else:
        raise ValueError('Supported activation functions are Tanh and ReLU.')

    return in_features, width, depth, activation_fn


def load_model(
    file_path: Union[str, Path],
    model: Optional[nn.Module] = None,
    device: Union[str, torch.device] = 'cpu'
) -> nn.Module:
    """Load a saved model from disk.

    If no model is provided, the function will parse the file path and
    construct a new FC model using the extracted parameters.

    Args:
        file_path (Union[str, Path]): File path of the saved model.
        model (nn.Module, optional): An existing model to load state_dict into.
            If None, creates a new model with parameters inferred from the file name.
            Defaults to None.
        device (Union[str, torch.device], optional): Device to move the model to.
            Defaults to 'cpu'.

    Returns:
        nn.Module: The model with loaded state.
    """
    if model is None:
        in_features, width, depth, activation = parse_model_path(file_path)
        model = FC(
            in_features=in_features,
            width=width,
            depth=depth,
            variances=None,
            activation=activation
        )
        model.to(device)
        model.eval()

    model.load_state_dict(torch.load(file_path, weights_only=True))
    return model


def load_reconstruction_network(
    file_path: Union[str, Path],
    re_net: Optional[ReconstructionNetwork] = None,
    device: Union[str, torch.device] = 'cpu'
) -> Tuple[nn.Module, ReconstructionNetwork]:
    """Load a reconstruction network (and its underlying model) from disk.

    If no reconstruction network is provided, a new one will be built
    based on parameters inferred from the file path.

    Args:
        file_path (Union[str, Path]): File path of the saved reconstruction network.
        re_net (ReconstructionNetwork, optional): An existing reconstruction network.
            If None, creates a new one. Defaults to None.
        device (Union[str, torch.device], optional): Device to move the network to.
            Defaults to 'cpu'.

    Returns:
        Tuple[nn.Module, ReconstructionNetwork]: The underlying model (nn.Module) and
            the loaded ReconstructionNetwork.
    """
    if re_net is None:
        in_features, width, depth, activation = parse_model_path(file_path)
        model = FC(
            in_features=in_features,
            width=width,
            depth=depth,
            variances=None,
            activation=activation
        )
        model.to(device)
        model.eval()
        re_net = ReconstructionNetwork(model, device=device, lr=1e-3)

    re_net.load(file_path)
    return re_net.net, re_net


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 10,
    lr: float = 0.001,
    l2_penalty: float = 0,
    device: Union[str, torch.device] = 'cpu',
    optimizer: str = 'Adam',
    early_stopping: int = 0,
    model_save_path: str = '',
    verbose: bool = True
) -> dict:
    """Train a binary classification model using BCEWithLogitsLoss.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Dataloader for training data.
        val_loader (DataLoader): Dataloader for validation data.
        n_epochs (int, optional): Number of epochs to train. Defaults to 10.
        lr (float, optional): Learning rate. Defaults to 0.001.
        l2_penalty (float, optional): Weight decay (L2 penalty). Defaults to 0.
        device (Union[str, torch.device], optional): Device to train on. Defaults to 'cpu'.
        optimizer (str, optional): Optimizer type, 'Adam' or 'SGD'. Defaults to 'Adam'.
        early_stopping (int, optional): Number of epochs with no improvement
            after which training will be stopped. If 0, early stopping is disabled.
            Defaults to 0.
        model_save_path (str, optional): File path to save the best model.
            If empty, no checkpoint is saved. Defaults to ''.
        verbose (bool, optional): Whether to print progress and information.
            Defaults to True.

    Returns:
        dict: A dictionary containing training and validation losses and accuracies:
            {
                'train_loss': [...],
                'train_acc': [...],
                'val_loss': [...],
                'val_acc': [...]
            }
    """
    model.to(device)
    loss_fn = nn.BCEWithLogitsLoss()

    # Choose optimizer
    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_penalty)
    elif optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=l2_penalty)
    else:
        raise ValueError("Supported optimizers are 'Adam' or 'SGD'.")

    stopping_counter = 0
    model_state_dict = None

    # Track training and validation performance
    history = {
        'train_acc': [],
        'train_loss': [],
        'val_acc': [],
        'val_loss': []
    }

    for epoch in range(n_epochs):
        train_loss = 0.0
        train_acc = 0.0
        val_loss = 0.0
        val_acc = 0.0

        model.train()
        with tqdm.tqdm(total=len(train_loader), disable=(not verbose)) as progress:
            progress.set_description(f'Training epoch {epoch + 1}/{n_epochs}')
            for X, y in train_loader:
                X = X.to(device)
                y = y.to(device).squeeze()

                # Forward pass
                y_pred = model(X).squeeze()
                loss = loss_fn(y_pred, y.float())
                train_loss += loss.item()

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Compute accuracy: logistic output > 0 => class 1
                acc = ((y_pred > 0) == y).float().mean().item()
                train_acc += acc

                # Update progress
                progress.set_postfix_str(f'loss={loss.item():.3f}, acc={acc:.3f}')
                progress.update()

        # Validation
        model.eval()
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                y = y.to(device)
                y_pred = model(X)
                loss = loss_fn(y_pred, y.float()).item()
                val_loss += loss

                acc = ((y_pred > 0) == y).float().mean().item()
                val_acc += acc

        # Average losses and accuracies
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc /= len(train_loader)
        val_acc /= len(val_loader)

        # Record metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if verbose:
            print(
                f'Train loss/accuracy: {train_loss:.3f}/{train_acc:.3f}  '
                f'\tValidation loss/accuracy: {val_loss:.3f}/{val_acc:.3f}'
            )

        # Check if validation accuracy improved
        if val_acc == max(history['val_acc']):
            model_state_dict = copy.deepcopy(model.state_dict())
            stopping_counter = 0

            # Save model if a path is provided
            if model_save_path:
                torch.save(model_state_dict, model_save_path)
                if verbose:
                    print('Saved model state.')
        elif early_stopping:
            # Increment the early stopping counter if no improvement
            if stopping_counter < early_stopping:
                stopping_counter += 1
            if stopping_counter == early_stopping:
                # Restore best parameters
                model.load_state_dict(model_state_dict)
                if verbose:
                    print('Loading best params, exiting training loop.')
                break

    return history


def plot_roc(model: nn.Module, data: torch.Tensor, labels: torch.Tensor) -> None:
    """Plot the ROC curve given a model and data.

    Args:
        model (nn.Module): A trained binary classification model.
        data (torch.Tensor): The input data of shape [batch_size, features].
        labels (torch.Tensor): Ground-truth labels of shape [batch_size].
    """
    model.eval()
    with torch.no_grad():
        logits = model(data)

    # Convert logits to probabilities using Sigmoid
    sigmoid = nn.Sigmoid()
    probabilities = sigmoid(logits)

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(labels.cpu(), probabilities.cpu())

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='ROC curve')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

    # Calculate and print AUC
    auc = roc_auc_score(labels.cpu(), probabilities.cpu())
    print(f'AUC: {auc}')


def get_activation_fn(activation: str) -> nn.Module:
    """Retrieve an activation function based on a string identifier.

    Args:
        activation (str): The name of the activation function ('Tanh' or 'ReLU').

    Returns:
        nn.Module: The corresponding PyTorch activation class.

    Raises:
        ValueError: If the provided activation is not recognized.
    """
    if activation == 'Tanh':
        return nn.Tanh
    elif activation == 'ReLU':
        return nn.ReLU
    else:
        raise ValueError('Supported activation functions are Tanh and ReLU.')


def generate_entropy_data(
    output_path: Path,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: Union[str, torch.device] = 'cpu',
    epochs: int = 2,
    network_input_width: int = 7500,
    network_width: int = 256,
    network_depth: int = 50,
    activation: str = 'Tanh',
    variance_min: float = 0.5,
    variance_max: float = 5.0,
    variance_step: float = 0.5,
    entropy: str = 'differential',
    entropy_fn_kwargs: Optional[dict] = None,
    comment: str = ''
) -> Path:
    """Generate and save entropy data for a range of weight variances.

    This function trains a reconstruction network for each variance value, then
    measures entropies of intermediate activations using a specified entropy function.

    Args:
        output_path (Path): Path to which the results will be saved.
        train_loader (DataLoader): Dataloader for training the reconstruction network.
        test_loader (DataLoader): Dataloader for obtaining sample data to compute entropy.
        device (Union[str, torch.device], optional): Device to perform computations on.
            Defaults to 'cpu'.
        epochs (int, optional): Number of epochs to train each reconstruction. Defaults to 2.
        network_input_width (int, optional): Input dimension for the FC network. Defaults to 7500.
        network_width (int, optional): Width of hidden layers. Defaults to 256.
        network_depth (int, optional): Number of layers in the FC network. Defaults to 50.
        activation (str, optional): Activation function name ('Tanh' or 'ReLU'). Defaults to 'Tanh'.
        variance_min (float, optional): Minimum weight variance to explore. Defaults to 0.5.
        variance_max (float, optional): Maximum weight variance to explore. Defaults to 5.0.
        variance_step (float, optional): Step size for weight variance. Defaults to 0.5.
        entropy (str, optional): Which entropy function to use ('differential', 'differential-clamped', 'shannon').
            Defaults to 'differential'.
        entropy_fn_kwargs (dict, optional): Additional keyword args to pass to the entropy function. Defaults to None.
        comment (str, optional): Additional string to append to the file name. Defaults to ''.

    Returns:
        Path: The file path where the entropy data is saved.

    Raises:
        ValueError: If the specified entropy function is not recognized.
    """
    weight_variances: np.ndarray = np.arange(variance_min, variance_max + variance_step, variance_step)

    if comment:
        comment = f'_{comment}'
    file_name = '{}-entropy_F{}_W{}_D{}_{}_{}_{}_{}{}.npy'.format(
        entropy,
        network_input_width,
        network_width,
        network_depth,
        activation,
        variance_min,
        variance_max,
        variance_step,
        comment.replace(' ', '-')
    )

    # Select the entropy function
    activation_fn = get_activation_fn(activation)
    if entropy == 'differential':
        entropy_fn = diff_entropy
    elif entropy == 'differential-clamped':
        entropy_fn = diff_entropy_clamped
    elif entropy == 'shannon':
        entropy_fn = shannon_entropy
    else:
        raise ValueError('Supported entropy functions are "differential" and "shannon".')
    entropy_fn_kwargs = entropy_fn_kwargs if entropy_fn_kwargs is not None else {}

    # Prepare array to hold entropy results
    entropy_data: np.ndarray = np.zeros((len(weight_variances), network_depth - 1))

    with tqdm.tqdm(total=len(weight_variances)) as progress:
        progress.set_description('Generating entropy data')
        for i, var_w in enumerate(weight_variances):
            # Create a new FC and ReconstructionNetwork for each variance
            net = FC(
                in_features=network_input_width,
                width=network_width,
                depth=network_depth,
                variances=(var_w, 0.05),
                activation=activation_fn
            )
            re_net = ReconstructionNetwork(net, device, lr=5e-4)

            # Train the reconstruction network
            _ = re_net.train_network(train_loader, epochs=epochs, verbose=False)

            # Obtain a batch of test data and compute cascades
            X, _ = next(iter(test_loader))
            cascades = re_net.cascade(X)

            # Compute entropy for each intermediate layer (excluding first and last)
            entropies = [entropy_fn(c, **entropy_fn_kwargs).tolist() for c in cascades[1:-1]]
            entropies.insert(0, var_w)
            entropy_data[i] = entropies

            # Save intermediate results
            with open(output_path / file_name, 'wb') as f:
                np.save(f, entropy_data)

            progress.update()

    return output_path / file_name


def generate_accuracy_data(
    output_path: Path,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: Union[str, torch.device] = 'cpu',
    epochs: int = 50,
    early_stopping: int = 8,
    lr: float = 0.001,
    optimizer: str = 'Adam',
    network_input_width: int = 7500,
    network_width: int = 256,
    activation: str = 'Tanh',
    max_depth: int = 50,
    depth_step: int = 5,
    variance_min: float = 0.5,
    variance_max: float = 5.0,
    variance_step: float = 0.5,
    comment: str = ''
) -> Path:
    """Generate accuracy data for a range of network depths and weight variances.

    For each combination of depth and variance, train a model and record
    the maximum validation accuracy and the epoch at which it occurs.

    Args:
        output_path (Path): Directory to save the generated accuracy data.
        train_loader (DataLoader): Dataloader for training data.
        val_loader (DataLoader): Dataloader for validation data.
        device (Union[str, torch.device], optional): Device to train on. Defaults to 'cpu'.
        epochs (int, optional): Number of epochs for each model. Defaults to 50.
        early_stopping (int, optional): Early stopping patience in epochs. Defaults to 8.
        lr (float, optional): Learning rate. Defaults to 0.001.
        optimizer (str, optional): Optimizer name ('Adam' or 'SGD'). Defaults to 'Adam'.
        network_input_width (int, optional): Input dimension. Defaults to 7500.
        network_width (int, optional): Width of hidden layers. Defaults to 256.
        activation (str, optional): Activation function name. Defaults to 'Tanh'.
        max_depth (int, optional): Maximum depth to explore. Defaults to 50.
        depth_step (int, optional): Step size between depths. Defaults to 5.
        variance_min (float, optional): Minimum variance for weights. Defaults to 0.5.
        variance_max (float, optional): Maximum variance for weights. Defaults to 5.0.
        variance_step (float, optional): Step size for variance. Defaults to 0.5.
        comment (str, optional): Additional text appended to file name. Defaults to ''.

    Returns:
        Path: The file path to which the accuracy data was saved.
    """
    weight_variances = np.arange(variance_min, variance_max + variance_step, variance_step)

    if comment:
        comment = f'_{comment}'
    file_name = 'accuracy_F{}_W{}_S{}_{}_{}_{}_{}_{}{}.npy'.format(
        network_input_width,
        network_width,
        depth_step,
        activation,
        optimizer,
        variance_min,
        variance_max,
        variance_step,
        comment.replace(' ', '-')
    )

    activation_fn = get_activation_fn(activation)
    depths = range(depth_step, max_depth + 1, depth_step)

    # The array shape: [len(weight_variances), len(depths) + 1, 2]
    # The +1 column in the second dimension holds the variance as row 0.
    # Each row = [variance, [accuracy, epoch], [accuracy, epoch] ...]
    accuracy_data = np.zeros((len(weight_variances), len(depths) + 1, 2))

    with tqdm.tqdm(total=len(weight_variances) * len(depths)) as progress:
        progress.set_description('Generating accuracy data')
        for i, var_w in enumerate(weight_variances):
            accuracy_data[i][0] = var_w  # store variance
            for j, depth in enumerate(depths):
                # Build and train model for this variance and depth
                net = FC(
                    in_features=network_input_width,
                    width=network_width,
                    depth=depth,
                    variances=(var_w, 0.05),
                    activation=activation_fn
                )

                history = train_model(
                    net,
                    train_loader,
                    val_loader,
                    n_epochs=epochs,
                    lr=lr,
                    device=device,
                    optimizer=optimizer,
                    early_stopping=early_stopping,
                    model_save_path='',
                    verbose=False
                )

                # Determine maximum validation accuracy and the epoch it occurred
                max_accuracy = max(history['val_acc'])
                max_accuracy_epochs = history['val_acc'].index(max_accuracy) + 1

                # Store results
                accuracy_data[i][j + 1][0] = max_accuracy
                accuracy_data[i][j + 1][1] = max_accuracy_epochs

                # Save partial results
                with open(output_path / file_name, 'wb') as f:
                    np.save(f, accuracy_data)

                progress.update()

    return output_path / file_name


def generate_maxpool_data(
    output_path: Path,
    maxpool_ops: List,
    dataset_kwargs: dict,
    device: Union[str, torch.device] = 'cpu',
    epochs: int = 3,
    network_depth: int = 120,
    activation: str = 'Tanh',
    variance_min: float = 0.5,
    variance_max: float = 6.0,
    variance_step: float = 0.5,
    entropy_bs: int = 1000,
    compute_shannon_entropy: bool = True,
    comment: str = ''
) -> Union[Path, Tuple[Path, Path]]:
    """Generate differential or Shannon entropy data for networks with different maxpool operations.

    Args:
        output_path (Path): Directory to save generated data.
        maxpool_ops (List): A list of transforms or transformations including MaxPool ops.
        dataset_kwargs (dict): Arguments to construct the dataset (e.g., for DatasetGenerator).
        device (Union[str, torch.device], optional): Device to use. Defaults to 'cpu'.
        epochs (int, optional): Number of epochs for each reconstruction network. Defaults to 3.
        network_depth (int, optional): Number of layers in the FC network. Defaults to 120.
        activation (str, optional): Activation function name. Defaults to 'Tanh'.
        variance_min (float, optional): Minimum variance. Defaults to 0.5.
        variance_max (float, optional): Maximum variance. Defaults to 6.0.
        variance_step (float, optional): Step for variance range. Defaults to 0.5.
        entropy_bs (int, optional): Batch size for computing entropy. Defaults to 1000.
        compute_shannon_entropy (bool, optional): Whether to also compute Shannon entropy.
            Defaults to True.
        comment (str, optional): Additional text for file naming. Defaults to ''.

    Returns:
        Union[Path, Tuple[Path, Path]]: Path(s) to the saved file(s). If `compute_shannon_entropy`
            is True, returns a tuple of paths (diff_entropy_path, shannon_entropy_path).
            Otherwise, returns a single path for differential entropy.
    """
    weight_variances: np.ndarray = np.arange(variance_min, variance_max + variance_step, variance_step)
    maxpool_sizes = [mp[-2].kernel_size if len(mp) > 1 else 1 for mp in maxpool_ops]
    network_features = [mp[-1].new_shape[0] for mp in maxpool_ops]
    network_widths = [min(feat, 1000) for feat in network_features]

    if comment:
        comment = f'_{comment}'

    file_name_diff = 'mp-data-differential-entropy_D{}_{}_{}_{}_{}{}.npy'.format(
        network_depth, activation,
        variance_min, variance_max, variance_step,
        comment.replace(' ', '-')
    )
    file_name_shannon = 'mp-data-shannon-entropy_D{}_{}_{}_{}_{}{}.npy'.format(
        network_depth, activation,
        variance_min, variance_max, variance_step,
        comment.replace(' ', '-')
    )

    activation_fn = get_activation_fn(activation)

    # Prepare arrays to store results
    mp_diff_entropy: np.ndarray = np.zeros((len(maxpool_ops) + 1, len(weight_variances) + 1))
    mp_diff_entropy[0, 1:] = weight_variances
    mp_diff_entropy[1:, 0] = maxpool_sizes

    if compute_shannon_entropy:
        mp_shannon_entropy = np.copy(mp_diff_entropy)

    # Iterate over each MaxPool transform
    for i, mp in enumerate(maxpool_ops):
        # Construct dataset for this specific MaxPool transform
        dataset_kwargs['transforms'] = mp
        dataset_generator = DatasetGenerator(**dataset_kwargs)
        train_data, val_data = dataset_generator.generate()

        with tqdm.tqdm(total=len(weight_variances)) as progress:
            progress.set_description(f'Generating data for MaxPool({maxpool_sizes[i]})')
            for j, var_w in enumerate(weight_variances):
                net = FC(
                    in_features=network_features[i],
                    width=network_widths[i],
                    depth=network_depth,
                    variances=(var_w, 0.05),
                    activation=activation_fn
                )
                re_net = ReconstructionNetwork(net, device, lr=5e-4)

                train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
                _ = re_net.train_network(train_loader, epochs=epochs, verbose=False)

                # Compute the final layer reconstruction entropy
                entropy_loader = DataLoader(val_data, batch_size=entropy_bs, shuffle=False)
                xb, _ = next(iter(entropy_loader))
                cascades = re_net.cascade(xb)

                # Use diff_entropy_clamped on the last cascade
                mp_diff_entropy[i + 1, j + 1] = diff_entropy_clamped(cascades[-1]).item()

                # Compute Shannon entropy if requested
                if compute_shannon_entropy:
                    mp_shannon_entropy[i + 1, j + 1] = shannon_entropy(cascades[-1], entropy_bs).item()

                # Save partial results
                with open(output_path / file_name_diff, 'wb') as f:
                    np.save(f, mp_diff_entropy)
                if compute_shannon_entropy:
                    with open(output_path / file_name_shannon, 'wb') as f:
                        np.save(f, mp_shannon_entropy)

                progress.update()

    # Return paths to the saved files
    if compute_shannon_entropy:
        return output_path / file_name_diff, output_path / file_name_shannon
    else:
        return output_path / file_name_diff


def generate_maxpool_entropy_data(
    output_path: Path,
    maxpool_ops: List,
    dataset_kwargs: dict,
    device: Union[str, torch.device] = 'cpu',
    epochs: int = 3,
    network_depth: int = 120,
    activation: str = 'Tanh',
    variance_min: float = 0.5,
    variance_max: float = 6.0,
    variance_step: float = 0.5,
    entropy_bs: int = 1000,
    comment: str = ''
) -> Path:
    """Generate differential entropy data for cascades at each layer of a network
    with different MaxPool transforms.

    Args:
        output_path (Path): Directory to save results.
        maxpool_ops (List): A list of transformations including MaxPool ops.
        dataset_kwargs (dict): Arguments to build the DatasetGenerator.
        device (Union[str, torch.device], optional): Device to run on. Defaults to 'cpu'.
        epochs (int, optional): Number of epochs for training reconstruction networks. Defaults to 3.
        network_depth (int, optional): Depth of the FC network. Defaults to 120.
        activation (str, optional): Activation function name. Defaults to 'Tanh'.
        variance_min (float, optional): Minimum variance to explore. Defaults to 0.5.
        variance_max (float, optional): Maximum variance to explore. Defaults to 6.0.
        variance_step (float, optional): Step size for variance range. Defaults to 0.5.
        entropy_bs (int, optional): Batch size for computing entropy. Defaults to 1000.
        comment (str, optional): Extra comment appended to file name. Defaults to ''.

    Returns:
        Path: The file path where the generated data is saved.
    """
    weight_variances: np.ndarray = np.arange(variance_min, variance_max + variance_step, variance_step)
    maxpool_sizes = [mp[-2].kernel_size if len(mp) > 1 else 1 for mp in maxpool_ops]
    network_features = [mp[-1].new_shape[0] for mp in maxpool_ops]
    network_widths = [min(feat, 1000) for feat in network_features]

    if comment:
        comment = f'_{comment}'
    file_name = 'mp-data-entropy_D{}_{}_{}_{}_{}{}.npy'.format(
        network_depth, activation,
        variance_min, variance_max, variance_step,
        comment.replace(' ', '-')
    )

    activation_fn = get_activation_fn(activation)

    # Shape: [len(maxpool_ops)+1, len(weight_variances)+1, network_depth-1]
    # The first row/column track indices/variance placeholders.
    mp_entropy: np.ndarray = np.zeros((len(maxpool_ops) + 1, len(weight_variances) + 1, network_depth - 1))
    mp_entropy[0, 1:, 0] = weight_variances
    mp_entropy[1:, 0, 0] = maxpool_sizes
    # For convenience, embed some info in [0,0,1:]
    mp_entropy[0, 0, 1:] = np.arange(network_depth - 2)

    for i, mp in enumerate(maxpool_ops):
        # Construct dataset for the transforms
        dataset_kwargs['transforms'] = mp
        dataset_generator = DatasetGenerator(**dataset_kwargs)
        train_data, val_data = dataset_generator.generate()

        with tqdm.tqdm(total=len(weight_variances)) as progress:
            progress.set_description(f'Generating data for MaxPool({maxpool_sizes[i]})')
            for j, var_w in enumerate(weight_variances):
                net = FC(
                    in_features=network_features[i],
                    width=network_widths[i],
                    depth=network_depth,
                    variances=(var_w, 0.05),
                    activation=activation_fn
                )
                re_net = ReconstructionNetwork(net, device, lr=5e-4)

                train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
                _ = re_net.train_network(train_loader, epochs=epochs, verbose=False)

                # Compute reconstruction entropies at each layer
                entropy_loader = DataLoader(val_data, batch_size=entropy_bs, shuffle=False)
                xb, _ = next(iter(entropy_loader))
                cascades = re_net.cascade(xb)

                # Calculate diff entropy for each cascade except the first and last
                entropies = [diff_entropy(c).tolist() for c in cascades[1:-1]]
                mp_entropy[i + 1, j + 1, 1:] = np.array(entropies)

                # Save intermediate results
                with open(output_path / file_name, 'wb') as f:
                    np.save(f, mp_entropy)

                progress.update()

    return output_path / file_name


def generate_maxpool_accuracy_data(
    output_path: Path,
    maxpool_ops: List,
    dataset_kwargs: dict,
    device: Union[str, torch.device] = 'cpu',
    epochs: int = 50,
    early_stopping: int = 8,
    lr: float = 0.001,
    optimizer: str = 'Adam',
    activation: str = 'Tanh',
    network_depth: int = 120,
    depth_step: int = 0,
    variance_min: float = 0.5,
    variance_max: float = 6.0,
    variance_step: float = 0.5,
    comment: str = ''
) -> Path:
    """Generate accuracy data for networks with MaxPool transforms across multiple variances.

    Args:
        output_path (Path): Directory to save the results.
        maxpool_ops (List): List of transformations including MaxPool ops.
        dataset_kwargs (dict): Arguments for DatasetGenerator.
        device (Union[str, torch.device], optional): Device to train on. Defaults to 'cpu'.
        epochs (int, optional): Number of epochs per training. Defaults to 50.
        early_stopping (int, optional): Early stopping patience. Defaults to 8.
        lr (float, optional): Learning rate. Defaults to 0.001.
        optimizer (str, optional): Optimizer name. Defaults to 'Adam'.
        activation (str, optional): Activation function name ('Tanh' or 'ReLU'). Defaults to 'Tanh'.
        network_depth (int, optional): Base or max depth for the FC network. Defaults to 120.
        depth_step (int, optional): If nonzero, steps through depths from [depth_step .. network_depth].
            Otherwise uses `network_depth` only. Defaults to 0.
        variance_min (float, optional): Minimum weight variance. Defaults to 0.5.
        variance_max (float, optional): Maximum weight variance. Defaults to 6.0.
        variance_step (float, optional): Step size for variance range. Defaults to 0.5.
        comment (str, optional): Additional text to append to filename. Defaults to ''.

    Returns:
        Path: Path to the saved accuracy data file.
    """
    weight_variances: np.ndarray = np.arange(variance_min, variance_max + variance_step, variance_step)

    # Determine the set of depths
    if depth_step:
        depths = np.arange(depth_step, network_depth + depth_step, depth_step)
    else:
        depths = [network_depth]

    maxpool_sizes = [mp[-2].kernel_size if len(mp) > 1 else 1 for mp in maxpool_ops]
    network_features = [mp[-1].new_shape[0] for mp in maxpool_ops]
    network_widths = [min(feat, 1000) for feat in network_features]

    if comment:
        comment = f'_{comment}'
    file_name = 'mp-data-accuracy_{}_{}_{}_{}_{}{}.npy'.format(
        network_depth, activation,
        variance_min, variance_max, variance_step,
        comment.replace(' ', '-')
    )

    activation_fn = get_activation_fn(activation)

    # If multiple depths, we store an extra dimension; otherwise, just store the single depth
    if len(depths) > 1:
        # mp_accuracy shape: [#maxpool + 1, #variances + 1, #depths + 1, 2]
        mp_accuracy: np.ndarray = np.zeros((len(maxpool_ops) + 1, len(weight_variances) + 1, len(depths) + 1, 2))
        mp_accuracy[0, 1:, 0, 0] = weight_variances
        mp_accuracy[1:, 0, 0, 0] = maxpool_sizes
        mp_accuracy[0, 0, 1:, 0] = depths
    else:
        # If only a single depth, shape: [#maxpool + 1, #variances + 1, 2]
        mp_accuracy: np.ndarray = np.zeros((len(maxpool_ops) + 1, len(weight_variances) + 1, 2))
        mp_accuracy[0, 1:, 0] = weight_variances
        mp_accuracy[1:, 0, 0] = maxpool_sizes

    # Construct dataset for each MaxPool and variance
    for i, mp in enumerate(maxpool_ops):
        dataset_kwargs['transforms'] = mp
        dataset_generator = DatasetGenerator(**dataset_kwargs)
        train_data, val_data = dataset_generator.generate()

        with tqdm.tqdm(total=len(weight_variances) * len(depths)) as progress:
            progress.set_description(f'Generating data for MaxPool({maxpool_sizes[i]})')
            for j, var_w in enumerate(weight_variances):
                for k, depth in enumerate(depths):
                    net = FC(
                        in_features=network_features[i],
                        width=network_widths[i],
                        depth=depth,
                        variances=(var_w, 0.05),
                        activation=activation_fn
                    )

                    train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
                    val_loader = DataLoader(val_data, batch_size=2048, shuffle=False)

                    # Train and record best validation accuracy
                    history = train_model(
                        net,
                        train_loader,
                        val_loader,
                        n_epochs=epochs,
                        lr=lr,
                        device=device,
                        optimizer=optimizer,
                        early_stopping=early_stopping,
                        model_save_path='',
                        verbose=False
                    )

                    max_accuracy = max(history['val_acc'])
                    max_accuracy_epochs = history['val_acc'].index(max_accuracy) + 1

                    # Store in the array
                    if len(depths) > 1:
                        mp_accuracy[i + 1, j + 1, k + 1, 0] = max_accuracy
                        mp_accuracy[i + 1, j + 1, k + 1, 1] = max_accuracy_epochs
                    else:
                        mp_accuracy[i + 1, j + 1, 0] = max_accuracy
                        mp_accuracy[i + 1, j + 1, 1] = max_accuracy_epochs

                    # Save partial results
                    with open(output_path / file_name, 'wb') as f:
                        np.save(f, mp_accuracy)

                    progress.update()

    return output_path / file_name


def create_axes_grid(
    n_rows: int,
    n_cols: int,
    wspace: float = 0.0,
    hspace: float = 0.0,
    aspect: float = 1.0,
    size: float = 1.0
) -> Tuple[plt.Figure, np.ndarray]:
    """Create a grid of subplots with customizable spacing and sizing.

    Args:
        n_rows (int): Number of rows in the grid.
        n_cols (int): Number of columns in the grid.
        wspace (float, optional): The horizontal spacing between subplots. Defaults to 0.0.
        hspace (float, optional): The vertical spacing between subplots. Defaults to 0.0.
        aspect (float, optional): Aspect ratio for each subplot. Defaults to 1.0.
        size (float, optional): Base size scalar for the figure. Defaults to 1.0.

    Returns:
        Tuple[plt.Figure, np.ndarray]: The created Figure and array of Axes.
    """
    gridspec_kw = dict(
        wspace=wspace,
        hspace=hspace,
        top=1. - 0.5 / (n_rows + 1),
        bottom=0.5 / (n_rows + 1),
        left=0.5 / (n_cols + 1),
        right=1 - 0.5 / (n_cols + 1)
    )

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(size * (n_cols + 1), aspect * size * (n_rows + 1)),
        gridspec_kw=gridspec_kw
    )

    return fig, axes


def get_ticks(values: List[float], ticks: Union[int, tuple] = 6) -> Tuple[List[int], List[float]]:
    """Compute which ticks and labels to display on an axis.

    Args:
        values (List[float]): The list of values from which ticks will be derived.
        ticks (Union[int, tuple], optional): Number of ticks or a tuple of values. Defaults to 6.

    Returns:
        Tuple[List[int], List[float]]: Positions (indices) and corresponding labels.
    """
    values = np.array(values).round(1).tolist()

    if isinstance(ticks, int):
        step = len(values) // ticks if ticks != 0 else 1
        positions = list(range(0, len(values), step))
        labels = [values[i] for i in positions]
    elif isinstance(ticks, tuple):
        positions = [values.index(i) for i in ticks if i in values]
        labels = [i for i in ticks if i in values]
    else:
        positions = list(range(0, len(values)))
        labels = [values[i] for i in positions]

    return positions, labels


def heatmap_display_range(
    data: np.ndarray,
    display_range: Optional[Tuple[Optional[float], Optional[float]]]
) -> Tuple[float, float]:
    """Determine the color scale (vmin, vmax) for a heatmap.

    Args:
        data (np.ndarray): The 2D array to be displayed in the heatmap.
        display_range (Optional[Tuple[Optional[float], Optional[float]]]): A tuple (min, max).
            If None, automatically compute from data. If any value in the tuple is None,
            it is automatically computed from data.

    Returns:
        Tuple[float, float]: (v_min, v_max) values for the heatmap color range.
    """
    if display_range is None:
        v_min = np.nanmin(data[data != -np.inf])
        v_max = np.nanmax(data[data != np.inf])
    else:
        v_min, v_max = display_range
        if v_min is None:
            v_min = np.nanmin(data[data != -np.inf])
        if v_max is None:
            v_max = np.nanmax(data[data != np.inf])
    return v_min, v_max


def add_colorbar(
    im: plt.Axes,
    axes: Union[np.ndarray, plt.Axes],
    width: float = 0.025,
    pad: float = 0.05,
    **kwargs
) -> plt.colorbar:
    """Add a colorbar to the figure next to the provided Axes.

    Args:
        im (plt.Axes): The AxesImage or QuadMesh object for which the colorbar is created.
        axes (Union[np.ndarray, plt.Axes]): The axes to align the colorbar with.
        width (float, optional): The width of the colorbar axis. Defaults to 0.025.
        pad (float, optional): The gap between the axes and colorbar. Defaults to 0.05.

    Returns:
        plt.colorbar: The created colorbar object.
    """
    if isinstance(axes, np.ndarray):
        # Find bounding box for last Axes object
        if axes.ndim == 1:
            l, b, w, h = axes[-1].get_position().bounds
        else:
            l = axes[-1, -1].get_position().x0
            b = axes[-1, -1].get_position().y0
            w = axes[0, -1].get_position().x1 - axes[0, -1].get_position().x0
            h = axes[0, -1].get_position().y1 - axes[-1, -1].get_position().y0
    else:
        l, b, w, h = axes.get_position().bounds

    width = width or 0.1 * w
    pad = pad or width
    pos = [l + w + pad, b, width, h]
    fig = im.axes.figure
    cax = fig.add_axes(pos)
    return fig.colorbar(im, cax=cax, **kwargs)


def create_heatmap(
    data: np.ndarray,
    x_values: List[float],
    y_values: List[float],
    display_range: Optional[Tuple[Optional[float], Optional[float]]],
    title: str,
    x_label: str,
    y_label: str,
    z_label: str,
    data_labels: Optional[np.ndarray],
    x_ticks: Union[int, tuple],
    y_ticks: Union[int, tuple],
    cbar_ticks: Union[None, tuple] = None,
    aspect: str = 'auto'
) -> Tuple[plt.Figure, plt.Axes]:
    """Create a heatmap with optional data annotations and colorbar.

    Args:
        data (np.ndarray): The 2D array to display.
        x_values (List[float]): Labels or values for the x-axis.
        y_values (List[float]): Labels or values for the y-axis.
        display_range (Optional[Tuple[Optional[float], Optional[float]]]): Range for color scale.
        title (str): Plot title.
        x_label (str): X-axis label.
        y_label (str): Y-axis label.
        z_label (str): Label for the colorbar.
        data_labels (Optional[np.ndarray]): If provided, displays these labels over the heatmap cells.
        x_ticks (Union[int, tuple]): Tick specification for the x-axis.
        y_ticks (Union[int, tuple]): Tick specification for the y-axis.
        aspect (str, optional): Aspect ratio for imshow. Defaults to 'auto'.

    Returns:
        Tuple[plt.Figure, plt.Axes]: The created figure and axes objects.
    """
    v_min, v_max = heatmap_display_range(data, display_range)
    fig, ax = plt.subplots()

    # Display the data as an image
    im = ax.imshow(
        data,
        aspect=aspect,
        origin='lower',
        cmap='Spectral',
        vmin=v_min,
        vmax=v_max
    )

    # Add the colorbar on the right side
    cbar = add_colorbar(im, ax)
    cbar.set_label(z_label, rotation=90)
    if cbar_ticks is not None:
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels(cbar_ticks)

    # Title and axis labels
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Set ticks and labels
    ax.set_xticks(*get_ticks(x_values, x_ticks))
    ax.set_yticks(*get_ticks(y_values, y_ticks))

    # Optionally annotate each cell
    if data_labels is not None:
        for i in range(len(y_values)):
            for j in range(len(x_values)):
                ax.text(
                    j,
                    i,
                    data_labels[i, j],
                    ha='center',
                    va='center',
                    color='w',
                    size=5
                )

    plt.grid(False)
    plt.show()
    return fig, ax


def load_entropy_data(
    file_path: Union[str, Path],
    save_figure: bool = False,
    display_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
    x_ticks: Union[int, tuple] = 6,
    y_ticks: Union[int, tuple] = 6,
    log: bool = False
) -> None:
    """Load and optionally plot a heatmap of saved entropy data.

    Args:
        file_path (Union[str, Path]): Path to the .npy file containing entropy data.
        save_figure (bool, optional): Whether to save the figure as a .png file. Defaults to False.
        display_range (Optional[Tuple[Optional[float], Optional[float]]], optional):
            Color range for heatmap. Defaults to None.
        x_ticks (Union[int, tuple], optional): Tick specification for x-axis. Defaults to 6.
        y_ticks (Union[int, tuple], optional): Tick specification for y-axis. Defaults to 6.
        log (bool, optional): Whether to apply log transform to the data (excluding the first row). Defaults to False.
    """
    entropy_data: np.ndarray = np.load(file_path)
    # Transpose so variances are along x and depth is along y
    entropy_data = entropy_data.transpose()

    variances = entropy_data[0]
    if not log:
        # If not taking log, just remove the first row (variances)
        entropy_data = np.delete(entropy_data, 0, 0)
    else:
        # If taking log, remove the first row then log-transform the rest
        entropy_data = np.log(np.delete(entropy_data, 0, 0))

    file_name = Path(file_path).stem
    entropy_name, features, width, depth, activation = file_name.split('_')[:5]
    entropy_name = entropy_name.replace('-', ' ').capitalize()

    # Last component might be a comment
    comment = file_name.split('_')[-1]
    if is_number(comment):
        comment = ''
    else:
        comment = f'\n({comment.replace("-", " ")})'

    depths = list(range(1, len(entropy_data) + 1))

    # Generate heatmap
    fig, ax = create_heatmap(
        entropy_data,
        variances.tolist(),
        depths,
        display_range,
        f'input: {features[1:]}, width: {width[1:]}, activation: {activation}{comment}',
        'Variance $\\sigma_w^2$',
        'Depth',
        entropy_name,
        None,
        x_ticks,
        y_ticks
    )

    # Save figure if requested
    if save_figure:
        figure_path = file_path.parent / f'{file_path.stem}.png'
        fig.savefig(figure_path, dpi=300)


def load_accuracy_data(
    file_path: Union[str, Path],
    save_figure: bool = False,
    display_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
    show_epochs: bool = False,
    x_ticks: Union[int, tuple] = 6,
    y_ticks: Union[int, tuple] = 6
) -> None:
    """Load and optionally plot a heatmap of saved accuracy data.

    The data is assumed to be in a shape consistent with how `generate_accuracy_data`
    produces it.

    Args:
        file_path (Union[str, Path]): Path to the .npy file containing accuracy data.
        save_figure (bool, optional): Whether to save the generated figure. Defaults to False.
        display_range (Optional[Tuple[Optional[float], Optional[float]]], optional):
            Color range for the heatmap. Defaults to None.
        show_epochs (bool, optional): If True, show the epoch count in each cell. Defaults to False.
        x_ticks (Union[int, tuple], optional): Tick specification for x-axis. Defaults to 6.
        y_ticks (Union[int, tuple], optional): Tick specification for y-axis. Defaults to 6.
    """
    accuracy_data = np.load(file_path)  # type: np.ndarray
    accuracy_data = accuracy_data.transpose()

    # If dimension is 3, the second dimension likely holds [accuracy, epochs]
    if accuracy_data.ndim == 3:
        accuracies, epochs = accuracy_data
        if show_epochs:
            epochs = epochs.astype(int)
            epochs = np.delete(epochs, 0, 0)
        else:
            epochs = None
    else:
        accuracies = accuracy_data
        epochs = None

    # The first row is variance; remove it
    variances = accuracies[0]
    accuracies = np.delete(accuracies, 0, 0)

    file_name = Path(file_path).stem
    features, width, step, activation, optimizer = file_name.split('_')[1:6]
    step = int(step[1:])
    # Depth is implied by row count
    depths = list(range(step, (len(accuracies) + 1) * step, step))

    suffix = file_name.split('_')[-1]
    if is_number(suffix):
        suffix = ''
    else:
        suffix = f'\n({suffix.replace("-", " ")})'

    fig, ax = create_heatmap(
        accuracies,
        variances.tolist(),
        depths,
        display_range,
        f'input: {features[1:]}, width: {width[1:]}, activation: {activation}, '
        f'optimizer: {optimizer}{suffix}',
        'Variance $\\sigma_w^2$',
        'Depth',
        'Accuracy',
        epochs,
        x_ticks,
        y_ticks
    )

    if save_figure:
        figure_path = file_path.parent / f'{file_path.stem}.png'
        fig.savefig(figure_path, dpi=300)


def load_maxpool_data(
    file_path: Union[str, Path],
    save_figure: bool = False,
    display_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
    x_ticks: Union[int, tuple] = 6,
    y_ticks: Union[int, tuple] = 6
) -> None:
    """Load and optionally plot maxpool-based differential or Shannon entropy data.

    Args:
        file_path (Union[str, Path]): Path to the .npy file containing the data.
        save_figure (bool, optional): Whether to save the figure. Defaults to False.
        display_range (Optional[Tuple[Optional[float], Optional[float]]], optional):
            Color range for heatmap. Defaults to None.
        x_ticks (Union[int, tuple], optional): Tick specification for x-axis. Defaults to 6.
        y_ticks (Union[int, tuple], optional): Tick specification for y-axis. Defaults to 6.
    """
    maxpool_data: np.ndarray = np.load(file_path)

    # The first row and column might hold the variance and maxpool sizes
    variances = maxpool_data[0, 1:]
    mp_sizes = tuple(maxpool_data[1:, 0].astype(int).tolist())

    # Remove the first row and column so that only the data remains
    maxpool_data = np.delete(maxpool_data, 0, 0)
    maxpool_data = np.delete(maxpool_data, 0, 1)

    if isinstance(y_ticks, int) and y_ticks > len(mp_sizes):
        y_ticks = mp_sizes

    file_name = Path(file_path).stem
    # e.g., 'mp-data-differential-entropy_D120_Tanh_0.5_6.0_0.5_comment'
    entropy_name, depth, activation = file_name.split('_')[:3]
    entropy_name = ' '.join(entropy_name.split('-')[2:]).capitalize()

    comment = file_name.split('_')[-1]
    if is_number(comment):
        comment = ''
    else:
        comment = f'\n({comment.replace("-", " ")})'

    fig, ax = create_heatmap(
        maxpool_data,
        variances.tolist(),
        list(mp_sizes),
        display_range,
        f'depth: {depth[1:]}, activation: {activation}{comment}',
        'Variance $\\sigma_w^2$',
        'MaxPool Kernel Size',
        entropy_name,
        None,
        x_ticks,
        y_ticks
    )

    if save_figure:
        figure_path = file_path.parent / f'{file_path.stem}.png'
        fig.savefig(figure_path, dpi=300)


def load_maxpool_accuracy_data(
    file_path: Union[str, Path],
    save_figure: bool = False,
    display_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
    show_epochs: bool = False,
    x_ticks: Union[int, tuple] = 6,
    y_ticks: Union[int, tuple] = 6
) -> None:
    """Load and optionally plot maxpool-based accuracy data.

    The data is assumed to have shape consistent with `generate_maxpool_accuracy_data`.

    Args:
        file_path (Union[str, Path]): Path to the .npy file containing maxpool accuracy data.
        save_figure (bool, optional): Whether to save the figure as a PNG. Defaults to False.
        display_range (Optional[Tuple[Optional[float], Optional[float]]], optional):
            Color range for the heatmap. Defaults to None.
        show_epochs (bool, optional): If True, display epoch numbers in the heatmap cells. Defaults to False.
        x_ticks (Union[int, tuple], optional): Tick specification for x-axis. Defaults to 6.
        y_ticks (Union[int, tuple], optional): Tick specification for y-axis. Defaults to 6.
    """
    maxpool_data: np.ndarray = np.load(file_path)

    # The first row contains the variances, the first column the maxpool sizes
    variances = maxpool_data[0, 1:, 0]
    mp_sizes = tuple(maxpool_data[1:, 0, 0].astype(int).tolist())

    # Remove the first row and column
    maxpool_data = np.delete(maxpool_data, 0, 0)
    maxpool_data = np.delete(maxpool_data, 0, 1)

    # The last dimension is [accuracy, epoch]
    accuracies, epochs = np.moveaxis(maxpool_data, 2, 0)
    epochs = epochs.astype(int) if show_epochs else None

    if isinstance(y_ticks, int) and y_ticks > len(mp_sizes):
        y_ticks = mp_sizes

    file_name = Path(file_path).stem
    depth, activation = file_name.split('_')[1:3]
    comment = file_name.split('_')[-1]
    if is_number(comment):
        comment = ''
    else:
        comment = f'\n({comment.replace("-", " ")})'

    fig, ax = create_heatmap(
        accuracies,
        variances.tolist(),
        list(mp_sizes),
        display_range,
        f'depth: {depth}, activation: {activation}{comment}',
        'Variance $\\sigma_w^2$',
        'MaxPool Kernel Size',
        'Accuracy',
        epochs,
        x_ticks,
        y_ticks
    )

    if save_figure:
        figure_path = file_path.parent / f'{file_path.stem}.png'
        fig.savefig(figure_path, dpi=300)