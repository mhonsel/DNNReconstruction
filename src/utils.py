import copy
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import tqdm
import smtplib

from email.mime.text import MIMEText
from functools import wraps
from pathlib import Path
from torch.utils.data import DataLoader

from sklearn.metrics import roc_curve, roc_auc_score
from typing import Union

from src.dataset import DatasetGenerator
from src.entropies import diff_entropy, diff_entropy_clamped, shannon_entropy
from src.network import FC
from src.reconstruction_network import ReconstructionNetwork


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def shannon_entropy_from_array(arr: np.ndarray, bins, normalize=True) -> float:
    p = np.histogram(arr.ravel(), bins=bins)[0] / arr.size
    #w = (arr.max() - arr.min()) * 1.0 / bins
    p = p[p > 0]
    #H = -np.sum(p * np.log(p / w))  # https://xuk.ai/blog/estimate-entropy-wrong.html
    H = -np.sum(p * np.log(p))
    if normalize:
        return H / arr.size
    else:
        return H


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} '
              f'Took {total_time:.4f} seconds')
        return result

    return timeit_wrapper


def parse_model_path(file_path):
    file_name = Path(file_path).stem
    in_features, width, depth = \
        [int(item[1:]) for item in file_name.split('_')[1:4]]
    activation = file_name.split('_')[4]
    if activation == 'Tanh':
        activation_fn = nn.Tanh
    elif activation == 'ReLU':
        activation_fn = nn.ReLU
    else:
        raise ValueError('Supported activation functions are Tanh and ReLU.')

    return in_features, width, depth, activation_fn


def load_model(file_path, model=None, device='cpu'):
    if model is None:
        in_features, width, depth, activation = parse_model_path(file_path)
        model = FC(in_features=in_features, width=width, depth=depth,
                   variances=None, activation=activation)
        model.to(device)
        model.eval()

    model.load_state_dict(torch.load(file_path, weights_only=True))

    return model


def load_reconstruction_network(file_path, re_net=None, device='cpu'):
    if re_net is None:
        in_features, width, depth, activation = parse_model_path(file_path)
        model = FC(in_features=in_features, width=width, depth=depth,
                   variances=None, activation=activation)
        model.to(device)
        model.eval()
        re_net = ReconstructionNetwork(model, device=device, lr=1e-3)

    re_net.load(file_path)

    return re_net.net, re_net


def train_model(model, train_loader, val_loader,
                n_epochs=10, lr=0.001, l2_penalty=0,
                device: Union[str, torch.device] = 'cpu',
                optimizer='Adam',
                early_stopping=0, model_save_path='', verbose=True):
    model.to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=l2_penalty)
    elif optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=l2_penalty)

    stopping_counter = 0
    model_state_dict = None
    history = {
        'train_acc': [],
        'train_loss': [],
        'val_acc': [],
        'val_loss': []
    }

    for epoch in range(n_epochs):
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        model.train()
        with tqdm.tqdm(
                total=len(train_loader), disable=(not verbose)) as progress:
            progress.set_description(f'Training epoch {epoch + 1}/{n_epochs}')
            for X, y in train_loader:
                X = X.to(device)
                y = y.to(device).squeeze()
                # forward pass
                y_pred = model(X).squeeze()
                loss = loss_fn(y_pred, y.float())
                train_loss += loss.item()
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # accuracy
                # acc = (y_pred.round() == y).float().mean().item()
                acc = ((y_pred > 0) == y).float().mean().item()
                train_acc += acc
                # update progress bar
                progress.set_postfix_str(f'loss={loss.item():.3f}, acc={acc:.3f}')
                progress.update()

        model.eval()
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                y = y.to(device)
                y_pred = model(X)
                loss = loss_fn(y_pred, y.float()).item()
                val_loss += loss
                # acc = (y_pred.round() == y).float().mean().item()
                acc = ((y_pred > 0) == y).float().mean().item()
                val_acc += acc

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        train_acc /= len(train_loader)
        val_acc /= len(val_loader)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if verbose:
            print(
                f'Train loss/accuracy: {train_loss:.3f}/{train_acc:.3f}'
                f'\tValidation loss/accuracy: {val_loss:.3f}/{val_acc:.3f}'
            )

        if val_acc == max(history['val_acc']):
            model_state_dict = copy.deepcopy(model.state_dict())
            stopping_counter = 0

            if model_save_path:
                torch.save(model_state_dict, model_save_path)
                if verbose:
                    print('Saved model state.')

        elif early_stopping:
            if stopping_counter < early_stopping:
                stopping_counter += 1
            if stopping_counter == early_stopping:
                model.load_state_dict(model_state_dict)
                if verbose:
                    print('Loading best params, exiting training loop.')
                break

    return history


def plot_roc(model, data, labels):
    # get predictions and calculate probabilities
    model.eval()
    with torch.no_grad():
        logits = model(data)

    sigmoid = nn.Sigmoid()
    probabilities = sigmoid(logits)

    # calculate roc curve
    fpr, tpr, _ = roc_curve(labels.to('cpu'), probabilities.to('cpu'))

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

    # calculate roc auc
    auc = roc_auc_score(labels.to('cpu'), probabilities.to('cpu'))
    print(f'AUC: {auc}')


def get_activation_fn(activation):
    if activation == 'Tanh':
        return nn.Tanh
    elif activation == 'ReLU':
        return nn.ReLU
    else:
        raise ValueError('Supported activation functions are Tanh and ReLU.')


def generate_entropy_data(
        output_path,
        train_loader,
        test_loader,
        device='cpu',
        epochs=2,
        network_input_width=7500,
        network_width=256,
        network_depth=50,
        activation='Tanh',
        variance_min=0.5,
        variance_max=5.0,
        variance_step=0.5,
        entropy='differential',
        entropy_fn_kwargs=None,
        comment='',
):
    weight_variances: np.ndarray = np.arange(
        variance_min, variance_max + variance_step, variance_step)

    if comment:
        comment = f'_{comment}'
    file_name = '{}-entropy_F{}_W{}_D{}_{}_{}_{}_{}{}.npy'.format(
        entropy,
        network_input_width, network_width, network_depth, activation,
        variance_min, variance_max, variance_step,
        comment.replace(' ', '-')
    )

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

    entropy_data: np.ndarray = np.zeros((len(weight_variances), network_depth - 1))
    with tqdm.tqdm(total=len(weight_variances)) as progress:
        progress.set_description('Generating entropy data')
        for i, var_w in enumerate(weight_variances):
            net = FC(in_features=network_input_width,
                     width=network_width, depth=network_depth,
                     variances=(var_w, 0.05), activation=activation_fn)
            re_net = ReconstructionNetwork(net, device, lr=5e-4)
            #train_loader = DataLoader(train_data, batch_size=64, shuffle=True)  # I think this was left in by mistake?!
            _ = re_net.train_network(train_loader, epochs=epochs, verbose=False)

            X, _ = next(iter(test_loader))
            cascades = re_net.cascade(X)

            entropies = [entropy_fn(c, **entropy_fn_kwargs).tolist() for c in cascades[1:-1]]
            entropies.insert(0, var_w)
            entropy_data[i] = entropies
            with open(output_path / file_name, 'wb') as f:
                np.save(f, entropy_data)

            progress.update()

    return output_path / file_name


def generate_accuracy_data(
        output_path,
        train_loader,
        val_loader,
        device='cpu',
        epochs=50,
        early_stopping=8,
        lr=0.001,
        optimizer='Adam',
        network_input_width=7500,
        network_width=256,
        activation='Tanh',
        max_depth=50,
        depth_step=5,
        variance_min=0.5,
        variance_max=5.0,
        variance_step=0.5,
        comment='',
):
    weight_variances = np.arange(
        variance_min, variance_max + variance_step, variance_step)  # type: np.ndarray

    if comment:
        comment = f'_{comment}'
    file_name = 'accuracy_F{}_W{}_S{}_{}_{}_{}_{}_{}{}.npy'.format(
        network_input_width, network_width, depth_step, activation, optimizer,
        variance_min, variance_max, variance_step,
        comment.replace(' ', '-')
    )

    activation_fn = get_activation_fn(activation)

    depths = range(depth_step, max_depth + 1, depth_step)
    accuracy_data = np.zeros((len(weight_variances), len(depths) + 1, 2))  # type: np.ndarray

    with tqdm.tqdm(total=len(weight_variances) * len(depths)) as progress:
        progress.set_description('Generating accuracy data')
        for i, var_w in enumerate(weight_variances):
            accuracy_data[i][0] = var_w
            for j, depth in enumerate(depths):
                net = FC(
                    in_features=network_input_width, width=network_width,
                    depth=depth, variances=(var_w, 0.05), activation=activation_fn
                )

                history = train_model(
                    net, train_loader, val_loader, n_epochs=epochs, lr=lr,
                    device=device, optimizer=optimizer,
                    early_stopping=early_stopping,
                    model_save_path='', verbose=False
                )

                max_accuracy = max(history['val_acc'])
                max_accuracy_epochs = history['val_acc'].index(max_accuracy) + 1
                accuracy_data[i][j + 1][0] = max_accuracy
                accuracy_data[i][j + 1][1] = max_accuracy_epochs

                with open(output_path / file_name, 'wb') as f:
                    np.save(f, accuracy_data)

                progress.update()

    return output_path / file_name


# old version for single depth
def generate_maxpool_data(
        output_path,
        maxpool_ops,
        dataset_kwargs,
        device='cpu',
        epochs=3,
        network_depth=120,
        activation='Tanh',
        variance_min=0.5,
        variance_max=6.0,
        variance_step=0.5,
        entropy_bs=1000,
        compute_shannon_entropy=True,
        comment='',
):
    weight_variances: np.ndarray = np.arange(
        variance_min, variance_max + variance_step, variance_step)

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

    mp_diff_entropy: np.ndarray = np.zeros((len(maxpool_ops) + 1, len(weight_variances) + 1))
    mp_diff_entropy[0, 1:] = weight_variances
    mp_diff_entropy[1:, 0] = maxpool_sizes
    if compute_shannon_entropy:
        mp_shannon_entropy = np.copy(mp_diff_entropy)

    for i, mp in enumerate(maxpool_ops):
        # load data with MaxPool transform
        dataset_kwargs['transforms'] = mp
        dataset_generator = DatasetGenerator(**dataset_kwargs)
        train_data, val_data = dataset_generator.generate()

        with tqdm.tqdm(total=len(weight_variances)) as progress:
            progress.set_description(f'Generating data for MaxPool({maxpool_sizes[i]})')
            for j, var_w in enumerate(weight_variances):
                net = FC(in_features=network_features[i],
                         width=network_widths[i], depth=network_depth,
                         variances=(var_w, 0.05), activation=activation_fn)
                re_net = ReconstructionNetwork(net, device, lr=5e-4)
                train_loader = DataLoader(train_data, batch_size=64,
                                          shuffle=True)
                _ = re_net.train_network(train_loader, epochs=epochs, verbose=False)

                # calculate reconstruction entropy at last layer
                entropy_loader = DataLoader(val_data,
                                            batch_size=entropy_bs,
                                            shuffle=False)
                xb, _ = next(iter(entropy_loader))
                cascades = re_net.cascade(xb)
                mp_diff_entropy[i + 1, j + 1] = diff_entropy_clamped(cascades[-1]).item()
                if compute_shannon_entropy:
                    mp_shannon_entropy[i + 1, j + 1] = shannon_entropy(cascades[-1], entropy_bs).item()

                with open(output_path / file_name_diff, 'wb') as f:
                    np.save(f, mp_diff_entropy)
                if compute_shannon_entropy:
                    with open(output_path / file_name_shannon, 'wb') as f:
                        np.save(f, mp_shannon_entropy)

                progress.update()

    if compute_shannon_entropy:
        return output_path / file_name_diff, output_path / file_name_shannon
    else:
        return output_path / file_name_diff


def generate_maxpool_entropy_data(
        output_path,
        maxpool_ops,
        dataset_kwargs,
        device='cpu',
        epochs=3,
        network_depth=120,
        activation='Tanh',
        variance_min=0.5,
        variance_max=6.0,
        variance_step=0.5,
        entropy_bs=1000,
        comment='',
):
    weight_variances: np.ndarray = np.arange(
        variance_min, variance_max + variance_step, variance_step)

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

    mp_entropy: np.ndarray = np.zeros((len(maxpool_ops) + 1, len(weight_variances) + 1, network_depth - 1))
    mp_entropy[0, 1:, 0] = weight_variances
    mp_entropy[1:, 0, 0] = maxpool_sizes
    mp_entropy[0, 0, 1:] = np.arange(network_depth - 2)

    for i, mp in enumerate(maxpool_ops):
        # load data with MaxPool transform
        dataset_kwargs['transforms'] = mp
        dataset_generator = DatasetGenerator(**dataset_kwargs)
        train_data, val_data = dataset_generator.generate()

        with tqdm.tqdm(total=len(weight_variances)) as progress:
            progress.set_description(f'Generating data for MaxPool({maxpool_sizes[i]})')
            for j, var_w in enumerate(weight_variances):
                net = FC(in_features=network_features[i],
                         width=network_widths[i], depth=network_depth,
                         variances=(var_w, 0.05), activation=activation_fn)
                re_net = ReconstructionNetwork(net, device, lr=5e-4)
                train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
                _ = re_net.train_network(train_loader, epochs=epochs, verbose=False)

                # calculate reconstruction entropy
                entropy_loader = DataLoader(val_data, batch_size=entropy_bs, shuffle=False)
                xb, _ = next(iter(entropy_loader))
                cascades = re_net.cascade(xb)
                entropies = [diff_entropy(c).tolist() for c in cascades[1:-1]]
                mp_entropy[i + 1, j + 1, 1:] = np.array(entropies)

                with open(output_path / file_name, 'wb') as f:
                    np.save(f, mp_entropy)

                progress.update()

    return output_path / file_name


def generate_maxpool_accuracy_data(
        output_path,
        maxpool_ops,
        dataset_kwargs,
        device='cpu',
        epochs=50,
        early_stopping=8,
        lr=0.001,
        optimizer='Adam',
        activation='Tanh',
        network_depth=120,
        depth_step=0,
        variance_min=0.5,
        variance_max=6.0,
        variance_step=0.5,
        comment='',
):
    weight_variances: np.ndarray = np.arange(
        variance_min, variance_max + variance_step, variance_step)
    depths = np.arange(depth_step, network_depth + depth_step, depth_step)\
        if depth_step else [network_depth]

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

    if len(depths) > 1:
        mp_accuracy: np.ndarray = np.zeros((len(maxpool_ops) + 1, len(weight_variances) + 1, len(depths) + 1, 2))
        mp_accuracy[0, 1:, 0, 0] = weight_variances
        mp_accuracy[1:, 0, 0, 0] = maxpool_sizes
        mp_accuracy[0, 0, 1:, 0] = depths
    else:
        mp_accuracy: np.ndarray = np.zeros((len(maxpool_ops) + 1, len(weight_variances) + 1, 2))
        mp_accuracy[0, 1:, 0] = weight_variances
        mp_accuracy[1:, 0, 0] = maxpool_sizes

    for i, mp in enumerate(maxpool_ops):
        # load data with MaxPool transform
        dataset_kwargs['transforms'] = mp
        dataset_generator = DatasetGenerator(**dataset_kwargs)
        train_data, val_data = dataset_generator.generate()

        with tqdm.tqdm(total=len(weight_variances) * len(depths)) as progress:
            progress.set_description(f'Generating data for MaxPool({maxpool_sizes[i]})')
            for j, var_w in enumerate(weight_variances):
                for k, depth in enumerate(depths):
                    net = FC(in_features=network_features[i],
                             width=network_widths[i], depth=depth,
                             variances=(var_w, 0.05), activation=activation_fn)

                    train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
                    val_loader = DataLoader(val_data, batch_size=2048, shuffle=False)

                    history = train_model(
                        net, train_loader, val_loader, n_epochs=epochs, lr=lr,
                        device=device, optimizer=optimizer,
                        early_stopping=early_stopping,
                        model_save_path='', verbose=False
                    )

                    max_accuracy = max(history['val_acc'])
                    max_accuracy_epochs = history['val_acc'].index(max_accuracy) + 1
                    if len(depths) > 1:
                        mp_accuracy[i + 1, j + 1, k + 1, 0] = max_accuracy
                        mp_accuracy[i + 1, j + 1, k + 1, 1] = max_accuracy_epochs
                    else:
                        mp_accuracy[i + 1, j + 1, 0] = max_accuracy
                        mp_accuracy[i + 1, j + 1, 1] = max_accuracy_epochs

                    with open(output_path / file_name, 'wb') as f:
                        np.save(f, mp_accuracy)

                    progress.update()

    return output_path / file_name


def create_axes_grid(n_rows, n_cols, wspace=0.0, hspace=0.0, aspect=1.0, size=1.0):
    gridspec_kw = dict(
        wspace=wspace, hspace=hspace,
        top=1. - 0.5 / (n_rows + 1),
        bottom=0.5 / (n_rows + 1),
        left=0.5 / (n_cols + 1),
        right=1 - 0.5 / (n_cols + 1)
    )

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(size * (n_cols + 1), aspect * size * (n_rows + 1)),
        gridspec_kw=gridspec_kw
    )

    return fig, axes


def get_ticks(values, ticks: Union[int, tuple] = 6):
    values = np.array(values).round(1).tolist()

    if type(ticks) == int:
        step = len(values) // ticks
        positions = range(0, len(values), step)
        labels = values[:: step]
    elif type(ticks) == tuple:
        positions = [values.index(i) for i in ticks if i in values]
        labels = [i for i in ticks if i in values]
    else:
        positions = range(0, len(values))
        labels = positions

    return positions, labels


def heatmap_display_range(data, display_range):
    if display_range is None:
        v_min = np.nanmin(data[data != -np.inf])
        v_max = np.nanmax(data[data != np.inf])
    else:
        v_min, v_max = display_range
        v_min = v_min if v_min else np.nanmin(data[data != -np.inf])
        v_max = v_max if v_max else np.nanmax(data[data != np.inf])

    return v_min, v_max


def add_colorbar(im, axes, width=0.025, pad=0.05, **kwargs):
    if isinstance(axes, np.ndarray):
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


def create_heatmap(data, x_values, y_values, display_range,
                   title, x_label, y_label, z_label, data_labels, x_ticks, y_ticks, aspect='auto'):
    v_min, v_max = heatmap_display_range(data, display_range)
    fig, ax = plt.subplots()
    im = ax.imshow(
        data, aspect=aspect, origin='lower', cmap='Spectral',
        vmin=v_min, vmax=v_max
    )

    #cbar = ax.figure.colorbar(im, ax=ax)
    #cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar = add_colorbar(im, ax)
    cbar.set_label(z_label, rotation=90)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel(x_label)
    ax.set_xticks(*get_ticks(x_values, x_ticks))
    ax.set_ylabel(y_label)
    ax.set_yticks(*get_ticks(y_values, y_ticks))

    if data_labels is not None:
        for i in range(len(y_values)):
            for j in range(len(x_values)):
                _ = ax.text(j, i, data_labels[i, j],
                               ha='center', va='center', color='w', size=5)

    #plt.tick_params(left=True, bottom=True)

    plt.grid(False)
    plt.show()

    return fig, ax


def load_entropy_data(file_path, save_figure=False, display_range=None,
                      x_ticks=6, y_ticks=6, log=False):
    # load data
    entropy_data: np.ndarray = np.load(file_path)
    entropy_data = entropy_data.transpose()
    variances = entropy_data[0]
    if not log:
        entropy_data = np.delete(entropy_data, 0, 0)
    else:
        entropy_data = np.log(np.delete(entropy_data, 0, 0))

    # extract information from file name
    file_name = Path(file_path).stem
    entropy_name, features, width, depth, activation = file_name.split('_')[:5]
    entropy_name = entropy_name.replace('-', ' ').capitalize()
    comment = file_name.split('_')[-1]
    if is_number(comment):
        comment = ''
    else:
        comment = '\n({})'.format(comment.replace('-', ' '))

    depths = list(range(1, len(entropy_data) + 1))
    fig, ax = create_heatmap(
        entropy_data, variances, depths, display_range,
        f'input: {features[1:]}, width: {width[1:]}, '
        f'activation: {activation}{comment}',
        'Variance $\\sigma_w^2$',
        'Depth',
        entropy_name,
        None,
        x_ticks, y_ticks
    )

    if save_figure:
        figure_path = file_path.parent / f'{file_path.stem}.png'
        fig.savefig(figure_path, dpi=300)


def load_accuracy_data(file_path, save_figure=False, display_range=None,
                       show_epochs=False, x_ticks=6, y_ticks=6):
    # load data
    accuracy_data = np.load(file_path)  # type: np.ndarray
    accuracy_data = accuracy_data.transpose()

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

    variances = accuracies[0]
    accuracies = np.delete(accuracies, 0, 0)

    # extract information from file name
    file_name = Path(file_path).stem
    features, width, step, activation, optimizer = file_name.split('_')[1:6]
    step = int(step[1:])
    depths = list(range(step, (len(accuracies) + 1) * step, step))
    suffix = file_name.split('_')[-1]
    if is_number(suffix):
        suffix = ''
    else:
        suffix = '\n({})'.format(suffix.replace('-', ' '))

    fig, ax = create_heatmap(
        accuracies, variances, depths,
        display_range,
        f'input: {features[1:]}, width: {width[1:]}, activation: {activation}, '
        f'optimizer: {optimizer}{suffix}',
        'Variance $\\sigma_w^2$',
        'Depth',
        'Accuracy',
        epochs, x_ticks, y_ticks
    )

    if save_figure:
        figure_path = file_path.parent / f'{file_path.stem}.png'
        fig.savefig(figure_path, dpi=300)


def load_maxpool_data(file_path, save_figure=False, display_range=None,
                      x_ticks=6, y_ticks=6):
    # load data
    maxpool_data: np.ndarray = np.load(file_path)
    #maxpool_data = maxpool_data.transpose()
    variances = maxpool_data[0, 1:]
    mp_sizes = tuple(maxpool_data[1:, 0].astype(int).tolist())
    maxpool_data = np.delete(maxpool_data, 0, 0)
    maxpool_data = np.delete(maxpool_data, 0, 1)

    if type(y_ticks) is int:
        if y_ticks > len(mp_sizes):
            y_ticks = mp_sizes

    # extract information from file name
    file_name = Path(file_path).stem
    entropy_name, depth, activation = file_name.split('_')[:3]
    entropy_name = ' '.join(entropy_name.split('-')[2:]).capitalize()
    #entropy_name = entropy_name.replace('-', ' ').capitalize()
    comment = file_name.split('_')[-1]
    if is_number(comment):
        comment = ''
    else:
        comment = '\n({})'.format(comment.replace('-', ' '))

    fig, ax = create_heatmap(
        maxpool_data, variances, mp_sizes, display_range,
        f'depth: {depth[1:]}, activation: {activation}{comment}',
        'Variance $\\sigma_w^2$',
        'MaxPool Kernel Size',
        entropy_name,
        None,
        x_ticks, y_ticks
    )

    if save_figure:
        figure_path = file_path.parent / f'{file_path.stem}.png'
        fig.savefig(figure_path, dpi=300)


def load_maxpool_accuracy_data(file_path, save_figure=False, display_range=None,
                      show_epochs=False, x_ticks=6, y_ticks=6):
    # load data
    maxpool_data: np.ndarray = np.load(file_path)
    variances = maxpool_data[0, 1:, 0]
    mp_sizes = tuple(maxpool_data[1:, 0, 0].astype(int).tolist())
    maxpool_data = np.delete(maxpool_data, 0, 0)
    maxpool_data = np.delete(maxpool_data, 0, 1)

    accuracies, epochs = np.moveaxis(maxpool_data, 2, 0)
    epochs = epochs.astype(int) if show_epochs else None

    if type(y_ticks) is int:
        if y_ticks > len(mp_sizes):
            y_ticks = mp_sizes

    # extract information from file name
    file_name = Path(file_path).stem
    depth, activation = file_name.split('_')[1:3]
    comment = file_name.split('_')[-1]
    if is_number(comment):
        comment = ''
    else:
        comment = '\n({})'.format(comment.replace('-', ' '))

    fig, ax = create_heatmap(
        accuracies, variances, mp_sizes, display_range,
        f'depth: {depth}, activation: {activation}{comment}',
        'Variance $\\sigma_w^2$',
        'MaxPool Kernel Size',
        'Accuracy',
        epochs,
        x_ticks, y_ticks
    )

    if save_figure:
        figure_path = file_path.parent / f'{file_path.stem}.png'
        fig.savefig(figure_path, dpi=300)
