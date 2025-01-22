import torch
import numpy as np

# Constants and references:
#   - https://math.stackexchange.com/questions/1804805/how-is-the-entropy-of-the-normal-distribution-derived
#   - https://de.wikipedia.org/wiki/Differentielle_Entropie
NORMALFACTOR = 0.5 * np.log(2 * np.pi * np.e)


def diff_entropy(actis: torch.Tensor) -> torch.Tensor:
    """Calculate the differential entropy for a set of activations.

    This function computes the differential entropy by first normalizing the
    activations row-wise (each row sums to 1), then computing the standard deviation,
    and finally applying the analytical expression for the normal distribution's
    differential entropy.

    Args:
        actis (torch.Tensor): A 2D tensor of activations of shape [batch_size, num_activations].

    Returns:
        torch.Tensor: A scalar tensor representing the average differential entropy
            across all activations.
    """
    # Normalize each row so that it sums to 1
    actis /= torch.sum(actis, dim=1, keepdim=True)

    # Compute the standard deviation across the batch dimension
    std = torch.std(actis, dim=0)

    # Return the mean of the differential entropy formula for each activation
    return torch.mean(NORMALFACTOR + torch.log(std))


def diff_entropy_clamped(
    actis: torch.Tensor,
    normalization_fix: bool = False
) -> torch.Tensor:
    """Calculate the differential entropy for a set of activations, ignoring zero-std positions.

    Unlike `diff_entropy`, this function only computes the entropy for elements whose
    standard deviation is non-zero, effectively "clamping" out converged values.

    Args:
        actis (torch.Tensor): A 2D tensor of activations of shape [batch_size, num_activations].
        normalization_fix (bool, optional): If True, uses a different normalization
            strategy for the calculated entropy. Defaults to False.

    Returns:
        torch.Tensor: A scalar tensor representing the differential entropy.
    """
    # Normalize each row so that it sums to 1
    actis /= torch.sum(actis, dim=1, keepdim=True)

    # Compute the standard deviation across the batch dimension
    std = torch.std(actis, dim=0)

    # If `normalization_fix` is True, sum over the valid std positions and divide by len(std).
    # Otherwise, compute the mean over non-zero std positions.
    if normalization_fix:
        return torch.sum(NORMALFACTOR + torch.log(std[std > 0])) / len(std)
    else:
        return torch.mean(NORMALFACTOR + torch.log(std[std > 0]))


def shannon_entropy(actis: torch.Tensor, num_bins: int) -> torch.Tensor:
    """Calculate the Shannon entropy for a set of activations using histogram binning.

    This function bins the activations of each feature (column in `actis`) into
    `num_bins` bins (between -1 and 1) and computes the Shannon entropy of that
    distribution. The final result is the average Shannon entropy across all features.

    Args:
        actis (torch.Tensor): A 2D tensor of activations [batch_size, num_activations].
        num_bins (int): The number of bins to use for histogramming.

    Returns:
        torch.Tensor: A scalar tensor representing the average Shannon entropy across
            all activation features.
    """
    entropy = 0.0
    # Create linearly spaced bins from -1 to 1
    bins = torch.linspace(-1, 1, num_bins)

    # Compute the histogram and Shannon entropy for each activation
    for i in range(actis.shape[1]):
        # Get histogram values for the i-th activation dimension
        hist = torch.histogram(actis[:, i], bins=bins).hist
        # Keep only non-zero entries
        hist = hist[hist > 0]
        # Normalize to make probabilities
        hist /= torch.sum(hist)
        # Add the negative sum of p*log(p) to the total entropy
        entropy -= torch.sum(torch.log(hist) * hist)

    # Return the mean over the number of activation features
    return entropy / actis.shape[1]
