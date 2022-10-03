"""Evaluation metrics, where `p` are predictions and `y` are labels."""


import matplotlib.pyplot as plt
import numpy as np


def mae(p, y):
    """Mean absolute error."""
    return np.mean(np.abs(p - y))

def mse(p, y):
    """Mean squared error."""
    return np.mean((p - y)**2)

def mae_nonzeros(p, y):
    """Mean absolute error calculated only among elements with nonzero true values."""
    return np.mean(np.abs(p[y > 0] - y[y > 0]))

def min_error(p, y):
    """Error with lowest value (which may be a large negative number)."""
    return np.min(p - y)

def max_error(p, y):
    """Error with highest value."""
    return np.max(p - y)

def fraction_of_zeros_correct(p, y) -> float:
    """Fraction of all true 0 values for which a 0 was predicted. Similar to false negative."""
    zero_predictions = p[y == 0]
    if zero_predictions.size == 0:
        return np.nan
    else:
        return (zero_predictions == 0).sum() / zero_predictions.size

def fraction_of_zeros_incorrect(p, y) -> float:
    """Fraction of all true 0 values for which a nonzero value was predicted. Similar to false positive."""
    zero_predictions = p[y == 0]
    if zero_predictions.size == 0:
        return np.nan
    else:
        return (zero_predictions != 0).sum() / zero_predictions.size

def plot_error_by_label(p, y):
    """Plot the errors vs. true values sorted from lowest to highest. Intended to reveal which ranges of values produce inaccurate predictions."""
    p, y = p.flatten(), y.flatten()
    p, y = np.array(sorted(
        np.concatenate([p[:, None], y[:, None]], axis=1),
        key=lambda row: row[1]
    )).transpose()

    plt.figure()
    plt.plot(y, p-y, '.')
    plt.grid()
    plt.xlabel('True')
    plt.ylabel('Error')
    plt.title('Errors By True Values')
    plt.show()

def plot_adjacency(p, y):
    """Plot the predictions and labels side-by-side as grayscale images."""
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(p, cmap='gray')
    plt.title('Predicted')
    plt.subplot(1, 2, 2)
    plt.imshow(y, cmap='gray')
    plt.title('True')
    plt.show()

def evaluate(p, y) -> dict:
    results = {
        'MAE': mae(p, y),
        'MSE': mse(p, y),
        'MAE among nonzeros': mae_nonzeros(p, y),
        'Minimum error': min_error(p, y),
        'Maximum error': max_error(p, y),
        'Zeros correct': fraction_of_zeros_correct(p, y),
        'Zeros incorrect': fraction_of_zeros_incorrect(p, y),
    }
    
    return results