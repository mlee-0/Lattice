"""Evaluation metrics, where `p` are predictions and `y` are labels."""


from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np


def me(p, y):
    """Mean error."""
    return np.mean(p - y)

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

def plot_predicted_vs_true(p, y):
    """Plot the predictions vs. true values sorted from lowest to highest. Intended to reveal which ranges of values produce inaccurate predictions."""
    p, y = p.flatten(), y.flatten()
    p, y = np.array(sorted(
        np.concatenate([p[:, None], y[:, None]], axis=1),
        key=lambda row: row[1]
    )).transpose()

    plt.figure()
    plt.plot(y, p, '.')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.title(f'Predictions vs. True Values ({p.size} data)')
    plt.show()

def plot_error_by_edge_distance(p, y, locations_1: List[Tuple[int, int, int]], locations_2: List[Tuple[int, int, int]]):
    """Plot the errors vs. distances from the edges. Intended to reveal where in the 3D volume predictions are inaccurate."""

    p, y = p.flatten(), y.flatten()
    
    # Calculate the distance of each node as the perpendicular distance to the nearest edge. Calculate the distance of each strut as the average of its two nodes' distance.
    h = w = d = 11
    distance = lambda x, y, z: min([x - 0, (h-1) - x, y - 0, (w-1) - y, z - 0, (d-1) - z])
    distances = np.mean([
        [distance(x, y, z) for x, y, z in locations_1],
        [distance(x, y, z) for x, y, z in locations_2],
    ], axis=0)

    error = (p - y).squeeze()
    unique_distances = np.unique(distances)

    plt.figure()
    plt.violinplot(
        [error[distances == d] for d in unique_distances],
        positions=unique_distances,
        showmeans=True,
        showextrema=False,
    )
    plt.xlabel('Edge Distance')
    plt.ylabel('Error')
    plt.title(f'Errors By Edge Distance ({p.size} data)')
    plt.show()

def plot_error_by_angle(p, y, locations_1: List[Tuple[int, int, int]], locations_2: List[Tuple[int, int, int]]):
    """Plot the errors vs. strut angles."""
    angles = [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 1),
    ]
    angle_indices = np.array([
        angles.index((abs(x2 - x1), abs(y2 - y1), abs(z2 - z1)))
        for (x1,y1,z1), (x2,y2,z2) in zip(locations_1, locations_2)
    ])

    error = (p - y).squeeze()
    unique_angle_indices = np.unique(angle_indices)

    plt.figure()
    plt.violinplot(
        [error[angle_indices == i] for i in unique_angle_indices],
        positions=unique_angle_indices,
        showmeans=True,
        showextrema=False,
    )
    plt.xticks(range(len(angles)), labels=[str(_) for _ in angles])
    plt.ylabel('Error')
    plt.title(f'Errors By Angle ({p.size} data)')
    plt.show()

def plot_histograms(p, y, bins=20):
    """Plot the histograms of predictions and true values on a single plot."""
    plt.figure()
    plt.hist(p, range=(0, 1), bins=bins, alpha=0.5, label=f'Predicted (mean {p.mean():.3e})')
    plt.hist(y, range=(0, 1), bins=bins, alpha=0.5, label=f'True (mean {y.mean():.3e})')
    plt.legend()
    plt.title('Histogram')
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

def evaluate(p, y) -> Dict[str, float]:
    results = {
        'ME': me(p, y),
        'MAE': mae(p, y),
        # 'MSE': mse(p, y),
        # 'MAE among nonzeros': mae_nonzeros(p, y),
        'Minimum error': min_error(p, y),
        'Maximum error': max_error(p, y),
        # 'Zeros correct': fraction_of_zeros_correct(p, y),
        # 'Zeros incorrect': fraction_of_zeros_incorrect(p, y),
        # 'Number of values out of bounds': ((p < 0) + (p > 1)).sum(),
        # 'Fraction of values out of bounds': ((p < 0) + (p > 1)).sum() / p.size,
    }
    
    return results