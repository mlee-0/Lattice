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

def mre(p, y):
    """Mean relative error."""
    return np.mean(np.abs(p - y) / y) * 100

def min_error(p, y):
    """Smallest error (which may be a large negative number)."""
    return np.min(p - y)

def max_error(p, y):
    """Largest error."""
    return np.max(p - y)

def fraction_of_zeros_correct(p, y) -> float:
    """Fraction [0, 1] of all true 0 values for which a 0 was predicted. Similar to false negative."""
    zero_predictions = p[y == 0]
    if zero_predictions.size == 0:
        return np.nan
    else:
        return (zero_predictions == 0).sum() / zero_predictions.size

def fraction_of_zeros_incorrect(p, y) -> float:
    """Fraction [0, 1] of all true 0 values for which a nonzero value was predicted. Similar to false positive."""
    zero_predictions = p[y == 0]
    if zero_predictions.size == 0:
        return np.nan
    else:
        return (zero_predictions != 0).sum() / zero_predictions.size

def plot_predicted_vs_true(p, y):
    """Plot the predictions vs. true values sorted from lowest to highest. Intended to reveal which ranges of values produce inaccurate predictions."""
    p, y = p.flatten(), y.flatten()

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
    """Plot the absolute errors vs. distances from the edges. Intended to reveal where in the 3D volume predictions are inaccurate."""

    p, y = p.flatten(), y.flatten()
    
    # Calculate the distance of each node as the perpendicular distance to the nearest edge. Calculate the distance of each strut as the average of its two nodes' distance.
    h = w = d = 11
    distance = lambda x, y, z: min([x - 0, (h-1) - x, y - 0, (w-1) - y, z - 0, (d-1) - z])
    distances = np.mean([
        [distance(x, y, z) for x, y, z in locations_1],
        [distance(x, y, z) for x, y, z in locations_2],
    ], axis=0)

    error = np.abs(p - y).squeeze()
    unique_distances = np.unique(distances)

    plt.figure()
    plt.violinplot(
        [error[distances == d] for d in unique_distances],
        positions=unique_distances,
        showmeans=True,
        showextrema=False,
    )
    plt.xlabel('Edge Distance')
    plt.ylabel('Absolute Error')
    plt.title(f'({p.size} data)')
    plt.show()

def plot_error_by_xy_edge_distance(p, y, locations_1: List[Tuple[int, int, int]], locations_2: List[Tuple[int, int, int]]):
    """Plot the absolute errors vs. distances from the x=0 and y=0 edges. To verify if a cropping issue in the density images is affecting predictions."""

    p, y = p.flatten(), y.flatten()
    
    # Calculate the distance of each node as the perpendicular distance to the nearest edge. Calculate the distance of each strut as the average of its two nodes' distance.
    h = w = d = 11
    distance = lambda x, y, z: min([x - 0, y - 0])
    distances = np.mean([
        [distance(x, y, z) for x, y, z in locations_1],
        [distance(x, y, z) for x, y, z in locations_2],
    ], axis=0)

    error = np.abs(p - y).squeeze()
    unique_distances = np.unique(distances)

    plt.figure()
    plt.violinplot(
        [error[distances == d] for d in unique_distances],
        positions=unique_distances,
        showmeans=True,
        showextrema=False,
    )
    plt.xlabel('XY Edge Distance')
    plt.ylabel('Absolute Error')
    plt.title(f'({p.size} data)')
    plt.show()

def plot_error_by_angle(p, y, locations_1: List[Tuple[int, int, int]], locations_2: List[Tuple[int, int, int]]):
    """Plot the absolute errors vs. strut angles."""
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

    error = np.abs(p - y).squeeze()
    unique_angle_indices = np.unique(angle_indices)

    plt.figure()
    plt.violinplot(
        [error[angle_indices == i] for i in unique_angle_indices],
        positions=unique_angle_indices,
        showmeans=True,
        showextrema=False,
    )
    plt.xticks(range(len(angles)), labels=[str(_) for _ in angles])
    plt.ylabel('Absolute Error')
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


if __name__ == '__main__':
    pass