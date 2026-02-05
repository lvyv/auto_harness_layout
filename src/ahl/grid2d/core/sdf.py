"""Signed Distance Field (SDF) computation for 2D grids."""

from typing import Tuple
import numpy as np
from scipy.ndimage import distance_transform_edt

from .cell_type import CellType


def compute_sdf(cells: np.ndarray) -> np.ndarray:
    """Compute Signed Distance Field to obstacles.

    Uses Euclidean Distance Transform to compute the distance from each
    cell to the nearest obstacle. This is used in A* to penalize paths
    that go too close to obstacles.

    Args:
        cells: (H, W) array of cell types

    Returns:
        (H, W) array of distances to nearest obstacle (float32)
        - 0.0 at obstacle cells
        - Positive values elsewhere (distance in cells)
    """
    # Create obstacle mask
    obstacle_mask = (cells == CellType.OBSTACLE)

    # Compute Euclidean distance transform
    # This gives distance from each free cell to nearest obstacle
    # Inverted mask: True for free cells, False for obstacles
    distances = distance_transform_edt(~obstacle_mask)

    return distances.astype(np.float32)


def compute_gradient_sdf(sdf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute gradient of SDF using central differences.

    The gradient points away from obstacles and has magnitude ≈ 1.

    Args:
        sdf: (H, W) SDF array

    Returns:
        Tuple of (grad_y, grad_x) arrays, each (H, W) float32
    """
    # Central differences (padded with edge values)
    grad_y = np.zeros_like(sdf)
    grad_x = np.zeros_like(sdf)

    # Interior cells (central difference)
    grad_y[1:-1, :] = (sdf[2:, :] - sdf[:-2, :]) / 2.0
    grad_x[:, 1:-1] = (sdf[:, 2:] - sdf[:, :-2]) / 2.0

    # Boundary cells (forward/backward difference)
    grad_y[0, :] = sdf[1, :] - sdf[0, :]
    grad_y[-1, :] = sdf[-1, :] - sdf[-2, :]
    grad_x[:, 0] = sdf[:, 1] - sdf[:, 0]
    grad_x[:, -1] = sdf[:, -1] - sdf[:, -2]

    return grad_y, grad_x
