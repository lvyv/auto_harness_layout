"""Basic 2D test scenarios with simple obstacle configurations."""

import numpy as np


def scattered_circles():
    """Several circular obstacles in a 10x10 space.

    Returns:
        (obstacles, bounds, resolution, start, goal)
    """
    obstacles = [
        {"type": "circle", "center": [3.0, 5.0], "radius": 1.0},
        {"type": "circle", "center": [6.0, 3.0], "radius": 0.8},
        {"type": "circle", "center": [5.0, 7.0], "radius": 1.2},
        {"type": "circle", "center": [7.5, 6.5], "radius": 0.6},
    ]
    bounds = ((0, 10), (0, 10))
    resolution = 0.05
    start = np.array([1.0, 1.0])
    goal = np.array([9.0, 9.0])
    return obstacles, bounds, resolution, start, goal


def narrow_passage():
    """Two large rectangular obstacles forming a narrow gap.

    The cable must thread through a ~0.8 unit gap between two walls.
    """
    obstacles = [
        {"type": "rectangle", "center": [5.0, 3.0], "half_extents": [4.5, 1.0]},
        {"type": "rectangle", "center": [5.0, 7.0], "half_extents": [4.5, 1.0]},
        {"type": "rectangle", "center": [2.0, 5.0], "half_extents": [1.5, 0.6]},
    ]
    bounds = ((0, 10), (0, 10))
    resolution = 0.05
    start = np.array([1.0, 1.0])
    goal = np.array([9.0, 9.0])
    return obstacles, bounds, resolution, start, goal


def u_shaped_obstacle():
    """U-shaped obstacle that forces the path to go around.

    Built from three rectangles forming a U opening to the right.
    """
    obstacles = [
        # Bottom wall
        {"type": "rectangle", "center": [5.0, 3.0], "half_extents": [2.5, 0.3]},
        # Top wall
        {"type": "rectangle", "center": [5.0, 7.0], "half_extents": [2.5, 0.3]},
        # Left wall (connecting bottom and top)
        {"type": "rectangle", "center": [2.5, 5.0], "half_extents": [0.3, 2.3]},
        # Additional obstacle on the right
        {"type": "circle", "center": [8.0, 5.0], "radius": 0.7},
    ]
    bounds = ((0, 10), (0, 10))
    resolution = 0.05
    start = np.array([5.0, 5.0])  # Inside the U
    goal = np.array([9.0, 5.0])
    return obstacles, bounds, resolution, start, goal
