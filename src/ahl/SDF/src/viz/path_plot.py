"""Visualization utilities for paths on SDF fields."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from src.sdf.grid import SDFGrid


def plot_path_on_sdf(sdf_grid: SDFGrid, path: np.ndarray,
                     start: np.ndarray = None, goal: np.ndarray = None,
                     ax=None, path_color: str = 'lime', path_linewidth: float = 2.5,
                     show_sdf: bool = True, title: str = "Path on SDF"):
    """Overlay a path on the SDF visualization.

    Args:
        sdf_grid: The signed distance field.
        path: (N, 2) array of (x, y) world coordinates.
        start: Optional start point marker.
        goal: Optional goal point marker.
        ax: Matplotlib axes (created if None).
        path_color: Color for the path line.
        path_linewidth: Width of the path line.
        show_sdf: Whether to show the SDF background.
        title: Plot title.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    if show_sdf:
        X, Y = sdf_grid.meshgrid_world()
        # Show SDF as background
        ax.contourf(X, Y, sdf_grid.values, levels=50, cmap='RdBu', alpha=0.6)
        ax.contour(X, Y, sdf_grid.values, levels=[0],
                   colors='black', linewidths=2)
        ax.contourf(X, Y, sdf_grid.values, levels=[-1e10, 0],
                    colors='gray', alpha=0.5)

    if len(path) > 0:
        ax.plot(path[:, 0], path[:, 1], color=path_color,
                linewidth=path_linewidth, zorder=5, label='Path')

    if start is not None:
        start = np.asarray(start)
        ax.plot(start[0], start[1], 'g^', markersize=14,
                zorder=10, label='Start', markeredgecolor='black')
    if goal is not None:
        goal = np.asarray(goal)
        ax.plot(goal[0], goal[1], 'r*', markersize=16,
                zorder=10, label='Goal', markeredgecolor='black')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    return ax


def plot_path_curvature(path: np.ndarray, curvatures: np.ndarray,
                        min_bend_radius: float = None, ax=None,
                        title: str = "Path Colored by Curvature"):
    """Plot path colored by curvature. Violations shown in red.

    Args:
        path: (N, 2) path points.
        curvatures: (N-2,) curvature at interior points.
        min_bend_radius: If provided, segments violating this radius are red.
        ax: Matplotlib axes.
        title: Plot title.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Build line segments from interior points
    points = path[1:-1].reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    if min_bend_radius is not None:
        max_curvature = 1.0 / min_bend_radius
        # Color: green if OK, red if violation
        colors = []
        for k in curvatures[:-1]:
            if k > max_curvature:
                colors.append('red')
            else:
                colors.append('green')
        lc = LineCollection(segments, colors=colors, linewidths=3)
    else:
        lc = LineCollection(segments, cmap='hot', linewidths=3)
        lc.set_array(curvatures[:-1])
        plt.colorbar(lc, ax=ax, label='Curvature (1/m)')

    ax.add_collection(lc)
    ax.autoscale()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.set_aspect('equal')
    return ax


def plot_comparison(sdf_grid: SDFGrid, paths: dict, start: np.ndarray,
                    goal: np.ndarray, title: str = "Algorithm Comparison"):
    """Plot multiple paths on the same SDF for algorithm comparison.

    Args:
        sdf_grid: The signed distance field.
        paths: Dict of {name: (path_array, cost)} or {name: PlanResult}.
        start: Start point.
        goal: Goal point.
        title: Plot title.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))

    X, Y = sdf_grid.meshgrid_world()
    ax.contourf(X, Y, sdf_grid.values, levels=50, cmap='RdBu', alpha=0.5)
    ax.contour(X, Y, sdf_grid.values, levels=[0], colors='black', linewidths=2)
    ax.contourf(X, Y, sdf_grid.values, levels=[-1e10, 0],
                colors='gray', alpha=0.5)

    colors = ['lime', 'cyan', 'magenta', 'yellow', 'orange']
    for i, (name, data) in enumerate(paths.items()):
        color = colors[i % len(colors)]
        if hasattr(data, 'path'):  # PlanResult
            path = data.path
            label = f"{name} (cost={data.cost:.2f}, t={data.time_seconds:.3f}s)"
        else:
            path, cost = data
            label = f"{name} (cost={cost:.2f})"

        if len(path) > 0:
            ax.plot(path[:, 0], path[:, 1], color=color, linewidth=2.5,
                    label=label, zorder=5 + i)

    start = np.asarray(start)
    goal = np.asarray(goal)
    ax.plot(start[0], start[1], 'g^', markersize=14, zorder=20,
            label='Start', markeredgecolor='black')
    ax.plot(goal[0], goal[1], 'r*', markersize=16, zorder=20,
            label='Goal', markeredgecolor='black')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=9)
    return fig, ax
