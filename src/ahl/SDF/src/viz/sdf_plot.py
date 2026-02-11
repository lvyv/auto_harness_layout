"""Visualization utilities for SDF fields."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from src.sdf.grid import SDFGrid


def plot_sdf_2d(sdf_grid: SDFGrid, ax=None, show_contours: bool = True,
                cmap: str = 'RdBu', title: str = "SDF Field"):
    """Plot 2D SDF as filled contour with zero-level boundary highlighted.

    Blue = positive (free space), Red = negative (inside obstacle),
    Black contour = zero level (obstacle boundary).
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    X, Y = sdf_grid.meshgrid_world()
    vals = sdf_grid.values

    # Use TwoSlopeNorm to center colormap at zero
    vmax = max(abs(vals.min()), abs(vals.max()))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    cf = ax.contourf(X, Y, vals, levels=50, cmap=cmap, norm=norm)
    plt.colorbar(cf, ax=ax, label='Signed Distance')

    if show_contours:
        # Zero contour (obstacle boundary)
        ax.contour(X, Y, vals, levels=[0], colors='black', linewidths=2)
        # Additional distance contours
        positive_levels = np.arange(0.5, vmax, 0.5)
        if len(positive_levels) > 0:
            ax.contour(X, Y, vals, levels=positive_levels,
                       colors='gray', linewidths=0.5, alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.set_aspect('equal')
    return ax


def plot_cost_field(sdf_grid: SDFGrid, safety_margin: float,
                    alpha: float = 10.0, ax=None,
                    title: str = "Cost Field"):
    """Plot the SDF-derived cost field used for path planning."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    X, Y = sdf_grid.meshgrid_world()
    cost = sdf_grid.cost_field(safety_margin, alpha)

    # Clip infinite values for visualization
    cost_vis = np.where(np.isinf(cost), np.nan, cost)

    cf = ax.contourf(X, Y, cost_vis, levels=50, cmap='YlOrRd')
    plt.colorbar(cf, ax=ax, label='Traversal Cost')

    # Show obstacle regions
    ax.contourf(X, Y, sdf_grid.values, levels=[-1e10, 0],
                colors='black', alpha=0.8)
    ax.contour(X, Y, sdf_grid.values, levels=[0],
               colors='white', linewidths=1.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.set_aspect('equal')
    return ax


def plot_gradient_field(sdf_grid: SDFGrid, ax=None, subsample: int = 5,
                        title: str = "SDF Gradient Field"):
    """Plot SDF gradient as a quiver (arrow) plot."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    X, Y = sdf_grid.meshgrid_world()
    grad = sdf_grid.gradient()  # (2, ny, nx): [gx, gy]
    gx, gy = grad[0], grad[1]

    # Subsample for readability
    s = subsample
    ax.quiver(X[::s, ::s], Y[::s, ::s],
              gx[::s, ::s], gy[::s, ::s],
              color='navy', alpha=0.6, scale=30)

    # Show obstacle boundary
    ax.contour(X, Y, sdf_grid.values, levels=[0],
               colors='red', linewidths=2)
    ax.contourf(X, Y, sdf_grid.values, levels=[-1e10, 0],
                colors='lightcoral', alpha=0.3)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.set_aspect('equal')
    return ax
