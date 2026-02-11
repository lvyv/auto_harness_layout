"""Example 04: Bend Radius Constraint and Path Smoothing.

Demonstrates:
- Computing curvature along a planned path
- Checking bend radius violations for automotive cables
- B-spline smoothing with bend radius constraint enforcement
- Before/after comparison with curvature color-coding
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from src.sdf import build_sdf_2d
from src.planning import AStarPlanner
from src.harness import (Cable, compute_curvature, compute_bend_radius,
                         check_bend_radius, smooth_path_bspline,
                         smooth_with_bend_constraint, path_statistics)
from src.scenarios.simple_2d import scattered_circles


def plot_path_with_curvature(ax, path, curvatures, min_bend_radius,
                             sdf_grid, title):
    """Plot path colored by bend radius compliance."""
    X, Y = sdf_grid.meshgrid_world()
    ax.contourf(X, Y, sdf_grid.values, levels=50, cmap='RdBu', alpha=0.4)
    ax.contour(X, Y, sdf_grid.values, levels=[0], colors='black', linewidths=2)
    ax.contourf(X, Y, sdf_grid.values, levels=[-1e10, 0],
                colors='gray', alpha=0.5)

    if len(path) >= 3 and len(curvatures) > 0:
        max_kappa = 1.0 / min_bend_radius
        # Color segments by violation status
        points = path[1:-1].reshape(-1, 1, 2)
        if len(points) > 1:
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            colors = ['red' if k > max_kappa else 'lime'
                      for k in curvatures[:-1]]
            lc = LineCollection(segments, colors=colors, linewidths=3, zorder=5)
            ax.add_collection(lc)

    ax.plot(path[0, 0], path[0, 1], 'g^', markersize=12, zorder=10)
    ax.plot(path[-1, 0], path[-1, 1], 'r*', markersize=14, zorder=10)
    ax.set_title(title)
    ax.set_aspect('equal')


def main():
    # Setup
    obstacles, bounds, resolution, start, goal = scattered_circles()
    sdf_grid = build_sdf_2d(obstacles, bounds, resolution)

    # Define automotive cable
    cable = Cable.automotive("Engine_Sensor_12AWG", diameter_mm=5.5,
                             bend_factor=4.0, color="orange")
    # Scale for our 10x10 scenario (treating units as decimeters for visibility)
    min_bend_radius = 0.5  # Minimum bend radius in scenario units

    print(f"Cable: {cable.name}")
    print(f"Min bend radius: {min_bend_radius}")
    print()

    # Plan path with A*
    planner = AStarPlanner(safety_margin=0.8, cost_weight=10.0)
    result = planner.plan(sdf_grid, start, goal)
    if not result.success:
        print("Planning failed!")
        return

    raw_path = result.path
    print(f"Raw A* path: {len(raw_path)} points")

    # -- Analyze raw path --
    raw_stats = path_statistics(raw_path, sdf_grid, min_bend_radius)
    print(f"  Length: {raw_stats['length']:.2f}")
    print(f"  Min clearance: {raw_stats['min_clearance']:.3f}")
    print(f"  Min bend radius: {raw_stats['min_bend_radius_actual']:.3f}")
    print(f"  Bend violations: {raw_stats['bend_violations_count']}")

    # -- Simple B-spline smoothing --
    smooth_path = smooth_path_bspline(raw_path, num_output_points=200)
    smooth_stats = path_statistics(smooth_path, sdf_grid, min_bend_radius)
    print(f"\nB-spline smoothed path: {len(smooth_path)} points")
    print(f"  Length: {smooth_stats['length']:.2f}")
    print(f"  Min clearance: {smooth_stats['min_clearance']:.3f}")
    print(f"  Min bend radius: {smooth_stats['min_bend_radius_actual']:.3f}")
    print(f"  Bend violations: {smooth_stats['bend_violations_count']}")

    # -- Constrained smoothing --
    result_constrained = smooth_with_bend_constraint(
        raw_path, min_bend_radius, sdf_grid,
        min_clearance=0.1, num_output_points=200,
    )
    constrained_path = result_constrained['path']
    constrained_stats = path_statistics(constrained_path, sdf_grid, min_bend_radius)
    print(f"\nConstrained smoothed path: {len(constrained_path)} points")
    print(f"  Converged: {result_constrained['converged']}")
    print(f"  Iterations: {result_constrained['iterations']}")
    print(f"  Length: {constrained_stats['length']:.2f}")
    print(f"  Min clearance: {constrained_stats['min_clearance']:.3f}")
    print(f"  Min bend radius: {constrained_stats['min_bend_radius_actual']:.3f}")
    print(f"  Bend violations: {constrained_stats['bend_violations_count']}")

    # -- Visualization --
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # 1. Raw A* path
    raw_curvatures = compute_curvature(raw_path)
    plot_path_with_curvature(axes[0, 0], raw_path, raw_curvatures,
                             min_bend_radius, sdf_grid,
                             f"Raw A* Path\n({raw_stats['bend_violations_count']} violations)")

    # 2. B-spline smoothed
    smooth_curvatures = compute_curvature(smooth_path)
    plot_path_with_curvature(axes[0, 1], smooth_path, smooth_curvatures,
                             min_bend_radius, sdf_grid,
                             f"B-spline Smoothed\n({smooth_stats['bend_violations_count']} violations)")

    # 3. Constrained smoothed
    constrained_curvatures = compute_curvature(constrained_path)
    plot_path_with_curvature(axes[1, 0], constrained_path, constrained_curvatures,
                             min_bend_radius, sdf_grid,
                             f"Constrained Smoothed\n({constrained_stats['bend_violations_count']} violations)")

    # 4. Bend radius comparison plot
    ax = axes[1, 1]
    if len(raw_curvatures) > 0:
        raw_radii = np.where(raw_curvatures > 1e-12, 1.0 / raw_curvatures, 10.0)
        ax.plot(np.linspace(0, 1, len(raw_radii)), np.clip(raw_radii, 0, 5),
                'b-', alpha=0.5, label='Raw A*')
    if len(smooth_curvatures) > 0:
        smooth_radii = np.where(smooth_curvatures > 1e-12, 1.0 / smooth_curvatures, 10.0)
        ax.plot(np.linspace(0, 1, len(smooth_radii)), np.clip(smooth_radii, 0, 5),
                'g-', alpha=0.7, label='B-spline')
    if len(constrained_curvatures) > 0:
        constr_radii = np.where(constrained_curvatures > 1e-12, 1.0 / constrained_curvatures, 10.0)
        ax.plot(np.linspace(0, 1, len(constr_radii)), np.clip(constr_radii, 0, 5),
                'm-', alpha=0.9, label='Constrained')
    ax.axhline(y=min_bend_radius, color='red', linestyle='--',
               linewidth=2, label=f'Min radius = {min_bend_radius}')
    ax.set_xlabel('Path Progress (normalized)')
    ax.set_ylabel('Bend Radius')
    ax.set_title('Bend Radius Along Path')
    ax.legend()
    ax.set_ylim(0, 5)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Bend Radius Constraint Enforcement\n"
                 f"(Green = OK, Red = Violation, min_R = {min_bend_radius})",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig("04_bend_radius.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved: 04_bend_radius.png")
    plt.show()


if __name__ == "__main__":
    main()
