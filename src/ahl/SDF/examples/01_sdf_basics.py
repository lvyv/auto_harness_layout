"""Example 01: SDF Generation and Visualization.

Demonstrates:
- Building a 2D SDF from obstacle definitions
- Visualizing the SDF field, zero-level contour, and gradient field
- Converting SDF to a cost field for path planning
"""

import sys
sys.path.insert(0, ".")

import matplotlib.pyplot as plt

from src.sdf import build_sdf_2d
from src.viz import plot_sdf_2d, plot_cost_field, plot_gradient_field
from src.scenarios.simple_2d import scattered_circles


def main():
    # Load scenario
    obstacles, bounds, resolution, start, goal = scattered_circles()

    # Build SDF
    print("Building SDF...")
    sdf_grid = build_sdf_2d(obstacles, bounds, resolution)
    print(f"  Grid shape: {sdf_grid.shape}")
    print(f"  SDF range: [{sdf_grid.values.min():.3f}, {sdf_grid.values.max():.3f}]")
    print(f"  Spacing: {sdf_grid.spacing:.4f}")

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # 1. SDF field
    plot_sdf_2d(sdf_grid, ax=axes[0], title="SDF Field")
    axes[0].plot(start[0], start[1], 'g^', markersize=12, label='Start')
    axes[0].plot(goal[0], goal[1], 'r*', markersize=14, label='Goal')
    axes[0].legend()

    # 2. Cost field (safety_margin = 1.0)
    plot_cost_field(sdf_grid, safety_margin=1.0, alpha=10.0,
                    ax=axes[1], title="Cost Field (margin=1.0)")

    # 3. Gradient field
    plot_gradient_field(sdf_grid, ax=axes[2], subsample=8,
                        title="SDF Gradient Field")

    plt.suptitle("SDF Basics - Scattered Circles Scenario", fontsize=14)
    plt.tight_layout()
    plt.savefig("01_sdf_basics.png", dpi=150, bbox_inches='tight')
    print("Saved: 01_sdf_basics.png")
    plt.show()


if __name__ == "__main__":
    main()
