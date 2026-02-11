"""Example 05: CHOMP-style Gradient Path Optimization.

Demonstrates:
- Taking an A* path as initial solution
- Refining it with gradient-based optimization (smoothness + obstacle avoidance)
- Visualizing the optimization process and convergence
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib.pyplot as plt

from src.sdf import build_sdf_2d
from src.planning import AStarPlanner, GradientOptimizer
from src.harness import path_statistics
from src.scenarios.simple_2d import scattered_circles, u_shaped_obstacle


def optimize_and_visualize(scenario_name, scenario_func):
    """Run gradient optimization on a scenario and visualize results."""
    obstacles, bounds, resolution, start, goal = scenario_func()
    sdf_grid = build_sdf_2d(obstacles, bounds, resolution)

    print(f"\n{'='*50}")
    print(f"Scenario: {scenario_name}")
    print(f"{'='*50}")

    # Initial path from A*
    planner = AStarPlanner(safety_margin=1.0, cost_weight=8.0)
    result = planner.plan(sdf_grid, start, goal)
    if not result.success:
        print("A* planning failed!")
        return None

    initial_path = result.path
    print(f"A* path: {len(initial_path)} points, cost={result.cost:.2f}")

    # Subsample A* path for optimization (too many points = slow)
    n_waypoints = min(60, len(initial_path))
    indices = np.linspace(0, len(initial_path) - 1, n_waypoints, dtype=int)
    opt_input = initial_path[indices]

    # Gradient optimization
    optimizer = GradientOptimizer(
        sdf_grid,
        safety_margin=1.2,
        lambda_obs=15.0,
        learning_rate=0.015,
        max_iters=400,
        convergence_tol=1e-5,
    )
    opt_result = optimizer.optimize(opt_input, record_history=True)
    optimized_path = opt_result['path']

    print(f"Optimization: {opt_result['iterations']} iterations, "
          f"time={opt_result['time_seconds']:.3f}s")

    # Statistics
    stats_before = path_statistics(initial_path, sdf_grid)
    stats_after = path_statistics(optimized_path, sdf_grid)
    print(f"Before: length={stats_before['length']:.2f}, "
          f"min_clearance={stats_before['min_clearance']:.3f}")
    print(f"After:  length={stats_after['length']:.2f}, "
          f"min_clearance={stats_after['min_clearance']:.3f}")

    # --- Visualization ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    X, Y = sdf_grid.meshgrid_world()

    # 1. Before optimization
    ax = axes[0, 0]
    ax.contourf(X, Y, sdf_grid.values, levels=50, cmap='RdBu', alpha=0.5)
    ax.contour(X, Y, sdf_grid.values, levels=[0], colors='black', linewidths=2)
    ax.contourf(X, Y, sdf_grid.values, levels=[-1e10, 0], colors='gray', alpha=0.5)
    ax.plot(initial_path[:, 0], initial_path[:, 1], 'lime', linewidth=2,
            label=f"A* (len={stats_before['length']:.1f})")
    ax.plot(start[0], start[1], 'g^', markersize=12)
    ax.plot(goal[0], goal[1], 'r*', markersize=14)
    ax.set_title('Before: A* Path')
    ax.set_aspect('equal')
    ax.legend()

    # 2. After optimization
    ax = axes[0, 1]
    ax.contourf(X, Y, sdf_grid.values, levels=50, cmap='RdBu', alpha=0.5)
    ax.contour(X, Y, sdf_grid.values, levels=[0], colors='black', linewidths=2)
    ax.contourf(X, Y, sdf_grid.values, levels=[-1e10, 0], colors='gray', alpha=0.5)
    ax.plot(initial_path[:, 0], initial_path[:, 1], 'lime', linewidth=1,
            alpha=0.4, label='A* (original)')
    ax.plot(optimized_path[:, 0], optimized_path[:, 1], 'cyan', linewidth=2.5,
            label=f"Optimized (len={stats_after['length']:.1f})")
    ax.plot(start[0], start[1], 'g^', markersize=12)
    ax.plot(goal[0], goal[1], 'r*', markersize=14)
    ax.set_title('After: Gradient Optimized')
    ax.set_aspect('equal')
    ax.legend()

    # 3. Optimization evolution (show intermediate paths)
    ax = axes[1, 0]
    ax.contourf(X, Y, sdf_grid.values, levels=50, cmap='RdBu', alpha=0.4)
    ax.contour(X, Y, sdf_grid.values, levels=[0], colors='black', linewidths=2)
    ax.contourf(X, Y, sdf_grid.values, levels=[-1e10, 0], colors='gray', alpha=0.5)

    history = opt_result.get('path_history', [])
    n_show = min(8, len(history))
    if n_show > 0:
        step = max(1, len(history) // n_show)
        cmap = plt.cm.cool
        for j, idx in enumerate(range(0, len(history), step)):
            p = history[idx]
            color = cmap(j / max(n_show - 1, 1))
            ax.plot(p[:, 0], p[:, 1], color=color, linewidth=1.5,
                    alpha=0.7, label=f'iter {idx}')
    ax.plot(start[0], start[1], 'g^', markersize=12)
    ax.plot(goal[0], goal[1], 'r*', markersize=14)
    ax.set_title('Optimization Evolution')
    ax.set_aspect('equal')
    ax.legend(fontsize=8)

    # 4. Cost convergence
    ax = axes[1, 1]
    costs = opt_result['cost_history']
    ax.plot(costs, 'b-', linewidth=1.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Total Cost')
    ax.set_title('Cost Convergence')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"Gradient Optimization - {scenario_name}", fontsize=14)
    plt.tight_layout()
    return fig


def main():
    scenarios = [
        ("Scattered Circles", scattered_circles),
        ("U-Shaped Obstacle", u_shaped_obstacle),
    ]

    for name, func in scenarios:
        fig = optimize_and_visualize(name, func)
        if fig:
            filename = f"05_gradient_{name.lower().replace(' ', '_').replace('-', '_')}.png"
            fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Saved: {filename}")
            plt.close(fig)

    plt.show()


if __name__ == "__main__":
    main()
