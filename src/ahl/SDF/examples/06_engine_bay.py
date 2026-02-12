"""Example 06: Complete Engine Bay Wire Harness Routing.

Demonstrates the full pipeline on a realistic automotive scenario:
  SDF generation -> Cost field -> A*/FMM planning -> Gradient optimization
  -> B-spline smoothing -> Bend radius constraint verification

Includes algorithm comparison table and multi-panel visualization.
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib.pyplot as plt

from src.sdf import build_sdf_2d
from src.planning import AStarPlanner, FastMarchingPlanner, GradientOptimizer
from src.harness import (path_statistics, smooth_path_bspline,
                         smooth_with_bend_constraint)
from src.scenarios.engine_bay import engine_bay, engine_bay_multi_route


def draw_obstacle_labels(ax, obstacles):
    """Draw labels for engine bay components."""
    for obs in obstacles:
        label = obs.get("label", "")
        if not label:
            continue
        center = obs["center"]
        ax.annotate(label, xy=center, fontsize=7, ha='center', va='center',
                    color='white', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15', fc='black', alpha=0.6))


def run_single_route():
    """Run the complete pipeline on a single ECU->Sensor route."""
    obstacles, bounds, resolution, start, goal, waypoints, meta = engine_bay()
    sdf_grid = build_sdf_2d(obstacles, bounds, resolution)

    # Cable spec
    min_bend_radius = 0.3  # in decimeters (~30mm for AWG16 at 4x)
    safety_margin = 0.4    # minimum clearance from obstacles

    print("="*60)
    print("Engine Bay Wire Harness Routing")
    print(f"Route: {meta['description']}")
    print(f"Min bend radius: {min_bend_radius} dm")
    print(f"Safety margin: {safety_margin} dm")
    print("="*60)

    # --- Algorithm 1: A* ---
    astar = AStarPlanner(safety_margin=safety_margin, cost_weight=12.0)
    result_astar = astar.plan(sdf_grid, start, goal)

    # --- Algorithm 2: FMM ---
    fmm = FastMarchingPlanner(safety_margin=safety_margin * 2, speed_exponent=2.0)
    result_fmm = fmm.plan(sdf_grid, start, goal)

    # --- Algorithm 3: A* + Gradient Optimization ---
    optimized_path = None
    if result_astar.success:
        n_wp = min(80, len(result_astar.path))
        indices = np.linspace(0, len(result_astar.path) - 1, n_wp, dtype=int)
        init_path = result_astar.path[indices]

        optimizer = GradientOptimizer(
            sdf_grid, safety_margin=safety_margin * 1.5,
            lambda_obs=20.0, learning_rate=0.012, max_iters=500,
        )
        opt_result = optimizer.optimize(init_path)
        optimized_path = opt_result['path']

    # --- Algorithm 4: A* + Gradient + Constrained B-spline ---
    final_path = None
    if optimized_path is not None:
        smooth_result = smooth_with_bend_constraint(
            optimized_path, min_bend_radius, sdf_grid,
            min_clearance=0.05, num_output_points=300,
        )
        final_path = smooth_result['path']

    # --- Statistics Table ---
    all_paths = {}
    if result_astar.success:
        all_paths["A*"] = result_astar.path
    if result_fmm.success:
        all_paths["FMM"] = result_fmm.path
    if optimized_path is not None:
        all_paths["A*+GradOpt"] = optimized_path
    if final_path is not None:
        all_paths["A*+GradOpt+Smooth"] = final_path

    print(f"\n{'Algorithm':<22} {'Length':>8} {'MinClr':>8} {'MinBR':>8} {'BR Viol':>8} {'Points':>8}")
    print("-" * 72)
    for name, path in all_paths.items():
        stats = path_statistics(path, sdf_grid, min_bend_radius)
        print(f"{name:<22} {stats['length']:>8.2f} {stats['min_clearance']:>8.3f} "
              f"{stats['min_bend_radius_actual']:>8.3f} "
              f"{stats['bend_violations_count']:>8d} {stats['num_points']:>8d}")

    # --- Visualization ---
    fig = plt.figure(figsize=(20, 14))

    # Panel layout: 2x3 grid
    ax1 = fig.add_subplot(2, 3, 1)  # SDF field
    ax2 = fig.add_subplot(2, 3, 2)  # Cost field
    ax3 = fig.add_subplot(2, 3, 3)  # A* path
    ax4 = fig.add_subplot(2, 3, 4)  # FMM path
    ax5 = fig.add_subplot(2, 3, 5)  # Optimized path
    ax6 = fig.add_subplot(2, 3, 6)  # Final comparison

    xx, yy = sdf_grid.meshgrid_world()

    def draw_base(ax, title, alpha=0.5):
        ax.contourf(xx, yy, sdf_grid.values, levels=50, cmap='RdBu', alpha=alpha)
        ax.contour(xx, yy, sdf_grid.values, levels=[0], colors='black', linewidths=1.5)
        ax.contourf(xx, yy, sdf_grid.values, levels=[-1e10, 0], colors='#404040', alpha=0.8)
        draw_obstacle_labels(ax, obstacles)
        ax.plot(start[0], start[1], 'g^', markersize=10, zorder=20, markeredgecolor='black')
        ax.plot(goal[0], goal[1], 'r*', markersize=12, zorder=20, markeredgecolor='black')
        ax.set_title(title, fontsize=10)
        ax.set_aspect('equal')
        ax.set_xlim(bounds[0])
        ax.set_ylim(bounds[1])

    # 1. SDF field
    from matplotlib.colors import TwoSlopeNorm
    vmax = max(abs(sdf_grid.values.min()), abs(sdf_grid.values.max()))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cf = ax1.contourf(xx, yy, sdf_grid.values, levels=50, cmap='RdBu', norm=norm)
    ax1.contour(xx, yy, sdf_grid.values, levels=[0], colors='black', linewidths=2)
    plt.colorbar(cf, ax=ax1, label='SDF', shrink=0.8)
    draw_obstacle_labels(ax1, obstacles)
    ax1.set_title('SDF Field', fontsize=10)
    ax1.set_aspect('equal')

    # 2. Cost field
    cost = sdf_grid.cost_field(safety_margin, alpha=12.0)
    cost_vis = np.where(np.isinf(cost), np.nan, cost)
    cf2 = ax2.contourf(xx, yy, cost_vis, levels=50, cmap='YlOrRd')
    ax2.contourf(xx, yy, sdf_grid.values, levels=[-1e10, 0], colors='black', alpha=0.8)
    plt.colorbar(cf2, ax=ax2, label='Cost', shrink=0.8)
    draw_obstacle_labels(ax2, obstacles)
    ax2.set_title(f'Cost Field (margin={safety_margin})', fontsize=10)
    ax2.set_aspect('equal')

    # 3. A* path
    draw_base(ax3, 'A* Path')
    if result_astar.success:
        ax3.plot(result_astar.path[:, 0], result_astar.path[:, 1],
                 'lime', linewidth=2, zorder=10)

    # 4. FMM path
    draw_base(ax4, 'FMM Path')
    if result_fmm.success:
        ax4.plot(result_fmm.path[:, 0], result_fmm.path[:, 1],
                 'cyan', linewidth=2, zorder=10)

    # 5. Gradient optimized
    draw_base(ax5, 'A* + Gradient Optimized')
    if optimized_path is not None:
        ax5.plot(result_astar.path[:, 0], result_astar.path[:, 1],
                 'lime', linewidth=1, alpha=0.3, zorder=8)
        ax5.plot(optimized_path[:, 0], optimized_path[:, 1],
                 'magenta', linewidth=2.5, zorder=10)

    # 6. Final comparison: all paths overlaid
    draw_base(ax6, 'All Algorithms Comparison', alpha=0.3)
    path_styles = {
        "A*": ('lime', 1.5, 0.5),
        "FMM": ('cyan', 1.5, 0.5),
        "A*+GradOpt": ('magenta', 2.0, 0.6),
        "A*+GradOpt+Smooth": ('yellow', 3.0, 1.0),
    }
    for name, path in all_paths.items():
        color, lw, alpha_d = path_styles.get(name, ('white', 1.5, 0.5))
        ax6.plot(path[:, 0], path[:, 1], color=color, linewidth=lw,
                 alpha=alpha_d, label=name, zorder=10)
    ax6.legend(fontsize=8, loc='upper right')

    plt.suptitle("Engine Bay Wire Harness Routing - Complete Pipeline",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("06_engine_bay.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved: 06_engine_bay.png")

    return fig


def run_multi_route():
    """Run multiple cable routes in the engine bay."""
    obstacles, bounds, resolution, routes = engine_bay_multi_route()
    sdf_grid = build_sdf_2d(obstacles, bounds, resolution)

    print("\n" + "="*60)
    print("Multi-Route Engine Bay Routing")
    print("="*60)

    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    xx, yy = sdf_grid.meshgrid_world()
    ax.contourf(xx, yy, sdf_grid.values, levels=50, cmap='RdBu', alpha=0.4)
    ax.contour(xx, yy, sdf_grid.values, levels=[0], colors='black', linewidths=1.5)
    ax.contourf(xx, yy, sdf_grid.values, levels=[-1e10, 0], colors='#404040', alpha=0.8)
    draw_obstacle_labels(ax, obstacles)

    for route in routes:
        name = route["name"]
        start, goal = route["start"], route["goal"]
        color = route["color"]

        # Plan with A* + gradient optimization
        planner = AStarPlanner(safety_margin=0.5, cost_weight=12.0)
        result = planner.plan(sdf_grid, start, goal)

        if result.success:
            # Optimize
            n_wp = min(60, len(result.path))
            indices = np.linspace(0, len(result.path) - 1, n_wp, dtype=int)
            opt = GradientOptimizer(sdf_grid, safety_margin=0.6,
                                    lambda_obs=15.0, learning_rate=0.015,
                                    max_iters=300)
            opt_result = opt.optimize(result.path[indices])
            path = smooth_path_bspline(opt_result['path'], 200)
            stats = path_statistics(path, sdf_grid)

            ax.plot(path[:, 0], path[:, 1], color=color, linewidth=3,
                    label=f"{name} (L={stats['length']:.1f})", zorder=10)
            print(f"  {name}: length={stats['length']:.2f}, "
                  f"min_clr={stats['min_clearance']:.3f}")
        else:
            print(f"  {name}: FAILED")

        ax.plot(start[0], start[1], 'o', color=color, markersize=10,
                zorder=20, markeredgecolor='black')
        ax.plot(goal[0], goal[1], 's', color=color, markersize=10,
                zorder=20, markeredgecolor='black')

    ax.set_title("Engine Bay - Multi-Route Harness Layout", fontsize=13)
    ax.set_aspect('equal')
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("06_engine_bay_multi.png", dpi=150, bbox_inches='tight')
    print(f"Saved: 06_engine_bay_multi.png")

    return fig


def main():
    run_single_route()
    run_multi_route()
    plt.show()


if __name__ == "__main__":
    main()
