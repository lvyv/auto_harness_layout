"""Example 07: Multi-Route Constraints — Attraction & Repulsion.

Demonstrates path-interaction constraints for multi-harness routing:
  1. Plan the first cable normally (baseline)
  2. Attract: second cable reuses the first cable's corridor
  3. Repel:  second cable avoids the first cable's corridor
  4. Comparison visualization of all three behaviors
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib.pyplot as plt

from src.sdf import build_sdf_2d
from src.planning import AStarPlanner, FastMarchingPlanner, GradientOptimizer
from src.harness import (smooth_path_bspline, path_statistics,
                         PathConstraint, build_cost_modifier,
                         build_speed_modifier)
from src.scenarios.engine_bay import engine_bay_multi_route


def draw_obstacle_labels(ax, obstacles):
    for obs in obstacles:
        label = obs.get("label", "")
        if not label:
            continue
        center = obs["center"]
        ax.annotate(label, xy=center, fontsize=6, ha='center', va='center',
                    color='white', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.12', fc='black', alpha=0.6))


def draw_base(ax, sdf_grid, obstacles, bounds, title):
    X, Y = sdf_grid.meshgrid_world()
    ax.contourf(X, Y, sdf_grid.values, levels=50, cmap='RdBu', alpha=0.35)
    ax.contour(X, Y, sdf_grid.values, levels=[0], colors='black', linewidths=1.2)
    ax.contourf(X, Y, sdf_grid.values, levels=[-1e10, 0], colors='#404040', alpha=0.8)
    draw_obstacle_labels(ax, obstacles)
    ax.set_title(title, fontsize=10)
    ax.set_aspect('equal')
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])


def plan_route(sdf_grid, start, goal, **planner_kwargs):
    """Plan a single route using FMM + gradient optimization."""
    fmm = FastMarchingPlanner(safety_margin=0.5, speed_exponent=2.0)
    result = fmm.plan(sdf_grid, start, goal, **planner_kwargs)
    if not result.success:
        return None

    n_wp = min(60, len(result.path))
    indices = np.linspace(0, len(result.path) - 1, n_wp, dtype=int)
    init_path = result.path[indices]

    opt_kwargs = {}
    # Extract path_constraints from planner_kwargs for gradient optimizer
    if "path_constraints" in planner_kwargs:
        opt_kwargs["path_constraints"] = planner_kwargs["path_constraints"]
        opt_kwargs["lambda_path"] = planner_kwargs.get("lambda_path", 10.0)

    opt = GradientOptimizer(sdf_grid, safety_margin=0.6,
                            lambda_obs=15.0, learning_rate=0.015,
                            max_iters=300, **opt_kwargs)
    opt_result = opt.optimize(init_path)
    return smooth_path_bspline(opt_result['path'], 200)


def main():
    obstacles, bounds, resolution, routes = engine_bay_multi_route()
    sdf_grid = build_sdf_2d(obstacles, bounds, resolution)

    # Use routes 0 and 1 for the demonstration
    route1 = routes[0]  # ECU -> Temp Sensor
    route2 = routes[1]  # Battery -> Starter

    print("=" * 60)
    print("Multi-Route Path Constraints Demo")
    print("=" * 60)

    # --- Step 1: Plan reference route (Route 1) ---
    print(f"\n[Route 1] {route1['name']} — planning baseline...")
    path1 = plan_route(sdf_grid, route1["start"], route1["goal"])
    if path1 is None:
        print("  FAILED to plan Route 1. Aborting.")
        return
    stats1 = path_statistics(path1, sdf_grid)
    print(f"  Length={stats1['length']:.2f}, MinClr={stats1['min_clearance']:.3f}")

    # --- Step 2: Plan Route 2 with NO constraint (baseline) ---
    print(f"\n[Route 2] {route2['name']} — no constraint (baseline)...")
    path2_none = plan_route(sdf_grid, route2["start"], route2["goal"])
    if path2_none is not None:
        stats2 = path_statistics(path2_none, sdf_grid)
        print(f"  Length={stats2['length']:.2f}, MinClr={stats2['min_clearance']:.3f}")

    # --- Step 3: Plan Route 2 with ATTRACT constraint ---
    print(f"\n[Route 2] {route2['name']} — attract to Route 1...")
    attract = PathConstraint(
        reference_path=path1, mode="attract",
        influence_radius=1.5, strength=2.0,
    )
    speed_mod_a = build_speed_modifier([attract], sdf_grid)
    path2_attract = plan_route(
        sdf_grid, route2["start"], route2["goal"],
        speed_modifier=speed_mod_a,
        path_constraints=[attract], lambda_path=12.0,
    )
    if path2_attract is not None:
        stats2a = path_statistics(path2_attract, sdf_grid)
        print(f"  Length={stats2a['length']:.2f}, MinClr={stats2a['min_clearance']:.3f}")

    # --- Step 4: Plan Route 2 with REPEL constraint ---
    print(f"\n[Route 2] {route2['name']} — repel from Route 1...")
    repel = PathConstraint(
        reference_path=path1, mode="repel",
        influence_radius=1.5, strength=2.5,
    )
    speed_mod_r = build_speed_modifier([repel], sdf_grid)
    path2_repel = plan_route(
        sdf_grid, route2["start"], route2["goal"],
        speed_modifier=speed_mod_r,
        path_constraints=[repel], lambda_path=12.0,
    )
    if path2_repel is not None:
        stats2r = path_statistics(path2_repel, sdf_grid)
        print(f"  Length={stats2r['length']:.2f}, MinClr={stats2r['min_clearance']:.3f}")

    # --- Visualization: 2x2 ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    X, Y = sdf_grid.meshgrid_world()

    # (0,0) Route 1 baseline
    ax = axes[0, 0]
    draw_base(ax, sdf_grid, obstacles, bounds, "Route 1 (Reference)")
    ax.plot(path1[:, 0], path1[:, 1], color=route1["color"], linewidth=3,
            label=route1["name"], zorder=10)
    ax.plot(*route1["start"], 'o', color=route1["color"], ms=10,
            markeredgecolor='black', zorder=20)
    ax.plot(*route1["goal"], 's', color=route1["color"], ms=10,
            markeredgecolor='black', zorder=20)
    ax.legend(fontsize=8)

    # (0,1) Route 2 — no constraint
    ax = axes[0, 1]
    draw_base(ax, sdf_grid, obstacles, bounds, "Route 2 — No Constraint")
    ax.plot(path1[:, 0], path1[:, 1], color=route1["color"], linewidth=1.5,
            alpha=0.4, label="Route 1 (ref)", zorder=8)
    if path2_none is not None:
        ax.plot(path2_none[:, 0], path2_none[:, 1], color=route2["color"],
                linewidth=3, label=route2["name"], zorder=10)
    ax.plot(*route2["start"], 'o', color=route2["color"], ms=10,
            markeredgecolor='black', zorder=20)
    ax.plot(*route2["goal"], 's', color=route2["color"], ms=10,
            markeredgecolor='black', zorder=20)
    ax.legend(fontsize=8)

    # (1,0) Route 2 — attract
    ax = axes[1, 0]
    draw_base(ax, sdf_grid, obstacles, bounds, "Route 2 — Attract to Route 1")
    # # Show influence zone
    # dist_field = np.minimum(
    #     *[np.abs(sdf_grid.values)] if False else
    #     [np.ones(sdf_grid.shape) * np.inf]
    # )
    from src.harness.path_constraints import compute_path_distance_field
    dist_field = compute_path_distance_field(path1, sdf_grid)
    influence_mask = dist_field < attract.influence_radius
    ax.contourf(X, Y, influence_mask.astype(float), levels=[0.5, 1.5],
                colors=['lime'], alpha=0.15)
    ax.plot(path1[:, 0], path1[:, 1], color=route1["color"], linewidth=1.5,
            alpha=0.4, label="Route 1 (ref)", zorder=8)
    if path2_attract is not None:
        ax.plot(path2_attract[:, 0], path2_attract[:, 1], color='orange',
                linewidth=3, label="Route 2 (attract)", zorder=10)
    if path2_none is not None:
        ax.plot(path2_none[:, 0], path2_none[:, 1], color=route2["color"],
                linewidth=1.5, alpha=0.3, linestyle='--',
                label="Route 2 (no constraint)", zorder=9)
    ax.plot(*route2["start"], 'o', color='orange', ms=10,
            markeredgecolor='black', zorder=20)
    ax.plot(*route2["goal"], 's', color='orange', ms=10,
            markeredgecolor='black', zorder=20)
    ax.legend(fontsize=8)

    # (1,1) Route 2 — repel
    ax = axes[1, 1]
    draw_base(ax, sdf_grid, obstacles, bounds, "Route 2 — Repel from Route 1")
    # Show influence zone
    ax.contourf(X, Y, influence_mask.astype(float), levels=[0.5, 1.5],
                colors=['red'], alpha=0.12)
    ax.plot(path1[:, 0], path1[:, 1], color=route1["color"], linewidth=1.5,
            alpha=0.4, label="Route 1 (ref)", zorder=8)
    if path2_repel is not None:
        ax.plot(path2_repel[:, 0], path2_repel[:, 1], color='magenta',
                linewidth=3, label="Route 2 (repel)", zorder=10)
    if path2_none is not None:
        ax.plot(path2_none[:, 0], path2_none[:, 1], color=route2["color"],
                linewidth=1.5, alpha=0.3, linestyle='--',
                label="Route 2 (no constraint)", zorder=9)
    ax.plot(*route2["start"], 'o', color='magenta', ms=10,
            markeredgecolor='black', zorder=20)
    ax.plot(*route2["goal"], 's', color='magenta', ms=10,
            markeredgecolor='black', zorder=20)
    ax.legend(fontsize=8)

    plt.suptitle("Multi-Route Constraints: Attraction & Repulsion",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("07_multi_route_constraints.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved: 07_multi_route_constraints.png")
    plt.show()


if __name__ == "__main__":
    main()
