"""Example 07: Multi-Route Constraints — Attraction & Repulsion.

Pipeline: A* → Gradient Optimization → B-spline Smoothing

Constraints are injected at TWO stages for maximum effect:
  Stage 1 — A* cost_modifier   : steers the discrete grid search
  Stage 2 — GradientOptimizer  : refines the continuous trajectory

Demonstrates:
  1. Plan the first cable normally (baseline)
  2. Attract: second cable reuses the first cable's corridor
  3. Repel:  second cable avoids the first cable's corridor
  4. Comparison visualisation of all three behaviours
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib.pyplot as plt

from src.sdf import build_sdf_2d
from src.planning import AStarPlanner, GradientOptimizer
from src.harness import (smooth_path_bspline, path_statistics,
                         PathConstraint, build_cost_modifier,
                         compute_path_distance_field)
from src.harness.path_constraints import nearest_point_on_path
from src.scenarios.engine_bay import engine_bay_multi_route


# ── helpers ──────────────────────────────────────────────────────────────

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
    xx, yy = sdf_grid.meshgrid_world()
    ax.contourf(xx, yy, sdf_grid.values, levels=50, cmap='RdBu', alpha=0.35)
    ax.contour(xx, yy, sdf_grid.values, levels=[0], colors='black', linewidths=1.2)
    ax.contourf(xx, yy, sdf_grid.values, levels=[-1e10, 0],
                colors='#404040', alpha=0.8)
    draw_obstacle_labels(ax, obstacles)
    ax.set_title(title, fontsize=10)
    ax.set_aspect('equal')
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])


def avg_dist_to_ref(path, ref_path):
    """Mean distance from *path* to *ref_path* polyline."""
    _, dists = nearest_point_on_path(path, ref_path)
    return dists.mean()


# ── planning ─────────────────────────────────────────────────────────────

def plan_route(sdf_grid, start, goal,
               cost_modifier=None, path_constraints=None, lambda_path=12.0):
    """A* → GradientOptimizer → B-spline.

    Args:
        sdf_grid:          SDFGrid.
        start, goal:       (2,) world coordinates.
        cost_modifier:     (ny, nx) multiplier for the A* cost field.
        path_constraints:  list[PathConstraint] for GradientOptimizer.
        lambda_path:       weight of the path-constraint cost term.

    Returns:
        Smoothed (N, 2) path or None on failure.
    """
    # ---- Stage 1: A* with optional cost modifier ----
    astar = AStarPlanner(safety_margin=0.3, cost_weight=12.0)
    astar_kw = {}
    if cost_modifier is not None:
        astar_kw["cost_modifier"] = cost_modifier
    result = astar.plan(sdf_grid, start, goal, **astar_kw)
    if not result.success:
        return None

    # Sub-sample for gradient optimiser
    n_wp = min(80, len(result.path))
    indices = np.linspace(0, len(result.path) - 1, n_wp, dtype=int)
    init_path = result.path[indices]

    # ---- Stage 2: Gradient Optimisation ----
    opt = GradientOptimizer(
        sdf_grid,
        safety_margin=0.6,
        lambda_obs=15.0,
        learning_rate=0.015,
        max_iters=400,
        path_constraints=path_constraints or [],
        lambda_path=lambda_path,
    )
    opt_result = opt.optimize(init_path)

    # ---- Stage 3: B-spline smoothing ----
    # return smooth_path_bspline(opt_result['path'], 200)
    # return opt_result['path']
    return init_path

# ── main ─────────────────────────────────────────────────────────────────

def main():
    obstacles, bounds, resolution, routes = engine_bay_multi_route()
    sdf_grid = build_sdf_2d(obstacles, bounds, resolution)

    route1 = routes[0]   # ECU -> Temp Sensor
    route2 = routes[1]   # Battery -> Starter

    print("=" * 60)
    print("Multi-Route Path Constraints Demo  (A* pipeline)")
    print("=" * 60)

    # ── 1. Reference route ──────────────────────────────────────────────
    print(f"\n[Route 1] {route1['name']} — baseline ...")
    path1 = plan_route(sdf_grid, route1["start"], route1["goal"])
    if path1 is None:
        print("  FAILED — aborting.")
        return
    s1 = path_statistics(path1, sdf_grid)
    print(f"  Length={s1['length']:.2f}  MinClr={s1['min_clearance']:.3f}")

    # ── 2. Route 2 — no constraint (baseline) ──────────────────────────
    print(f"\n[Route 2] {route2['name']} — no constraint ...")
    path2_none = plan_route(sdf_grid, route2["start"], route2["goal"])
    if path2_none is not None:
        s2 = path_statistics(path2_none, sdf_grid)
        d2 = avg_dist_to_ref(path2_none, path1)
        print(f"  Length={s2['length']:.2f}  MinClr={s2['min_clearance']:.3f}"
              f"  AvgDist-R1={d2:.2f}")

    # ── 3. Route 2 — ATTRACT ───────────────────────────────────────────
    print(f"\n[Route 2] {route2['name']} — attract to Route 1 ...")
    attract = PathConstraint(
        reference_path=path1, mode="attract",
        influence_radius=1.5, strength=2.0,
    )
    cost_mod_a = build_cost_modifier([attract], sdf_grid)
    path2_attract = plan_route(
        sdf_grid, route2["start"], route2["goal"],
        cost_modifier=cost_mod_a,
        path_constraints=[attract], lambda_path=12.0,
    )
    if path2_attract is not None:
        s2a = path_statistics(path2_attract, sdf_grid)
        d2a = avg_dist_to_ref(path2_attract, path1)
        print(f"  Length={s2a['length']:.2f}  MinClr={s2a['min_clearance']:.3f}"
              f"  AvgDist-R1={d2a:.2f}")

    # ── 4. Route 2 — REPEL ─────────────────────────────────────────────
    print(f"\n[Route 2] {route2['name']} — repel from Route 1 ...")
    repel = PathConstraint(
        reference_path=path1, mode="repel",
        influence_radius=1.5, strength=2.5,
    )
    cost_mod_r = build_cost_modifier([repel], sdf_grid)
    path2_repel = plan_route(
        sdf_grid, route2["start"], route2["goal"],
        cost_modifier=cost_mod_r,
        path_constraints=[repel], lambda_path=12.0,
    )
    if path2_repel is not None:
        s2r = path_statistics(path2_repel, sdf_grid)
        d2r = avg_dist_to_ref(path2_repel, path1)
        print(f"  Length={s2r['length']:.2f}  MinClr={s2r['min_clearance']:.3f}"
              f"  AvgDist-R1={d2r:.2f}")

    # ── Summary table ──────────────────────────────────────────────────
    print(f"\n{'Scenario':<25} {'Length':>8} {'MinClr':>8} {'AvgDist-R1':>11}")
    print("-" * 55)
    if path2_none is not None:
        print(f"{'No constraint':<25} {s2['length']:>8.2f} "
              f"{s2['min_clearance']:>8.3f} {d2:>11.2f}")
    if path2_attract is not None:
        print(f"{'Attract':<25} {s2a['length']:>8.2f} "
              f"{s2a['min_clearance']:>8.3f} {d2a:>11.2f}")
    if path2_repel is not None:
        print(f"{'Repel':<25} {s2r['length']:>8.2f} "
              f"{s2r['min_clearance']:>8.3f} {d2r:>11.2f}")

    # ── Visualisation 2x2 ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    xx, yy = sdf_grid.meshgrid_world()

    dist_field = compute_path_distance_field(path1, sdf_grid)
    influence_zone = dist_field < attract.influence_radius

    # (0,0) Route 1 reference
    ax = axes[0, 0]
    draw_base(ax, sdf_grid, obstacles, bounds,
              f"Route 1: {route1['name']}  (reference)")
    ax.plot(path1[:, 0], path1[:, 1], color=route1["color"], lw=3,
            label=route1["name"], zorder=10)
    ax.plot(*route1["start"], 'o', color=route1["color"], ms=10,
            markeredgecolor='black', zorder=20)
    ax.plot(*route1["goal"], 's', color=route1["color"], ms=10,
            markeredgecolor='black', zorder=20)
    ax.legend(fontsize=8)

    # (0,1) Route 2 — no constraint
    ax = axes[0, 1]
    draw_base(ax, sdf_grid, obstacles, bounds,
              f"Route 2: {route2['name']}  (no constraint)")
    ax.plot(path1[:, 0], path1[:, 1], color=route1["color"], lw=1.5,
            alpha=0.4, label="Route 1 (ref)", zorder=8)
    if path2_none is not None:
        ax.plot(path2_none[:, 0], path2_none[:, 1], color=route2["color"],
                lw=3, label=route2["name"], zorder=10)
    ax.plot(*route2["start"], 'o', color=route2["color"], ms=10,
            markeredgecolor='black', zorder=20)
    ax.plot(*route2["goal"], 's', color=route2["color"], ms=10,
            markeredgecolor='black', zorder=20)
    ax.legend(fontsize=8)

    # (1,0) Route 2 — attract
    ax = axes[1, 0]
    draw_base(ax, sdf_grid, obstacles, bounds,
              "Route 2 — Attract to Route 1")
    ax.contourf(xx, yy, influence_zone.astype(float), levels=[0.5, 1.5],
                colors=['lime'], alpha=0.15)
    ax.plot(path1[:, 0], path1[:, 1], color=route1["color"], lw=1.5,
            alpha=0.4, label="Route 1 (ref)", zorder=8)
    if path2_attract is not None:
        ax.plot(path2_attract[:, 0], path2_attract[:, 1], color='orange',
                lw=3, label="Route 2 (attract)", zorder=10)
    if path2_none is not None:
        ax.plot(path2_none[:, 0], path2_none[:, 1], color=route2["color"],
                lw=1.5, alpha=0.3, ls='--',
                label="Route 2 (baseline)", zorder=9)
    ax.plot(*route2["start"], 'o', color='orange', ms=10,
            markeredgecolor='black', zorder=20)
    ax.plot(*route2["goal"], 's', color='orange', ms=10,
            markeredgecolor='black', zorder=20)
    ax.legend(fontsize=8)

    # (1,1) Route 2 — repel
    ax = axes[1, 1]
    draw_base(ax, sdf_grid, obstacles, bounds,
              "Route 2 — Repel from Route 1")
    ax.contourf(xx, yy, influence_zone.astype(float), levels=[0.5, 1.5],
                colors=['red'], alpha=0.12)
    ax.plot(path1[:, 0], path1[:, 1], color=route1["color"], lw=1.5,
            alpha=0.4, label="Route 1 (ref)", zorder=8)
    if path2_repel is not None:
        ax.plot(path2_repel[:, 0], path2_repel[:, 1], color='magenta',
                lw=3, label="Route 2 (repel)", zorder=10)
    if path2_none is not None:
        ax.plot(path2_none[:, 0], path2_none[:, 1], color=route2["color"],
                lw=1.5, alpha=0.3, ls='--',
                label="Route 2 (baseline)", zorder=9)
    ax.plot(*route2["start"], 'o', color='magenta', ms=10,
            markeredgecolor='black', zorder=20)
    ax.plot(*route2["goal"], 's', color='magenta', ms=10,
            markeredgecolor='black', zorder=20)
    ax.legend(fontsize=8)

    plt.suptitle("Multi-Route Constraints: Attraction & Repulsion  (A* pipeline)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("07_multi_route_constraints.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved: 07_multi_route_constraints.png")
    plt.show()


if __name__ == "__main__":
    main()
