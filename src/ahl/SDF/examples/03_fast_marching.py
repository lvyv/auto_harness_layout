"""Example 03: Fast Marching Method Path Planning.

Demonstrates:
- FMM-based path planning using SDF-derived speed field
- Visualization of speed field and travel-time field
- Comparison between A* and FMM paths
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib.pyplot as plt

from src.sdf import build_sdf_2d
from src.planning import AStarPlanner, FastMarchingPlanner
from src.viz import plot_path_on_sdf, plot_comparison
from src.scenarios.simple_2d import scattered_circles, narrow_passage


def main():
    for scenario_name, scenario_func in [("Scattered Circles", scattered_circles),
                                          ("Narrow Passage", narrow_passage)]:
        obstacles, bounds, resolution, start, goal = scenario_func()
        sdf_grid = build_sdf_2d(obstacles, bounds, resolution)

        print(f"\n{'='*50}")
        print(f"Scenario: {scenario_name}")
        print(f"{'='*50}")

        # Plan with A*
        astar = AStarPlanner(safety_margin=1.0, cost_weight=10.0)
        result_astar = astar.plan(sdf_grid, start, goal)
        print(f"A*:  cost={result_astar.cost:.2f}, time={result_astar.time_seconds:.3f}s, "
              f"points={len(result_astar.path)}")

        # Plan with FMM
        fmm = FastMarchingPlanner(safety_margin=1.5, speed_exponent=2.0)
        result_fmm = fmm.plan(sdf_grid, start, goal)
        print(f"FMM: cost={result_fmm.cost:.2f}, time={result_fmm.time_seconds:.3f}s, "
              f"points={len(result_fmm.path)}")

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        # 1. Speed field
        X, Y = sdf_grid.meshgrid_world()
        speed = result_fmm.metadata.get('speed_field')
        if speed is not None:
            cf = axes[0].contourf(X, Y, speed, levels=50, cmap='viridis')
            plt.colorbar(cf, ax=axes[0], label='Speed')
            axes[0].contour(X, Y, sdf_grid.values, levels=[0],
                            colors='red', linewidths=2)
            axes[0].set_title('Speed Field (from SDF)')
            axes[0].set_aspect('equal')

        # 2. Travel-time field
        tt = result_fmm.metadata.get('travel_time_field')
        if tt is not None:
            tt_vis = np.where(np.isinf(tt), np.nan, tt)
            cf = axes[1].contourf(X, Y, tt_vis, levels=50, cmap='plasma')
            plt.colorbar(cf, ax=axes[1], label='Travel Time')
            axes[1].contour(X, Y, sdf_grid.values, levels=[0],
                            colors='white', linewidths=2)
            if result_fmm.success:
                axes[1].plot(result_fmm.path[:, 0], result_fmm.path[:, 1],
                             'lime', linewidth=2)
            axes[1].plot(start[0], start[1], 'g^', markersize=12)
            axes[1].plot(goal[0], goal[1], 'r*', markersize=14)
            axes[1].set_title('Travel-Time Field + FMM Path')
            axes[1].set_aspect('equal')

        # 3. A* vs FMM comparison
        paths = {}
        if result_astar.success:
            paths["A*"] = result_astar
        if result_fmm.success:
            paths["FMM"] = result_fmm
        if paths:
            plot_comparison(sdf_grid, paths, start, goal, title="A* vs FMM")
            # Reuse the figure from plot_comparison for the third panel
            # Instead, draw directly on axes[2]
            axes[2].contourf(X, Y, sdf_grid.values, levels=50,
                             cmap='RdBu', alpha=0.5)
            axes[2].contour(X, Y, sdf_grid.values, levels=[0],
                            colors='black', linewidths=2)
            axes[2].contourf(X, Y, sdf_grid.values, levels=[-1e10, 0],
                             colors='gray', alpha=0.5)
            if result_astar.success:
                axes[2].plot(result_astar.path[:, 0], result_astar.path[:, 1],
                             'lime', linewidth=2.5, label=f"A* (cost={result_astar.cost:.1f})")
            if result_fmm.success:
                axes[2].plot(result_fmm.path[:, 0], result_fmm.path[:, 1],
                             'cyan', linewidth=2.5, label=f"FMM (cost={result_fmm.cost:.1f})")
            axes[2].plot(start[0], start[1], 'g^', markersize=12)
            axes[2].plot(goal[0], goal[1], 'r*', markersize=14)
            axes[2].set_title('A* vs FMM Paths')
            axes[2].set_aspect('equal')
            axes[2].legend()

        plt.suptitle(f"Fast Marching Method - {scenario_name}", fontsize=14)
        plt.tight_layout()

        filename = f"03_fmm_{scenario_name.lower().replace(' ', '_')}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()

    plt.show()


if __name__ == "__main__":
    main()
