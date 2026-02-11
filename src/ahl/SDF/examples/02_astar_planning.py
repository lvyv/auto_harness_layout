"""Example 02: A* Path Planning on SDF.

Demonstrates:
- A* path planning using SDF-derived cost field
- Running on multiple scenarios
- Comparing different safety margins
"""

import sys
sys.path.insert(0, ".")

import matplotlib.pyplot as plt

from src.sdf import build_sdf_2d
from src.planning import AStarPlanner
from src.viz import plot_path_on_sdf, plot_comparison
from src.scenarios.simple_2d import scattered_circles, narrow_passage, u_shaped_obstacle


def run_scenario(name, scenario_func):
    """Run A* on a single scenario and display results."""
    obstacles, bounds, resolution, start, goal = scenario_func()

    print(f"\n{'='*50}")
    print(f"Scenario: {name}")
    print(f"{'='*50}")

    # Build SDF
    sdf_grid = build_sdf_2d(obstacles, bounds, resolution)
    print(f"Grid: {sdf_grid.shape}, spacing={sdf_grid.spacing:.4f}")

    # Plan with different safety margins
    results = {}
    for margin in [0.3, 0.8, 1.5]:
        planner = AStarPlanner(safety_margin=margin, cost_weight=10.0)
        result = planner.plan(sdf_grid, start, goal)

        status = "OK" if result.success else "FAILED"
        print(f"  margin={margin:.1f}: {status}, cost={result.cost:.2f}, "
              f"time={result.time_seconds:.3f}s, nodes={result.iterations}")
        if result.success:
            results[f"A* (margin={margin})"] = result

    # Plot comparison
    if results:
        fig, ax = plot_comparison(sdf_grid, results, start, goal,
                                  title=f"{name} - A* with Different Safety Margins")
        return fig
    return None


def main():
    scenarios = [
        ("Scattered Circles", scattered_circles),
        ("Narrow Passage", narrow_passage),
        ("U-Shaped Obstacle", u_shaped_obstacle),
    ]

    figs = []
    for name, func in scenarios:
        fig = run_scenario(name, func)
        if fig:
            figs.append((name, fig))

    # Save all figures
    for name, fig in figs:
        filename = f"02_astar_{name.lower().replace(' ', '_')}.png"
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")

    plt.show()


if __name__ == "__main__":
    main()
