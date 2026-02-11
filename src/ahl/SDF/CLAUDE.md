# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SDF-based 2D path planning framework for automotive wire harness routing. Uses Signed Distance Fields to represent obstacles and plans cable routes that satisfy automotive constraints (bend radius, clearance, cable gauge).

Python 3.10+. No linter configured.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt
# or editable install
pip install -e .

# Run all tests
python -m pytest tests/ -v

# Run a single test file or class
python -m pytest tests/test_sdf.py -v
python -m pytest tests/test_sdf.py::TestCircleSDF -v

# Run examples (each produces matplotlib plots)
python examples/01_sdf_basics.py
python examples/06_engine_bay.py
```

## Architecture

The pipeline flows: **Obstacles → SDF → Cost Field → Path Planning → Constraint Checking / Smoothing → Visualization**

All source lives under `src/` with these modules:

### `src/sdf/` — Signed Distance Field core
- **primitives.py**: Analytical SDFs (circle, rectangle, line segment) operating on `(N,2)` point arrays
- **composite.py**: Boolean CSG operations (union=min, intersection=max, difference) and smooth blending
- **grid.py**: `SDFGrid` — the central data structure. Discretized SDF on a regular 2D grid with world↔grid coordinate conversion, bilinear interpolation via `sample()`, gradient computation, and `cost_field()` generation for planners
- **from_obstacles.py**: `build_sdf_2d()` factory — takes obstacle dicts + bounds + resolution, returns `SDFGrid`

### `src/planning/` — Path planning algorithms
All planners implement `BasePlanner.plan(sdf_grid, start, goal) → PlanResult`.

- **astar.py**: `AStarPlanner` — 8-connected grid search using SDF-derived cost field
- **fast_marching.py**: `FastMarchingPlanner` — wave-front propagation via scikit-fmm, traces path by gradient descent on travel-time field. Exposes `travel_time_field` and `speed_field` in result metadata
- **gradient.py**: `GradientOptimizer` — CHOMP-style trajectory refinement minimizing smoothness + obstacle cost. Unlike the planners, call `optimizer.optimize(initial_path)` which returns a dict (not `PlanResult`)

### `src/harness/` — Wire harness domain
- **cable.py**: `Cable` dataclass with factory methods (`Cable.automotive()`, `Cable.from_awg()`). `HarnessSpec` bundles cables + start/goal/waypoints
- **constraints.py**: Discrete Menger curvature, bend radius checking, SDF-based clearance verification, `path_statistics()` for comprehensive metrics
- **smoothing.py**: B-spline fitting and iterative `smooth_with_bend_constraint()` that displaces path points to satisfy minimum bend radius while maintaining clearance

### `src/scenarios/` — Predefined test environments
- **simple_2d.py**: Three basic scenarios (scattered circles, narrow passage, U-shaped obstacle)
- **engine_bay.py**: Realistic automotive engine bay with 9 components, waypoints, and multi-route definitions

### `src/viz/` — Matplotlib visualization
- **sdf_plot.py**: SDF contour plots, cost field, gradient quiver plots
- **path_plot.py**: Path overlay on SDF, curvature color-coding, algorithm comparison plots

## Key Conventions

- SDF convention: **negative inside obstacles, positive in free space**
- All paths are `np.ndarray` with shape `(N, 2)` in world coordinates
- Tests use `sys.path.insert(0, ".")` and must be run from the project root
- Obstacle definitions are dicts with a `"type"` key (`"circle"`, `"rectangle"`, `"line_segment"`) plus type-specific parameters
- `GradientOptimizer` has a different interface than the other planners — it takes `sdf_grid` in its constructor and returns a dict from `optimize()`, not a `PlanResult`
