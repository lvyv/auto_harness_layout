"""Build an SDFGrid from obstacle definitions."""

import numpy as np

from .primitives import sdf_circle, sdf_rectangle, sdf_line_segment
from .composite import sdf_union
from .grid import SDFGrid


_SHAPE_FUNCS = {
    "circle": lambda pts, obs: sdf_circle(pts, obs["center"], obs["radius"]),
    "rectangle": lambda pts, obs: sdf_rectangle(
        pts, obs["center"], obs["half_extents"]
    ),
    "line_segment": lambda pts, obs: sdf_line_segment(
        pts, obs["a"], obs["b"], obs.get("thickness", 0.0)
    ),
}


def build_sdf_2d(
    obstacles: list[dict],
    bounds: tuple[tuple[float, float], tuple[float, float]],
    resolution: float,
) -> SDFGrid:
    """Build a 2D SDF from a list of obstacle definitions.

    Args:
        obstacles: List of obstacle dicts. Each must have a "type" key and
            shape-specific parameters. Supported types:
              - {"type": "circle", "center": [x, y], "radius": r}
              - {"type": "rectangle", "center": [x, y], "half_extents": [hx, hy]}
              - {"type": "line_segment", "a": [x1, y1], "b": [x2, y2],
                 "thickness": t}
        bounds: ((xmin, xmax), (ymin, ymax)) world-space bounding box.
        resolution: Grid cell size.

    Returns:
        SDFGrid with signed distance values. Negative inside obstacles.
    """
    (xmin, xmax), (ymin, ymax) = bounds
    nx = int(np.ceil((xmax - xmin) / resolution)) + 1
    ny = int(np.ceil((ymax - ymin) / resolution)) + 1

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys)  # (ny, nx) each
    points = np.stack([X.ravel(), Y.ravel()], axis=-1)  # (ny*nx, 2)

    # Evaluate each obstacle's SDF
    sdf_arrays = []
    for obs in obstacles:
        shape_type = obs["type"]
        if shape_type not in _SHAPE_FUNCS:
            raise ValueError(f"Unknown obstacle type: {shape_type}")
        sdf_vals = _SHAPE_FUNCS[shape_type](points, obs)
        sdf_arrays.append(sdf_vals.reshape(ny, nx))

    if not sdf_arrays:
        # No obstacles: distance is effectively infinite everywhere
        combined = np.full((ny, nx), float("inf"))
    else:
        # Union of all obstacles (min = closest obstacle surface)
        combined = sdf_union(*sdf_arrays)

    origin = np.array([xmin, ymin])
    spacing = (xmax - xmin) / (nx - 1) if nx > 1 else resolution
    return SDFGrid(values=combined, origin=origin, spacing=spacing)
