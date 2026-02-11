"""SDFGrid: core data structure for discretized 2D signed distance fields."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates


@dataclass
class SDFGrid:
    """Container for a discretized 2D SDF on a regular grid.

    Attributes:
        values: 2D array of signed distance values, shape (ny, nx).
                Convention: values[row, col] where row is y-axis, col is x-axis.
        origin: World-space coordinate of the grid corner (xmin, ymin).
        spacing: Uniform grid cell size in world units.
    """
    values: np.ndarray
    origin: np.ndarray
    spacing: float

    def __post_init__(self):
        self.origin = np.asarray(self.origin, dtype=float)

    @property
    def shape(self) -> tuple:
        return self.values.shape

    @property
    def ny(self) -> int:
        return self.values.shape[0]

    @property
    def nx(self) -> int:
        return self.values.shape[1]

    @property
    def bounds(self) -> tuple:
        """Return ((xmin, xmax), (ymin, ymax)) in world coordinates."""
        xmin, ymin = self.origin
        xmax = xmin + (self.nx - 1) * self.spacing
        ymax = ymin + (self.ny - 1) * self.spacing
        return ((xmin, xmax), (ymin, ymax))

    def world_to_grid(self, world_points: np.ndarray) -> np.ndarray:
        """Convert world coordinates (x, y) to grid indices (col, row)."""
        world_points = np.asarray(world_points, dtype=float)
        return (world_points - self.origin) / self.spacing

    def grid_to_world(self, grid_indices: np.ndarray) -> np.ndarray:
        """Convert grid indices (col, row) to world coordinates (x, y)."""
        grid_indices = np.asarray(grid_indices, dtype=float)
        return grid_indices * self.spacing + self.origin

    def sample(self, world_points: np.ndarray) -> np.ndarray:
        """Bilinear interpolation of SDF values at arbitrary world points.

        Args:
            world_points: (N, 2) array of (x, y) world coordinates.

        Returns:
            (N,) array of interpolated SDF values.
        """
        world_points = np.atleast_2d(world_points)
        grid_pts = self.world_to_grid(world_points)
        # map_coordinates expects (row_coords, col_coords)
        row_coords = grid_pts[:, 1]
        col_coords = grid_pts[:, 0]
        result = map_coordinates(self.values, [row_coords, col_coords],
                                 order=1, mode='nearest')
        return result

    def gradient(self, world_points: np.ndarray = None) -> np.ndarray:
        """Compute SDF gradient.

        If world_points is None, returns the full gradient field as (2, ny, nx)
        where [0] is dSDF/dx and [1] is dSDF/dy.

        If world_points is provided, returns interpolated gradients at those points
        as (N, 2) array of (dSDF/dx, dSDF/dy).
        """
        # Compute gradient on the full grid using central differences
        # np.gradient returns [d/dy (row), d/dx (col)]
        gy, gx = np.gradient(self.values, self.spacing)

        if world_points is None:
            return np.stack([gx, gy], axis=0)  # (2, ny, nx)

        # Interpolate at requested points
        world_points = np.atleast_2d(world_points)
        grid_pts = self.world_to_grid(world_points)
        row_coords = grid_pts[:, 1]
        col_coords = grid_pts[:, 0]
        grad_x = map_coordinates(gx, [row_coords, col_coords],
                                 order=1, mode='nearest')
        grad_y = map_coordinates(gy, [row_coords, col_coords],
                                 order=1, mode='nearest')
        return np.column_stack([grad_x, grad_y])

    def cost_field(self, safety_margin: float, alpha: float = 10.0) -> np.ndarray:
        """Convert SDF to a traversal cost field for path planning.

        Cost logic:
          - sdf <= 0:              INF (inside obstacle, impassable)
          - 0 < sdf <= margin:     1 + alpha * ((margin - sdf) / margin)^2
          - sdf > margin:          1.0 (free space, base cost)

        Args:
            safety_margin: Distance threshold for obstacle proximity penalty.
            alpha: Weight of the proximity penalty.

        Returns:
            (ny, nx) array of traversal costs.
        """
        cost = np.ones_like(self.values)
        # Proximity penalty zone
        mask_near = (self.values > 0) & (self.values <= safety_margin)
        penalty = ((safety_margin - self.values[mask_near]) / safety_margin) ** 2
        cost[mask_near] = 1.0 + alpha * penalty
        # Inside obstacles
        cost[self.values <= 0] = np.inf
        return cost

    def meshgrid_world(self) -> tuple:
        """Return meshgrid arrays (X, Y) of world coordinates.

        Useful for plotting with contourf/contour.
        """
        xs = self.origin[0] + np.arange(self.nx) * self.spacing
        ys = self.origin[1] + np.arange(self.ny) * self.spacing
        return np.meshgrid(xs, ys)
