"""Abstract base class and result type for path planners."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from src.sdf.grid import SDFGrid


@dataclass
class PlanResult:
    """Result of a path planning query."""
    path: np.ndarray          # (N, 2) world coordinates
    cost: float               # Total path cost
    time_seconds: float       # Wall-clock planning time
    success: bool             # Whether a valid path was found
    iterations: int = 0       # Algorithm iterations / nodes expanded
    metadata: dict = field(default_factory=dict)


class BasePlanner(ABC):
    """Abstract base class for all path planners."""

    @abstractmethod
    def plan(self, sdf_grid: SDFGrid, start: np.ndarray,
             goal: np.ndarray, **kwargs) -> PlanResult:
        """Plan a path from start to goal on the given SDF grid.

        Args:
            sdf_grid: The signed distance field.
            start: (2,) world coordinate of the start point.
            goal: (2,) world coordinate of the goal point.

        Returns:
            PlanResult with the planned path and metadata.
        """
        ...
