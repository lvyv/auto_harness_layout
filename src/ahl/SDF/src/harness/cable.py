"""Wire cable data model for automotive harness routing."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Cable:
    """Physical cable/wire specification.

    Attributes:
        name: Cable identifier.
        diameter: Outer diameter in meters.
        min_bend_radius: Minimum allowed bend radius in meters.
            Automotive standard: typically 3-6x outer diameter.
        weight_per_meter: Linear weight in kg/m.
        color: Color for visualization.
    """
    name: str
    diameter: float
    min_bend_radius: float
    weight_per_meter: float = 0.0
    color: str = "blue"

    @classmethod
    def automotive(cls, name: str, diameter_mm: float,
                   bend_factor: float = 4.0, color: str = "blue") -> 'Cable':
        """Create a cable with automotive-standard bend radius.

        Args:
            name: Cable identifier.
            diameter_mm: Outer diameter in millimeters.
            bend_factor: Bend radius multiplier (default 4x diameter).
            color: Visualization color.
        """
        d_m = diameter_mm / 1000.0
        return cls(
            name=name,
            diameter=d_m,
            min_bend_radius=d_m * bend_factor,
            weight_per_meter=d_m * 0.5,  # rough estimate
            color=color,
        )

    @classmethod
    def from_awg(cls, awg: int, name: str = "", color: str = "blue") -> 'Cable':
        """Create cable from AWG wire gauge standard.

        Common automotive AWG sizes and approximate outer diameters (mm):
        """
        awg_diameters_mm = {
            8: 8.5, 10: 6.5, 12: 5.5, 14: 4.5, 16: 3.8,
            18: 3.2, 20: 2.8, 22: 2.4, 24: 2.0,
        }
        if awg not in awg_diameters_mm:
            raise ValueError(f"AWG {awg} not in lookup table. "
                             f"Available: {sorted(awg_diameters_mm.keys())}")
        d_mm = awg_diameters_mm[awg]
        if not name:
            name = f"AWG{awg}"
        return cls.automotive(name, d_mm, bend_factor=4.0, color=color)


@dataclass
class HarnessSpec:
    """Specification for a wire harness routing problem.

    Attributes:
        cables: List of cables to route.
        start: Start connector position (x, y).
        goal: End connector position (x, y).
        waypoints: Required pass-through points (clip/clamp locations).
    """
    cables: list
    start: np.ndarray
    goal: np.ndarray
    waypoints: list = None

    def __post_init__(self):
        self.start = np.asarray(self.start, dtype=float)
        self.goal = np.asarray(self.goal, dtype=float)
        if self.waypoints is None:
            self.waypoints = []
        self.waypoints = [np.asarray(w, dtype=float) for w in self.waypoints]

    @property
    def bundle_diameter(self) -> float:
        """Approximate bundle diameter assuming circular packing."""
        if not self.cables:
            return 0.0
        if len(self.cables) == 1:
            return self.cables[0].diameter
        # Rough approximation: sqrt(n) * max_diameter
        max_d = max(c.diameter for c in self.cables)
        return max_d * np.sqrt(len(self.cables))

    @property
    def min_bend_radius(self) -> float:
        """Most restrictive bend radius across all cables in the bundle."""
        if not self.cables:
            return 0.0
        return max(c.min_bend_radius for c in self.cables)
