"""Pydantic models for configuration validation."""

from pydantic import BaseModel, Field, field_validator


class GridConfig(BaseModel):
    """Configuration for Grid creation and behavior.

    Attributes:
        width: Grid width (columns)
        height: Grid height (rows)
        default_cell: Default cell type for new cells (0=FREE, 1=OBSTACLE)
    """

    width: int = Field(gt=0, le=1000, description="Grid width in cells")
    height: int = Field(gt=0, le=1000, description="Grid height in cells")
    default_cell: int = Field(default=0, ge=0, le=4, description="Default cell type")

    @field_validator('width', 'height')
    @classmethod
    def validate_size(cls, v: int) -> int:
        """Validate grid size is reasonable."""
        if v > 1000:
            raise ValueError("Grid size must not exceed 1000 for performance reasons")
        return v


class AStarConfig(BaseModel):
    """Configuration for A* path planning algorithm.

    Attributes:
        diagonal_move: Allow diagonal movement
        sdf_weight: Weight for SDF penalty term (higher = stay further from obstacles)
        max_iterations: Maximum iterations to prevent infinite loops
        epsilon: Small value to prevent division by zero in SDF penalty
    """

    diagonal_move: bool = Field(default=False, description="Allow diagonal movement")
    sdf_weight: float = Field(default=0.5, ge=0.0, le=10.0, description="SDF penalty weight")
    max_iterations: int = Field(default=1_000_000, gt=0, description="Maximum iterations")
    epsilon: float = Field(default=0.1, gt=0.0, description="Epsilon for SDF penalty")

    @field_validator('sdf_weight')
    @classmethod
    def validate_sdf_weight(cls, v: float) -> float:
        """Validate SDF weight is non-negative."""
        if v < 0:
            raise ValueError("SDF weight must be non-negative")
        return v
