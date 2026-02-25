"""API 请求数据模型。"""

from typing import List, Optional
from pydantic import BaseModel, Field


class Point3D(BaseModel):
    """三维坐标点。"""
    x: int = Field(..., description="X 索引")
    y: int = Field(..., description="Y 索引")
    z: int = Field(..., description="Z 索引")

    def as_tuple(self) -> tuple:
        return (self.x, self.y, self.z)


class ObstacleRegion(BaseModel):
    """障碍物区域（轴对齐矩形）。"""
    min_point: Point3D
    max_point: Point3D


class GridCreateRequest(BaseModel):
    """创建网格请求。"""
    nx: int = Field(..., gt=0, le=500, description="X 方向格数")
    ny: int = Field(..., gt=0, le=500, description="Y 方向格数")
    nz: int = Field(..., gt=0, le=500, description="Z 方向格数")
    resolution: float = Field(default=1.0, gt=0, description="体素尺寸 (mm)")
    obstacles: Optional[List[ObstacleRegion]] = Field(
        default=None, description="障碍物区域列表"
    )


class RoutingRequest(BaseModel):
    """路由计算请求。"""
    start: Point3D = Field(..., description="起点")
    goal: Point3D = Field(..., description="终点")
    w_sdf: float = Field(default=0.5, ge=0, description="SDF 惩罚权重")
    connectivity: int = Field(default=26, description="邻域连通性 (6 或 26)")
    max_turns: Optional[int] = Field(
        default=None, ge=0, description="最大转弯次数（IP求解器）"
    )
    use_ip: bool = Field(
        default=False, description="是否使用 IP 求解器（而非 A*）"
    )


class BackboneRequest(BaseModel):
    """主干网络构建请求。"""
    terminals: List[Point3D] = Field(..., min_length=2, description="终端点列表")
    n_clusters: Optional[int] = Field(default=None, gt=0, description="聚类数目")
    w_sdf: float = Field(default=0.5, ge=0)
    cost_bias: float = Field(default=0.3, ge=0, le=1)
