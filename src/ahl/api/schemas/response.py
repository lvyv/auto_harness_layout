"""API 响应数据模型。"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """健康检查响应。"""
    status: str = "ok"
    version: str = "0.1.0"


class PathResult(BaseModel):
    """路径结果。"""
    path: List[List[int]] = Field(..., description="路径点 [[x,y,z], ...]")
    length: float = Field(..., description="路径长度")
    turns: int = Field(default=0, description="转弯次数")
    method: str = Field(default="astar", description="使用的算法")


class RoutingResponse(BaseModel):
    """路由计算响应。"""
    success: bool
    result: Optional[PathResult] = None
    error: Optional[str] = None


class GridInfoResponse(BaseModel):
    """网格信息响应。"""
    shape: List[int]
    resolution: float
    obstacle_count: int
    free_count: int
    total: int


class BackboneResponse(BaseModel):
    """主干网络构建响应。"""
    success: bool
    backbone_paths: List[PathResult] = Field(default_factory=list)
    branch_paths: List[PathResult] = Field(default_factory=list)
    cluster_centers: List[List[int]] = Field(default_factory=list)
    error: Optional[str] = None
