"""几何处理端点。"""

from fastapi import APIRouter, HTTPException

from ahl.api.schemas.request import GridCreateRequest
from ahl.api.schemas.response import GridInfoResponse
from ahl.geometry.voxel.grid3d import Grid3D, CellType

router = APIRouter(prefix="/geometry", tags=["geometry"])

# 内存中的网格实例（单例，生产环境应改为会话管理）
_current_grid: Grid3D | None = None


def get_grid() -> Grid3D:
    """获取当前网格，未创建则报错。"""
    if _current_grid is None:
        raise HTTPException(status_code=400, detail="网格尚未创建，请先 POST /geometry/grid")
    return _current_grid


@router.post("/grid", response_model=GridInfoResponse)
async def create_grid(req: GridCreateRequest):
    """创建体素网格。"""
    global _current_grid

    grid = Grid3D(req.nx, req.ny, req.nz, resolution=req.resolution)

    # 设置障碍物
    if req.obstacles:
        for obs in req.obstacles:
            mn = obs.min_point
            mx = obs.max_point
            for i in range(mn.x, mx.x + 1):
                for j in range(mn.y, mx.y + 1):
                    for k in range(mn.z, mx.z + 1):
                        if grid.is_valid(i, j, k):
                            grid.set_cell(i, j, k, CellType.OBSTACLE)

    _current_grid = grid

    return GridInfoResponse(
        shape=list(grid.shape),
        resolution=grid.resolution,
        obstacle_count=grid.obstacle_count(),
        free_count=grid.free_count(),
        total=grid.size,
    )


@router.get("/grid", response_model=GridInfoResponse)
async def get_grid_info():
    """获取当前网格信息。"""
    grid = get_grid()
    return GridInfoResponse(
        shape=list(grid.shape),
        resolution=grid.resolution,
        obstacle_count=grid.obstacle_count(),
        free_count=grid.free_count(),
        total=grid.size,
    )
