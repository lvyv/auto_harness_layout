"""健康检查端点。"""

from fastapi import APIRouter

from ahl.api.schemas.response import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """服务健康检查。"""
    return HealthResponse()
