"""FastAPI 应用实例。"""

from fastapi import FastAPI

from ahl.api.routes import health, geometry, routing
from ahl.api.middleware.error_handler import ErrorHandlerMiddleware


def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用。"""
    app = FastAPI(
        title="AHL - Auto Harness Layout",
        description="电气线束自动化布局 API",
        version="0.1.0",
    )

    # 中间件
    app.add_middleware(ErrorHandlerMiddleware)

    # 路由
    app.include_router(health.router)
    app.include_router(geometry.router)
    app.include_router(routing.router)

    return app


app = create_app()
