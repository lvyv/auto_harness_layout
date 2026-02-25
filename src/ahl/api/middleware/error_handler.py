"""统一错误处理中间件。"""

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """全局异常捕获中间件。"""

    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except ValueError as e:
            return JSONResponse(
                status_code=400,
                content={"detail": str(e)},
            )
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"detail": f"服务内部错误: {type(e).__name__}: {e}"},
            )
