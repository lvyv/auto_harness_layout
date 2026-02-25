"""日志配置。

统一的日志工厂，所有模块通过 get_logger(__name__) 获取 logger。
"""

import logging
import sys

_configured = False


def _setup_default():
    """配置默认日志格式（仅在首次调用时执行）。"""
    global _configured
    if _configured:
        return
    _configured = True

    root = logging.getLogger("ahl")
    root.setLevel(logging.INFO)

    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.DEBUG)
        fmt = logging.Formatter(
            "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """获取带 'ahl.' 前缀的 logger。

    Args:
        name: 模块名（通常传 __name__）

    Returns:
        配置好的 Logger 实例
    """
    _setup_default()
    if not name.startswith("ahl"):
        name = f"ahl.{name}"
    return logging.getLogger(name)
