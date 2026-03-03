"""日志工具"""

import os
import logging
from typing import Optional


_logger = None


def setup_logging(
    name: str = "TreeMatchRL",
    log_dir: Optional[str] = None,
    rank: int = 0,
) -> logging.Logger:
    """设置日志

    Args:
        name: logger 名称
        log_dir: 日志文件目录 (None = 仅输出到 stdout)
        rank: 当前进程 rank

    Returns:
        logger: 配置好的 logger
    """
    global _logger
    _logger = logging.getLogger(name)
    _logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)

    # 清除已有 handler
    _logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台输出
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    _logger.addHandler(console)

    # 文件输出
    if log_dir and rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, "training.log"))
        fh.setFormatter(formatter)
        _logger.addHandler(fh)

    return _logger


def main_print(*args, **kwargs):
    """仅在主进程打印

    如果 logger 已初始化则用 logger, 否则直接 print。
    """
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        if _logger is not None:
            _logger.info(" ".join(str(a) for a in args))
        else:
            print(*args, **kwargs)
