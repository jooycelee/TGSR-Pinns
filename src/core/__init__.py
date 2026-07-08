"""
PINN V2 Core Module 核心模块导出

本模块包含训练器和配置加载器。
"""

from .trainer import Trainer
from .config import load_config

__all__ = [
    "Trainer",
    "load_config",
]
