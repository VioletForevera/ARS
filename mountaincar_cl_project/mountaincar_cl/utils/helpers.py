"""
辅助工具模块

读YAML、随机种子、路径/时间戳run目录
"""

import yaml
import os
import random
import numpy as np
import torch
from datetime import datetime
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def set_random_seed(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_run_dir(base_dir: str = "runs") -> str:
    """创建运行目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def ensure_dir(directory: str) -> None:
    """确保目录存在"""
    os.makedirs(directory, exist_ok=True)


def get_timestamp() -> str:
    """获取时间戳"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_config(config: Dict[str, Any], filepath: str) -> None:
    """保存配置文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)








