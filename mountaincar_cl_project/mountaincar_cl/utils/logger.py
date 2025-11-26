"""
日志工具模块

控制台+文件+JSONL日志；run_dir管理
"""

import logging
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional


class Logger:
    """
    日志记录器
    
    支持控制台、文件和JSONL格式日志
    """
    
    def __init__(self, config: Dict[str, Any], run_dir: str = None):
        self.config = config
        self.run_dir = run_dir or self._create_run_dir()
        self.logger = None
        self.jsonl_file = None
        self._setup_logger()
    
    def _create_run_dir(self) -> str:
        """创建运行目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = f"runs/run_{timestamp}"
        os.makedirs(run_dir, exist_ok=True)
        return run_dir
    
    def _setup_logger(self):
        """设置日志记录器"""
        self.logger = logging.getLogger('cartpole_cl')
        self.logger.setLevel(getattr(logging, self.config.get('level', 'INFO')))
        
        # 清除现有处理器
        self.logger.handlers.clear()
        
        # 控制台处理器
        if self.config.get('console', True):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(self.config.get('format', '%(asctime)s - %(levelname)s - %(message)s'))
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # 文件处理器
        if self.config.get('file', True):
            log_file = os.path.join(self.run_dir, 'training.log')
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(self.config.get('format', '%(asctime)s - %(levelname)s - %(message)s'))
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # JSONL文件
        self.jsonl_file = os.path.join(self.run_dir, 'training.jsonl')
    
    def info(self, message: str):
        """记录信息日志"""
        self.logger.info(message)
    
    def debug(self, message: str):
        """记录调试日志"""
        self.logger.debug(message)
    
    def warning(self, message: str):
        """记录警告日志"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """记录错误日志"""
        self.logger.error(message)
    
    def log_episode(self, episode: int, reward: float, steps: int, task_id: int = None):
        """记录回合信息"""
        task_info = f" [任务 {task_id}]" if task_id is not None else ""
        self.info(f"回合 {episode}{task_info}: 回报={reward:.2f}, 步数={steps}")
    
    def log_task_switch(self, from_task: int, to_task: int):
        """记录任务切换"""
        self.info(f"任务切换: {from_task} -> {to_task}")
    
    def log_metrics(self, metrics: Dict[str, Any]):
        """记录指标"""
        self.info(f"指标更新: {metrics}")
    
    def log_jsonl(self, data: Dict[str, Any]):
        """记录JSONL格式日志"""
        if self.jsonl_file:
            with open(self.jsonl_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    def get_run_dir(self) -> str:
        """获取运行目录"""
        return self.run_dir








