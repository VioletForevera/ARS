"""
性能指标模块

平均回报、移动均值、mean_return、CF等
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from collections import deque


class Metrics:
    """
    性能指标跟踪器
    
    跟踪训练过程中的各种性能指标
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.episode_rewards = deque(maxlen=window_size)
        self.episode_steps = deque(maxlen=window_size)
        self.task_performances = {}
        self.catastrophic_forgetting = {}
        
    def record_episode(self, reward: float, steps: int, task_id: int = None):
        """记录回合数据"""
        self.episode_rewards.append(reward)
        self.episode_steps.append(steps)
        
        if task_id is not None:
            if task_id not in self.task_performances:
                self.task_performances[task_id] = []
            self.task_performances[task_id].append(reward)
    
    def get_mean_return(self, window: int = None) -> float:
        """获取平均回报"""
        if window is None:
            window = len(self.episode_rewards)
        
        if window == 0:
            return 0.0
        
        recent_rewards = list(self.episode_rewards)[-window:]
        return np.mean(recent_rewards)
    
    def get_moving_average(self, window: int = 10) -> List[float]:
        """获取移动平均"""
        if len(self.episode_rewards) < window:
            return list(self.episode_rewards)
        
        moving_avg = []
        for i in range(len(self.episode_rewards)):
            start_idx = max(0, i - window + 1)
            window_data = list(self.episode_rewards)[start_idx:i+1]
            moving_avg.append(np.mean(window_data))
        
        return moving_avg
    
    def calculate_cf(self, task_id: int, before_performance: float, 
                    after_performance: float) -> float:
        """计算灾难性遗忘程度"""
        if before_performance == 0:
            return 0.0
        
        cf = max(0, (before_performance - after_performance) / before_performance)
        self.catastrophic_forgetting[task_id] = cf
        return cf
    
    def get_task_performance(self, task_id: int) -> Dict[str, float]:
        """获取任务性能统计"""
        if task_id not in self.task_performances:
            return {}
        
        rewards = self.task_performances[task_id]
        return {
            'mean': np.mean(rewards),
            'std': np.std(rewards),
            'min': np.min(rewards),
            'max': np.max(rewards),
            'count': len(rewards)
        }
    
    def get_cf_summary(self) -> Dict[str, float]:
        """获取灾难性遗忘摘要"""
        return self.catastrophic_forgetting.copy()
    
    def reset(self):
        """重置指标"""
        self.episode_rewards.clear()
        self.episode_steps.clear()
        self.task_performances.clear()
        self.catastrophic_forgetting.clear()

