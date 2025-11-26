"""
MountainCar持续学习环境封装
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple, Optional


class MountainCarCL:
    """
    MountainCar持续学习环境
    
    支持不同变体的MountainCar任务：
    - 不同重力系数 (gravity)
    - 不同推力因子 (force_mag)
    """
    
    def __init__(self, render_mode: Optional[str] = None):
        self.render_mode = render_mode
        self.env = None
        self.current_variant = None
        
    def make_env(self, render: bool = False) -> gym.Env:
        """创建环境实例"""
        render_mode = "human" if render else None
        self.env = gym.make("MountainCar-v0", render_mode=render_mode)
        return self.env
    
    def set_variant(self, gravity: float = 0.0025, force_mag: float = 0.001) -> None:
        """设置环境变体参数"""
        if self.env is None:
            raise RuntimeError("请先调用make_env()创建环境")
        
        # 获取底层环境对象
        env_unwrapped = self.env.unwrapped
        
        # 设置参数 - MountainCar-v0使用'force'而不是'force_mag'
        env_unwrapped.gravity = gravity
        env_unwrapped.force = force_mag  # 修正：使用'force'参数名
        
        self.current_variant = {
            'gravity': gravity,
            'force_mag': force_mag
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        if self.env is None:
            raise RuntimeError("请先调用make_env()创建环境")
        return self.env.reset(seed=seed, options=options)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行动作"""
        if self.env is None:
            raise RuntimeError("请先调用make_env()创建环境")
        return self.env.step(action)
    
    def render(self):
        """渲染环境"""
        if self.env is not None:
            return self.env.render()
    
    def close(self):
        """关闭环境"""
        if self.env is not None:
            self.env.close()
            self.env = None
    
    def get_observation_space(self):
        """获取观测空间"""
        if self.env is None:
            raise RuntimeError("请先调用make_env()创建环境")
        return self.env.observation_space
    
    def get_action_space(self):
        """获取动作空间"""
        if self.env is None:
            raise RuntimeError("请先调用make_env()创建环境")
        return self.env.action_space


