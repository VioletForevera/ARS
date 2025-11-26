"""
统一Agent接口
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, List, Tuple


class BaseAgent(ABC):
    """
    统一Agent接口
    
    定义所有智能代理的标准接口
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.state_dim = config.get('state_dim', 4)
        self.action_dim = config.get('action_dim', 2)
        
    @abstractmethod
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """选择动作"""
        pass
    
    @abstractmethod
    def observe(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool) -> None:
        """观察经验"""
        pass
    
    @abstractmethod
    def update(self) -> Dict[str, float]:
        """更新策略"""
        pass
    
    @abstractmethod
    def save(self, filepath: str) -> None:
        """保存模型"""
        pass
    
    @abstractmethod
    def load(self, filepath: str) -> None:
        """加载模型"""
        pass
    
    def evaluate(self, env, episodes: int = 10) -> float:
        """评估智能代理性能"""
        total_rewards = []
        
        for _ in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = self.act(state, training=False)
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
            
            total_rewards.append(total_reward)
        
        return np.mean(total_rewards)
