"""
策略熵触发机制（占位）

默认禁用，预留接口
"""

import numpy as np
from typing import Dict, Any, List


class EntropyTrigger:
    """
    策略熵触发机制（占位）
    
    默认禁用，预留接口用于策略熵触发暂停/切换机制
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.enabled = config.get('enabled', False)
        self.threshold = config.get('threshold', 0.1)
        self.history = []
        
    def monitor_entropy(self, action_probs: np.ndarray) -> bool:
        """监控策略熵"""
        if not self.enabled:
            return False
        
        # 计算熵
        entropy = -np.sum(action_probs * np.log(action_probs + 1e-8))
        self.history.append(entropy)
        
        # 检查是否触发
        if len(self.history) > 10:
            recent_entropy = np.mean(self.history[-10:])
            return recent_entropy > self.threshold
        
        return False
    
    def reset(self):
        """重置监控"""
        self.history.clear()

