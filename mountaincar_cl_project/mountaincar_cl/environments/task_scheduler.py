"""
任务调度器 & 动态场景

从配置构建任务序列，并在完全在线持续学习模式下根据全局步数返回带漂移的环境参数。
"""

from typing import List, Dict, Any, Tuple

import numpy as np
import yaml


class TaskScheduler:
    """
    任务调度器
    
    从配置文件构建任务序列
    """
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.tasks: List[Dict[str, Any]] = []
        self.load_tasks()
    
    def load_tasks(self) -> None:
        """从配置文件加载任务"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        self.tasks = config.get('tasks', [])
    
    def get_task_sequence(self) -> List[Dict[str, Any]]:
        """获取任务序列"""
        return self.tasks
    
    def get_task_by_id(self, task_id: int) -> Dict[str, Any]:
        """根据ID获取任务"""
        for task in self.tasks:
            if task.get('id') == task_id:
                return task
        raise ValueError(f"任务ID {task_id} 不存在")
    
    def get_task_count(self) -> int:
        """获取任务数量"""
        return len(self.tasks)


class DynamicScenario:
    """
    MountainCar 动态场景生成器
    
    根据全局训练步数自动切换任务，并在任务内部注入漂移（未知任务边界）
    """
    
    DEFAULT_ENV_PARAMS = {
        'min_position': -1.2,
        'max_position': 0.6,
        'max_speed': 0.07,
        'goal_position': 0.5,
    }
    
    def __init__(self, task_scheduler: TaskScheduler, steps_per_task: int = 25000,
                 drift_type: str = 'none', drift_slope: float = 0.0, drift_delta: float = 0.0,
                 drift_amp: float = 0.0, drift_freq: float = 0.0):
        self.task_scheduler = task_scheduler
        self.steps_per_task = max(1, steps_per_task)
        self.drift_type = drift_type
        self.drift_slope = drift_slope
        self.drift_delta = drift_delta
        self.drift_amp = drift_amp
        self.drift_freq = drift_freq
        
        self.tasks = sorted(task_scheduler.get_task_sequence(), key=lambda t: t.get('id', 0))
        if not self.tasks:
            raise ValueError("配置文件中未定义任何任务")
        self.num_tasks = len(self.tasks)
    
    def get_config(self, global_step: int) -> Tuple[Dict[str, Any], int]:
        """
        根据全局步数返回当前任务配置（包含漂移后的参数）
        """
        task_idx = (global_step // self.steps_per_task) % self.num_tasks
        base_task = self.tasks[task_idx]
        
        base_force = base_task.get('force_mag', 0.001)
        base_gravity = base_task.get('gravity', 0.0025)
        task_id = base_task.get('id', task_idx + 1)
        task_name = base_task.get('name', f'T{task_id}')
        
        steps_within_task = global_step % self.steps_per_task
        
        cur_force = base_force
        cur_gravity = base_gravity
        
        if self.drift_type == 'progressive':
            # 模拟发动机磨损：推力随时间逐渐减小
            cur_force = max(1e-6, base_force - self.drift_slope * steps_within_task)
        elif self.drift_type == 'abrupt':
            # 以小概率突发事件：推力骤降 + 重力上升
            if np.random.rand() < 0.03:
                cur_force = max(1e-6, base_force - self.drift_delta)
                cur_gravity = base_gravity + self.drift_delta
        elif self.drift_type == 'periodic':
            # 重力呈正弦波动，模拟环境周期性变化
            cur_gravity = base_gravity + self.drift_amp * np.sin(2 * np.pi * self.drift_freq * steps_within_task)
        
        config = {
            'task_id': task_id,
            'task_name': task_name,
            'force_mag': cur_force,
            'gravity': cur_gravity,
        }
        # 补充 MountainCar 其他常量参数，便于上层记录
        for key, value in self.DEFAULT_ENV_PARAMS.items():
            config[key] = base_task.get(key, value)
        
        return config, task_id
    
    def get_all_task_ids(self) -> List[int]:
        """返回所有任务ID"""
        return [task.get('id', idx + 1) for idx, task in enumerate(self.tasks)]
