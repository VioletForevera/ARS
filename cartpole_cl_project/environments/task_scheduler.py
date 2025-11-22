"""
任务调度器

从配置构建任务序列（T1→T2…）
"""

from typing import List, Dict, Any, Optional, Tuple
import yaml
import numpy as np


class TaskScheduler:
    """
    任务调度器
    
    从配置文件构建任务序列
    """
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.tasks = []
        self.load_tasks()
    
    def load_tasks(self):
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
    动态场景生成器
    
    根据全局步数自动切换任务参数，实现未知任务边界的持续学习
    """
    
    def __init__(self, task_scheduler: TaskScheduler, steps_per_task: int = 20000,
                 drift_type: str = 'none', drift_slope: float = 0.0, 
                 drift_delta: float = 0.0, drift_amp: float = 0.0, 
                 drift_freq: float = 0.0):
        """
        初始化动态场景
        
        Args:
            task_scheduler: 任务调度器
            steps_per_task: 每个任务持续的训练步数
            drift_type: 漂移类型 ('none', 'progressive', 'abrupt', 'periodic')
            drift_slope: progressive 漂移的每步增量
            drift_delta: abrupt 漂移的突变增量
            drift_amp: periodic 漂移的振幅
            drift_freq: periodic 漂移的频率
        """
        self.task_scheduler = task_scheduler
        self.steps_per_task = steps_per_task
        self.drift_type = drift_type
        self.drift_slope = drift_slope
        self.drift_delta = drift_delta
        self.drift_amp = drift_amp
        self.drift_freq = drift_freq
        
        # 获取所有任务并按ID排序
        self.tasks = sorted(task_scheduler.get_task_sequence(), key=lambda t: t.get('id', 0))
        self.num_tasks = len(self.tasks)
        
        # 当前任务索引
        self.current_task_idx = 0
        
    def get_config(self, global_step: int) -> Tuple[Dict[str, Any], int]:
        """
        根据全局步数获取当前环境配置
        
        Args:
            global_step: 全局训练步数
            
        Returns:
            (config_dict, task_id): 配置字典和任务ID
        """
        # 计算当前应该处于哪个任务
        task_idx = (global_step // self.steps_per_task) % self.num_tasks
        self.current_task_idx = task_idx
        
        # 获取基础任务配置
        base_task = self.tasks[task_idx]
        base_length = base_task['length']
        base_wind = base_task['wind']
        task_id = base_task.get('id', task_idx + 1)
        
        # 计算任务内的相对步数（用于漂移）
        steps_within_task = global_step % self.steps_per_task
        
        # 应用漂移逻辑
        if self.drift_type == 'progressive':
            cur_wind = base_wind + self.drift_slope * steps_within_task
        elif self.drift_type == 'abrupt':
            # 3% 概率突发一次
            if np.random.rand() < 0.03:
                cur_wind = base_wind + self.drift_delta
            else:
                cur_wind = base_wind
        elif self.drift_type == 'periodic':
            cur_wind = base_wind + self.drift_amp * np.sin(2 * np.pi * self.drift_freq * steps_within_task)
        else:
            cur_wind = base_wind
        
        return {
            'length': base_length,
            'wind': cur_wind,
            'task_id': task_id,
            'task_name': base_task.get('name', f'T{task_id}')
        }, task_id
    
    def get_all_task_ids(self) -> List[int]:
        """获取所有任务ID列表"""
        return [task.get('id', i+1) for i, task in enumerate(self.tasks)]


