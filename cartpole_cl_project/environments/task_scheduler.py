"""
任务调度器

从配置构建任务序列（T1→T2…）
"""

from typing import List, Dict, Any
import yaml


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


