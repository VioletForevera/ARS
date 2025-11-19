"""
可视化工具模块

训练曲线、CF柱状图等
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, Any, List, Optional


class Visualizer:
    """
    可视化工具
    
    生成训练曲线、CF柱状图等
    """
    
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.plot_dir = os.path.join(run_dir, 'plots')
        os.makedirs(self.plot_dir, exist_ok=True)
        
        # 设置matplotlib样式
        plt.style.use('seaborn-v0_8')
    
    def plot_training_curves(self, metrics_data: Dict[str, List[float]], 
                           save_path: Optional[str] = None) -> None:
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 回报曲线
        for task_id, rewards in metrics_data.items():
            episodes = list(range(len(rewards)))
            axes[0, 0].plot(episodes, rewards, alpha=0.7, label=f'任务 {task_id}')
        
        axes[0, 0].set_xlabel('回合数')
        axes[0, 0].set_ylabel('回报')
        axes[0, 0].set_title('训练曲线 - 回报')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 移动平均曲线
        for task_id, rewards in metrics_data.items():
            moving_avg = self._calculate_moving_average(rewards, window=10)
            episodes = list(range(len(moving_avg)))
            axes[0, 1].plot(episodes, moving_avg, alpha=0.7, label=f'任务 {task_id}')
        
        axes[0, 1].set_xlabel('回合数')
        axes[0, 1].set_ylabel('移动平均回报')
        axes[0, 1].set_title('训练曲线 - 移动平均')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 回报分布
        all_rewards = []
        for rewards in metrics_data.values():
            all_rewards.extend(rewards)
        
        if all_rewards:
            axes[1, 0].hist(all_rewards, bins=30, alpha=0.7, edgecolor='black')
            axes[1, 0].set_xlabel('回报')
            axes[1, 0].set_ylabel('频次')
            axes[1, 0].set_title('回报分布')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 性能统计
        axes[1, 1].text(0.1, 0.8, f'总任务数: {len(metrics_data)}', transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.7, f'总回合数: {sum(len(rewards) for rewards in metrics_data.values())}', transform=axes[1, 1].transAxes)
        if all_rewards:
            axes[1, 1].text(0.1, 0.6, f'平均回报: {np.mean(all_rewards):.2f}', transform=axes[1, 1].transAxes)
            axes[1, 1].text(0.1, 0.5, f'最佳回报: {np.max(all_rewards):.2f}', transform=axes[1, 1].transAxes)
            axes[1, 1].text(0.1, 0.4, f'最差回报: {np.min(all_rewards):.2f}', transform=axes[1, 1].transAxes)
        
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_title('性能统计')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.plot_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_cf_analysis(self, cf_data: Dict[str, float], 
                        save_path: Optional[str] = None) -> None:
        """绘制灾难性遗忘分析图"""
        if not cf_data:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # CF柱状图
        tasks = list(cf_data.keys())
        cf_values = list(cf_data.values())
        
        bars = axes[0].bar(tasks, cf_values, alpha=0.7, color='red')
        axes[0].set_xlabel('任务')
        axes[0].set_ylabel('灾难性遗忘程度')
        axes[0].set_title('灾难性遗忘分析')
        axes[0].grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars, cf_values):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        
        # CF统计
        axes[1].text(0.1, 0.8, f'总任务数: {len(cf_data)}', transform=axes[1].transAxes)
        axes[1].text(0.1, 0.7, f'平均CF: {np.mean(cf_values):.3f}', transform=axes[1].transAxes)
        axes[1].text(0.1, 0.6, f'最大CF: {np.max(cf_values):.3f}', transform=axes[1].transAxes)
        axes[1].text(0.1, 0.5, f'最小CF: {np.min(cf_values):.3f}', transform=axes[1].transAxes)
        axes[1].text(0.1, 0.4, f'CF标准差: {np.std(cf_values):.3f}', transform=axes[1].transAxes)
        
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)
        axes[1].set_title('CF统计')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.plot_dir, 'cf_analysis.png'), dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def _calculate_moving_average(self, data: List[float], window: int = 10) -> List[float]:
        """计算移动平均"""
        if len(data) < window:
            return data
        
        moving_avg = []
        for i in range(len(data)):
            start_idx = max(0, i - window + 1)
            window_data = data[start_idx:i+1]
            moving_avg.append(np.mean(window_data))
        
        return moving_avg








