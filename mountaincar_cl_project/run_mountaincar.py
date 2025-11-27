#!/usr/bin/env python3
"""
MountainCar训练和演示
"""

import gymnasium as gym
import time
import argparse
import collections
from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from mountaincar_cl.environments import MountainCarCL, TaskScheduler, DynamicScenario

class MetricsTracker:
    """指标跟踪器"""
    
    def __init__(self):
        self.task_metrics = {}  # 存储每个任务的指标
        self.convergence_data = {}  # 收敛数据
        self.cf_data = {}  # 灾难性遗忘数据
        self.anytime_performance = {}  # 在线流式评估数据
        
    def record_task_start(self, task_id, task_name):
        """记录任务开始"""
        self.task_metrics[task_id] = {
            'task_name': task_name,
            'episode_rewards': [],
            'convergence_episode': None,
            'final_performance': None,
            'before_performance': None,
            'after_performance': None,
            'egp_triggers': 0,
            'egp_paused_steps': 0
        }
        
    def record_episode(self, task_id, episode, reward, epsilon):
        """记录回合数据"""
        if task_id in self.task_metrics:
            self.task_metrics[task_id]['episode_rewards'].append(reward)
            
            # 检查收敛（连续10个回合平均奖励>=-110，MountainCar目标约-110）
            if len(self.task_metrics[task_id]['episode_rewards']) >= 10:
                recent_avg = np.mean(self.task_metrics[task_id]['episode_rewards'][-10:])
                if recent_avg >= -110 and self.task_metrics[task_id]['convergence_episode'] is None:
                    self.task_metrics[task_id]['convergence_episode'] = episode
                    
    def record_task_performance(self, task_id, before_perf=None, after_perf=None):
        """记录任务性能"""
        if task_id in self.task_metrics:
            if before_perf is not None:
                self.task_metrics[task_id]['before_performance'] = before_perf
            if after_perf is not None:
                self.task_metrics[task_id]['after_performance'] = after_perf
                self.task_metrics[task_id]['final_performance'] = after_perf
    
    def record_anytime_performance(self, global_step: int, task_rewards: Dict[int, float]):
        """
        记录随时性能评估结果
        """
        for task_id, reward in task_rewards.items():
            if task_id not in self.anytime_performance:
                self.anytime_performance[task_id] = []
            self.anytime_performance[task_id].append((global_step, reward))
                
    def calculate_cf(self, task_id, new_task_id):
        """计算灾难性遗忘（使用减法，不使用百分比）"""
        if task_id in self.task_metrics and new_task_id in self.task_metrics:
            before_perf = self.task_metrics[task_id]['before_performance']
            after_perf = self.task_metrics[task_id]['after_performance']
            
            if before_perf is not None and after_perf is not None:
                # 使用单纯减法：性能变化量 = 新性能 - 旧性能
                # 正数表示性能提升，负数表示性能下降
                cf = after_perf - before_perf
                self.cf_data[f'{task_id}_after_{new_task_id}'] = {
                    'task_id': task_id,
                    'new_task_id': new_task_id,
                    'before_performance': before_perf,
                    'after_performance': after_perf,
                    'cf': cf
                }
                return cf
        return 0.0
        
    def get_convergence_speed(self, task_id):
        """获取收敛速度"""
        if task_id in self.task_metrics:
            return self.task_metrics[task_id]['convergence_episode']
        return None
        
    def get_average_reward(self, task_id, episodes=None):
        """获取平均回报"""
        if task_id in self.task_metrics:
            rewards = self.task_metrics[task_id]['episode_rewards']
            if episodes is not None:
                rewards = rewards[:episodes]
            return np.mean(rewards) if rewards else 0.0
        return 0.0
        
    def save_metrics(self, save_dir):
        """保存指标到文件"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存详细指标
        metrics_file = os.path.join(save_dir, 'metrics.json')
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.task_metrics, f, indent=2, ensure_ascii=False)
            
        # 保存CF数据
        cf_file = os.path.join(save_dir, 'catastrophic_forgetting.json')
        with open(cf_file, 'w', encoding='utf-8') as f:
            json.dump(self.cf_data, f, indent=2, ensure_ascii=False)
            
        # 生成报告
        self.generate_report(save_dir)
        
    def generate_report(self, save_dir):
        """生成指标报告和可视化"""
        # 生成文本报告
        self._generate_text_report(save_dir)
        
        # 生成可视化报告
        self._generate_visualization(save_dir)
        
        # 生成表格报告
        self._generate_table_report(save_dir)
        
        # 生成随时性能曲线
        if self.anytime_performance:
            self._generate_anytime_performance_plot(save_dir)
        
    def _generate_text_report(self, save_dir):
        """生成文本报告"""
        report_file = os.path.join(save_dir, 'metrics_report.txt')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("MountainCar持续学习指标报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 收敛速度报告
            f.write("1. 收敛速度 (Convergence Speed)\n")
            f.write("-" * 30 + "\n")
            convergence_speeds = []
            for task_id, metrics in self.task_metrics.items():
                conv_ep = metrics['convergence_episode']
                if conv_ep is not None:
                    f.write(f"任务 {task_id} ({metrics['task_name']}): {conv_ep} episodes\n")
                    convergence_speeds.append(conv_ep)
                else:
                    f.write(f"任务 {task_id} ({metrics['task_name']}): 未收敛\n")
            
            if convergence_speeds:
                avg_conv = np.mean(convergence_speeds)
                f.write(f"平均收敛速度: {avg_conv:.1f} episodes\n")
            else:
                f.write("平均收敛速度: N/A (无任务收敛)\n")
            f.write("\n")
            
            # 平均回报报告
            f.write("2. 平均回报 (Average Reward)\n")
            f.write("-" * 30 + "\n")
            all_avg_rewards = []
            for task_id, metrics in self.task_metrics.items():
                rewards = metrics['episode_rewards']
                if not rewards:
                    avg_reward = -200.0  # 空数据使用默认值
                    f.write(f"任务 {task_id} ({metrics['task_name']}): {avg_reward:.2f} (未开始训练)\n")
                else:
                    avg_reward = np.mean(rewards)
                    all_avg_rewards.append(avg_reward)
                    f.write(f"任务 {task_id} ({metrics['task_name']}): {avg_reward:.2f}\n")
            
            # 添加跨任务平均（只计算有数据的任务）
            if all_avg_rewards:
                overall_avg = np.mean(all_avg_rewards)
                f.write(f"跨任务平均奖励: {overall_avg:.2f}\n")
            else:
                f.write("跨任务平均奖励: N/A (无任务数据)\n")
            f.write("\n")
            
            # 灾难性遗忘报告
            f.write("3. 灾难性遗忘 (Catastrophic Forgetting)\n")
            f.write("-" * 30 + "\n")
            if self.cf_data:
                for cf_key, cf_info in self.cf_data.items():
                    f.write(f"任务 {cf_info['task_id']} 在训练任务 {cf_info['new_task_id']} 后:\n")
                    f.write(f"  训练前性能: {cf_info['before_performance']:.2f}\n")
                    f.write(f"  训练后性能: {cf_info['after_performance']:.2f}\n")
                    cf_val = cf_info['cf']
                    if cf_val >= 0:
                        f.write(f"  性能变化: +{cf_val:.2f} (正向迁移)\n\n")
                    else:
                        f.write(f"  性能变化: {cf_val:.2f} (灾难性遗忘)\n\n")
            else:
                f.write("暂无灾难性遗忘数据\n")
                
    def _generate_visualization(self, save_dir):
        """生成可视化图表"""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('MountainCar Continual Learning Metrics', fontsize=16, fontweight='bold')
        
        # 1. 收敛速度柱状图
        ax1 = axes[0, 0]
        task_ids = []
        conv_speeds = []
        task_names = []
        
        for task_id, metrics in self.task_metrics.items():
            task_ids.append(task_id)
            conv_ep = metrics['convergence_episode']
            conv_speeds.append(conv_ep if conv_ep is not None else 0)
            task_names.append(metrics['task_name'])
        
        # 检查是否有收敛任务
        max_conv = max(conv_speeds) if conv_speeds else 100
        min_conv = min(conv_speeds) if conv_speeds else 0
        
        # 如果没有任何收敛（都是0），设置合适的Y轴范围
        if max_conv == 0:
            ax1.set_ylim(-10, 100)
            bars = ax1.bar(task_ids, conv_speeds, color='lightcoral', alpha=0.7)
            # 在顶部添加"未收敛"文本
            for idx, tid in enumerate(task_ids):
                ax1.text(idx, 50, 'Not Converged', ha='center', va='center', 
                        rotation=90, fontsize=9, color='darkred')
        else:
            bars = ax1.bar(task_ids, conv_speeds, color='skyblue', alpha=0.7)
            # 如果存在未收敛任务（值为0），设置较宽松的Y轴范围
            if min_conv == 0:
                ax1.set_ylim(-max_conv*0.1, max_conv*1.2)
        
        ax1.set_xlabel('Task ID')
        ax1.set_ylabel('Convergence Episodes')
        ax1.set_title('Convergence Speed')
        ax1.set_xticks(task_ids)
        ax1.set_xticklabels([f'T{tid}' for tid in task_ids])
        
        # 添加数值标签
        for bar, speed in zip(bars, conv_speeds):
            if speed > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                        f'{speed}', ha='center', va='bottom')
            else:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                        'Not Converged', ha='center', va='bottom')
        
        # 2. 平均回报柱状图
        ax2 = axes[0, 1]
        avg_rewards = []
        for metrics in self.task_metrics.values():
            rewards = metrics['episode_rewards']
            if not rewards:
                avg_rewards.append(-200.0)  # 空数据使用默认值
            else:
                avg_rewards.append(np.mean(rewards))
        bars2 = ax2.bar(task_ids, avg_rewards, color='lightgreen', alpha=0.7)
        ax2.set_xlabel('Task ID')
        ax2.set_ylabel('Average Reward')
        ax2.set_title('Average Reward')
        ax2.set_xticks(task_ids)
        ax2.set_xticklabels([f'T{tid}' for tid in task_ids])
        
        # 添加数值标签
        for bar, reward in zip(bars2, avg_rewards):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                    f'{reward:.1f}', ha='center', va='bottom')
        
        # 3. 灾难性遗忘热力�


