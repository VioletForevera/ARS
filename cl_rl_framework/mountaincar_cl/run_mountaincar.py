#!/usr/bin/env python3
"""
MountainCar训练和演示
"""

import gymnasium as gym
import time
import argparse
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
from environments import MountainCarCL, TaskScheduler

class MetricsTracker:
    """指标跟踪器"""
    
    def __init__(self):
        self.task_metrics = {}  # 存储每个任务的指标
        self.convergence_data = {}  # 收敛数据
        self.cf_data = {}  # 灾难性遗忘数据
        
    def record_task_start(self, task_id, task_name):
        """记录任务开始"""
        self.task_metrics[task_id] = {
            'task_name': task_name,
            'episode_rewards': [],
            'convergence_episode': None,
            'final_performance': None,
            'before_performance': None,
            'after_performance': None
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
                avg_reward = np.mean(metrics['episode_rewards'])
                all_avg_rewards.append(avg_reward)
                f.write(f"任务 {task_id} ({metrics['task_name']}): {avg_reward:.2f}\n")
            
            # 添加跨任务平均
            overall_avg = np.mean(all_avg_rewards)
            f.write(f"跨任务平均奖励: {overall_avg:.2f}\n")
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
        avg_rewards = [np.mean(metrics['episode_rewards']) for metrics in self.task_metrics.values()]
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
        
        # 3. 灾难性遗忘热力图
        ax3 = axes[1, 0]
        if self.cf_data:
            # 创建CF矩阵
            cf_matrix = np.zeros((len(task_ids), len(task_ids)))
            cf_labels = []
            
            for cf_key, cf_info in self.cf_data.items():
                task_id = cf_info['task_id']
                new_task_id = cf_info['new_task_id']
                cf_value = cf_info['cf']
                
                if task_id in task_ids and new_task_id in task_ids:
                    i = task_ids.index(task_id)
                    j = task_ids.index(new_task_id)
                    cf_matrix[i, j] = cf_value
            
            im = ax3.imshow(cf_matrix, cmap='Reds', aspect='auto')
            ax3.set_xlabel('Subsequent Task')
            ax3.set_ylabel('Forgotten Task')
            ax3.set_title('Catastrophic Forgetting Heatmap')
            ax3.set_xticks(range(len(task_ids)))
            ax3.set_yticks(range(len(task_ids)))
            ax3.set_xticklabels([f'T{tid}' for tid in task_ids])
            ax3.set_yticklabels([f'T{tid}' for tid in task_ids])
            
            # 添加数值标签
            for i in range(len(task_ids)):
                for j in range(len(task_ids)):
                    if cf_matrix[i, j] > 0:
                        ax3.text(j, i, f'{cf_matrix[i, j]:.2f}', 
                               ha='center', va='center', color='white', fontweight='bold')
            
            plt.colorbar(im, ax=ax3, label='Forgetting Degree')
        else:
            ax3.text(0.5, 0.5, 'No CF Data Available', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Catastrophic Forgetting Heatmap')
        
        # 4. 训练曲线
        ax4 = axes[1, 1]
        for task_id, metrics in self.task_metrics.items():
            rewards = metrics['episode_rewards']
            episodes = list(range(1, len(rewards) + 1))
            ax4.plot(episodes, rewards, alpha=0.7, label=f'T{task_id}', linewidth=2)
        
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Reward')
        ax4.set_title('Training Curves')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'metrics_visualization.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def _generate_table_report(self, save_dir):
        """生成图表报告"""
        # 生成单独的指标图表
        self._generate_convergence_chart(save_dir)
        self._generate_reward_chart(save_dir)
        self._generate_cf_chart(save_dir)
        self._generate_summary_chart(save_dir)
        
    def _generate_convergence_chart(self, save_dir):
        """生成收敛速度图表"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
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
            ax.set_ylim(-10, 100)
            bars = ax.bar(task_ids, conv_speeds, color='lightcoral', alpha=0.7)
            # 在顶部添加"未收敛"文本
            for idx, tid in enumerate(task_ids):
                ax.text(idx, 50, 'Not Converged', ha='center', va='center', 
                        rotation=90, fontsize=10, color='darkred')
        else:
            bars = ax.bar(task_ids, conv_speeds, color='skyblue', alpha=0.7)
            # 如果存在未收敛任务（值为0），设置较宽松的Y轴范围
            if min_conv == 0:
                ax.set_ylim(-max_conv*0.1, max_conv*1.2)
        
        ax.set_xlabel('Task ID')
        ax.set_ylabel('Convergence Episodes')
        ax.set_title('Convergence Speed by Task')
        ax.set_xticks(task_ids)
        ax.set_xticklabels([f'T{tid}' for tid in task_ids])
        
        # 添加数值标签
        for bar, speed in zip(bars, conv_speeds):
            if speed > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                       f'{speed}', ha='center', va='bottom')
            else:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                       'Not Converged', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'convergence_speed.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def _generate_reward_chart(self, save_dir):
        """生成平均回报图表"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        task_ids = []
        avg_rewards = []
        max_rewards = []
        min_rewards = []
        
        for task_id, metrics in self.task_metrics.items():
            task_ids.append(task_id)
            rewards = metrics['episode_rewards']
            avg_rewards.append(np.mean(rewards))
            max_rewards.append(np.max(rewards))
            min_rewards.append(np.min(rewards))
        
        x = np.arange(len(task_ids))
        width = 0.25
        
        bars1 = ax.bar(x - width, avg_rewards, width, label='Average', color='lightgreen', alpha=0.7)
        bars2 = ax.bar(x, max_rewards, width, label='Maximum', color='darkgreen', alpha=0.7)
        bars3 = ax.bar(x + width, min_rewards, width, label='Minimum', color='lightcoral', alpha=0.7)
        
        ax.set_xlabel('Task ID')
        ax.set_ylabel('Reward')
        ax.set_title('Reward Statistics by Task')
        ax.set_xticks(x)
        ax.set_xticklabels([f'T{tid}' for tid in task_ids])
        ax.legend()
        
        # 添加数值标签
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'reward_statistics.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def _generate_cf_chart(self, save_dir):
        """生成灾难性遗忘表格图表"""
        if not self.cf_data:
            return
            
        # 获取所有任务ID并排序
        task_ids = sorted(list(self.task_metrics.keys()))
        n_tasks = len(task_ids)
        
        # 创建表格数据
        table_data = []
        headers = ['Task'] + [f'T{tid}' for tid in task_ids]
        
        for i, task_id in enumerate(task_ids):
            row = [f'T{task_id}']
            
            for j, target_task_id in enumerate(task_ids):
                if j <= i:  # 只显示上三角部分
                    row.append('n/a')
                else:
                    # 查找CF数据
                    cf_value = None
                    for cf_key, cf_info in self.cf_data.items():
                        if (cf_info['task_id'] == task_id and 
                            cf_info['new_task_id'] == target_task_id):
                            cf_value = cf_info['cf']
                            break
                    
                    if cf_value is not None:
                        row.append(f'T{task_id} CF after T{target_task_id}')
                    else:
                        row.append('n/a')
            
            table_data.append(row)
        
        # 创建表格图表
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # 创建表格
        table = ax.table(cellText=table_data, colLabels=headers, 
                       cellLoc='center', loc='center',
                       bbox=[0, 0, 1, 1])
        
        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # 设置表头样式
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # 设置数据行样式
        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                if j == 0:  # 任务列
                    table[(i, j)].set_facecolor('#f1f1f2')
                    table[(i, j)].set_text_props(weight='bold')
                else:
                    if table_data[i-1][j] == 'n/a':
                        table[(i, j)].set_facecolor('#e8e8e8')
                        table[(i, j)].set_text_props(color='gray')
                    else:
                        table[(i, j)].set_facecolor('#fff2cc')
                        table[(i, j)].set_text_props(color='darkred')
        
        plt.title('Catastrophic Forgetting Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'catastrophic_forgetting_table.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 同时生成一个简化的CF值表格
        self._generate_cf_values_table(save_dir, task_ids)
    
    def _generate_cf_values_table(self, save_dir, task_ids):
        """生成CF数值表格"""
        # 创建CF数值表格
        cf_values_data = []
        headers = ['Task'] + [f'T{tid}' for tid in task_ids]
        
        for i, task_id in enumerate(task_ids):
            row = [f'T{task_id}']
            
            for j, target_task_id in enumerate(task_ids):
                if j <= i:  # 只显示上三角部分
                    row.append('n/a')
                else:
                    # 查找CF数据
                    cf_value = None
                    for cf_key, cf_info in self.cf_data.items():
                        if (cf_info['task_id'] == task_id and 
                            cf_info['new_task_id'] == target_task_id):
                            cf_value = cf_info['cf']
                            break
                    
                    if cf_value is not None:
                        row.append(f'{cf_value:.3f}')
                    else:
                        row.append('n/a')
            
            cf_values_data.append(row)
        
        # 创建CF数值表格图表
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # 创建表格
        table = ax.table(cellText=cf_values_data, colLabels=headers, 
                       cellLoc='center', loc='center',
                       bbox=[0, 0, 1, 1])
        
        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # 设置表头样式
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # 设置数据行样式
        for i in range(1, len(cf_values_data) + 1):
            for j in range(len(headers)):
                if j == 0:  # 任务列
                    table[(i, j)].set_facecolor('#f1f1f2')
                    table[(i, j)].set_text_props(weight='bold')
                else:
                    if cf_values_data[i-1][j] == 'n/a':
                        table[(i, j)].set_facecolor('#e8e8e8')
                        table[(i, j)].set_text_props(color='gray')
                    else:
                        # 根据CF值大小设置颜色（基于实际奖励差值，非百分比）
                        cf_val = float(cf_values_data[i-1][j])
                        # 负数表示遗忘（性能下降）
                        if cf_val < -100:
                            table[(i, j)].set_facecolor('#ff3333')  # 深红色 - 严重遗忘
                        elif cf_val < -50:
                            table[(i, j)].set_facecolor('#ff6666')  # 红色 - 高遗忘
                        elif cf_val < -20:
                            table[(i, j)].set_facecolor('#ffff99')  # 黄色 - 中等遗忘
                        elif cf_val < 0:
                            table[(i, j)].set_facecolor('#ccffcc')  # 浅绿 - 轻微遗忘
                        # 正数表示正向迁移（性能提升）
                        elif cf_val < 20:
                            table[(i, j)].set_facecolor('#ccffff')  # 青色 - 轻微提升
                        elif cf_val < 50:
                            table[(i, j)].set_facecolor('#99ccff')  # 浅蓝 - 中等提升
                        else:
                            table[(i, j)].set_facecolor('#6666ff')  # 蓝色 - 显著提升
                        table[(i, j)].set_text_props(color='darkred', weight='bold')
        
        plt.title('Catastrophic Forgetting Values', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'catastrophic_forgetting_values.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def _generate_summary_chart(self, save_dir):
        """生成综合指标图表"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        task_ids = list(self.task_metrics.keys())
        
        # 1. 收敛速度
        conv_speeds = [self.task_metrics[tid]['convergence_episode'] or 0 for tid in task_ids]
        
        # 检查是否有收敛任务
        max_conv = max(conv_speeds) if conv_speeds else 100
        min_conv = min(conv_speeds) if conv_speeds else 0
        
        # 如果没有任何收敛（都是0），设置合适的Y轴范围
        if max_conv == 0:
            ax1.set_ylim(-10, 100)
            ax1.bar(task_ids, conv_speeds, color='lightcoral', alpha=0.7)
            # 在顶部添加"未收敛"文本
            for idx, tid in enumerate(task_ids):
                ax1.text(idx, 50, 'Not Converged', ha='center', va='center', 
                        rotation=90, fontsize=10, color='darkred')
        else:
            ax1.bar(task_ids, conv_speeds, color='skyblue', alpha=0.7)
            # 如果存在未收敛任务（值为0），设置较宽松的Y轴范围
            if min_conv == 0:
                ax1.set_ylim(-max_conv*0.1, max_conv*1.2)
            
        ax1.set_title('Convergence Speed')
        ax1.set_xlabel('Task ID')
        ax1.set_ylabel('Episodes')
        ax1.set_xticks(task_ids)
        ax1.set_xticklabels([f'T{tid}' for tid in task_ids])
        
        # 2. 平均回报
        avg_rewards = [np.mean(self.task_metrics[tid]['episode_rewards']) for tid in task_ids]
        ax2.bar(task_ids, avg_rewards, color='lightgreen', alpha=0.7)
        ax2.set_title('Average Reward')
        ax2.set_xlabel('Task ID')
        ax2.set_ylabel('Reward')
        ax2.set_xticks(task_ids)
        ax2.set_xticklabels([f'T{tid}' for tid in task_ids])
        
        # 3. 训练曲线
        for task_id in task_ids:
            rewards = self.task_metrics[task_id]['episode_rewards']
            episodes = list(range(1, len(rewards) + 1))
            ax3.plot(episodes, rewards, alpha=0.7, label=f'T{task_id}', linewidth=2)
        ax3.set_title('Training Curves')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Reward')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 任务性能对比
        if self.cf_data:
            task_cf_avg = []
            for task_id in task_ids:
                task_cfs = [cf_info for cf_info in self.cf_data.values() if cf_info['task_id'] == task_id]
                avg_cf = np.mean([cf['cf'] for cf in task_cfs]) if task_cfs else 0
                task_cf_avg.append(avg_cf)
            
            # 根据CF值大小选择不同颜色（基于实际差值）
            cf_colors = []
            for cf in task_cf_avg:
                if cf < -100:
                    cf_colors.append('#ff3333')  # 深红 - 严重遗忘
                elif cf < -50:
                    cf_colors.append('#ff6666')  # 红色 - 高遗忘
                elif cf < -20:
                    cf_colors.append('#ffaa66')  # 橙红 - 中等遗忘
                elif cf < 0:
                    cf_colors.append('#ffee99')  # 浅黄 - 轻微遗忘
                elif cf < 20:
                    cf_colors.append('#99ccff')  # 浅蓝 - 轻微提升
                elif cf < 50:
                    cf_colors.append('#6699ff')  # 中蓝 - 中等提升
                else:
                    cf_colors.append('#3366ff')  # 深蓝 - 显著提升
            
            bars = ax4.bar(task_ids, task_cf_avg, color=cf_colors, alpha=0.7)
            ax4.set_title('Average CF by Task')
            ax4.set_xlabel('Task ID')
            ax4.set_ylabel('Average CF')
            ax4.set_xticks(task_ids)
            ax4.set_xticklabels([f'T{tid}' for tid in task_ids])
            
            # 设置合适的Y轴范围（考虑正负值）
            cf_min = min(task_cf_avg) if task_cf_avg else 0
            cf_max = max(task_cf_avg) if task_cf_avg else 0
            # 添加25%的边界
            y_range = max(abs(cf_min), abs(cf_max))
            if y_range > 0:
                ax4.set_ylim(cf_min - y_range*0.3, cf_max + y_range*0.3)
            
            # 在Y=0处画一条线，区分正负
            ax4.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No CF Data Available', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Average CF by Task')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'summary_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()

class DQNNetwork(nn.Module):
    def __init__(self, input_size=2, hidden_size=256, output_size=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),  # 添加dropout防止过拟合
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

def train(episodes=1000, max_steps=200, task_id=0, config_path="config/mountaincar_config.yaml", 
          metrics_tracker=None, before_performance=None):
    """训练模型 - 真正的DQN实现"""
    # 加载配置和任务信息
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 获取任务信息
    task_scheduler = TaskScheduler(config_path)
    task = task_scheduler.get_task_by_id(task_id)
    
    print(f"训练任务 {task_id}: {task['name']}")
    print(f"任务参数: gravity={task['gravity']}, force_mag={task['force_mag']}")
    
    # 初始化指标跟踪
    if metrics_tracker is not None:
        metrics_tracker.record_task_start(task_id, task['name'])
        if before_performance is not None:
            metrics_tracker.record_task_performance(task_id, before_perf=before_performance)
    
    # 创建训练环境
    env_wrapper = MountainCarCL()
    env = env_wrapper.make_env(render=False)
    env_wrapper.set_variant(
        gravity=task['gravity'],
        force_mag=task['force_mag']
    )
    
    # 创建模型和优化器
    model = DQNNetwork()
    target_model = DQNNetwork()  # 目标网络
    target_model.load_state_dict(model.state_dict())  # 初始化目标网络
    optimizer = optim.Adam(model.parameters(), lr=0.0005)  # 降低学习率
    criterion = nn.MSELoss()
    
    # 经验回放缓冲区
    from collections import deque
    replay_buffer = deque(maxlen=10000)  # 减少缓冲区大小，提高效率
    batch_size = 32  # 减少批次大小，提高训练速度
    gamma = 0.99
    
    print("开始训练MountainCar智能体...")
    print(f"训练参数: {episodes}回合, 每回合最多{max_steps}步")
    print("=" * 60)
    
    # 记录训练统计
    episode_rewards = []
    episode_steps = []
    recent_rewards = []
    losses = []
    
    for episode in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(max_steps):
            # 将观察转换为张量
            state = torch.FloatTensor(obs).unsqueeze(0)
            
            # epsilon贪婪策略选择动作 - MountainCar需要更多探索
            epsilon = max(0.05, 1.0 - 0.5 * episode / episodes)  # 从1.0衰减到0.05，保持更多探索
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = model(state).max(1)[1].item()
            
            # 执行动作
            next_obs, reward, terminated, truncated, _ = env.step(action)
            
            # MountainCar改进奖励：鼓励摆动和接近目标
            position, velocity = next_obs
            prev_position, prev_velocity = obs
            
            improved_reward = reward  # 基础奖励 -1
            
            # 位置奖励 - 鼓励向右移动（接近目标）
            if position > prev_position:
                improved_reward += 1.0  # 增加位置奖励
            elif position < prev_position:
                improved_reward -= 0.5  # 轻微惩罚后退
            
            # 速度奖励 - 鼓励获得动能（摆动策略）
            if abs(velocity) > abs(prev_velocity):
                improved_reward += 0.5  # 增加速度奖励
            
            # 高度奖励 - 鼓励爬升 (更细致的奖励设计)
            if position > -0.3:  # 非常接近山顶
                improved_reward += 3.0
            elif position > -0.5:  # 接近山顶
                improved_reward += 2.0
            elif position > -0.8:  # 中等高度
                improved_reward += 1.0
            elif position > -1.0:  # 离开谷底
                improved_reward += 0.5
            elif position > -1.1:  # 轻微离开谷底
                improved_reward += 0.2
                
            # 到达目标的大奖励
            if terminated:
                improved_reward += 100
            
            total_reward += improved_reward
            steps += 1
            
            # 存储经验
            done = terminated or truncated
            replay_buffer.append((obs, action, improved_reward, next_obs, done))
            
            # 训练网络
            if len(replay_buffer) >= batch_size:
                # 采样批次数据 - 最高效的张量创建
                batch_indices = np.random.choice(len(replay_buffer), batch_size, replace=False)
                
                # 直接使用numpy数组，避免Python列表
                states = torch.FloatTensor(np.vstack([replay_buffer[i][0] for i in batch_indices]))
                actions = torch.LongTensor(np.array([replay_buffer[i][1] for i in batch_indices]))
                rewards = torch.FloatTensor(np.array([replay_buffer[i][2] for i in batch_indices]))
                next_states = torch.FloatTensor(np.vstack([replay_buffer[i][3] for i in batch_indices]))
                dones = torch.BoolTensor(np.array([replay_buffer[i][4] for i in batch_indices]))
                
                # 计算当前Q值
                current_q_values = model(states).gather(1, actions.unsqueeze(1))
                
                # 计算目标Q值
                with torch.no_grad():
                    next_q_values = target_model(next_states).max(1)[0]
                    target_q_values = rewards + (gamma * next_q_values * ~dones)
                
                # 计算损失
                loss = criterion(current_q_values.squeeze(), target_q_values)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
            
            if terminated or truncated:
                break
                
            obs = next_obs
        
        # 更新目标网络 - 降低更新频率
        if episode % 20 == 0:
            target_model.load_state_dict(model.state_dict())
        
        # 记录统计信息
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        recent_rewards.append(total_reward)
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)
        
        # 记录指标跟踪
        if metrics_tracker is not None:
            metrics_tracker.record_episode(task_id, episode, total_reward, epsilon)
        
        # 计算平均奖励
        avg_reward = np.mean(recent_rewards)
        avg_loss = np.mean(losses[-100:]) if losses else 0
        
        # 打印详细训练信息
        if episode % 10 == 0 or episode < 20:
            print(f"回合 {episode:4d} | 奖励: {total_reward:6.1f} | 步数: {steps:3d} | "
                  f"ε: {epsilon:.3f} | 平均奖励: {avg_reward:.2f} | 损失: {avg_loss:.4f}")
        elif episode % 50 == 0:
            print(f"回合 {episode:4d} | 奖励: {total_reward:6.1f} | 步数: {steps:3d} | "
                  f"ε: {epsilon:.3f} | 平均奖励: {avg_reward:.2f} | 损失: {avg_loss:.4f}")
        
        # 检查是否达到目标性能（MountainCar目标约-110，提高标准）
        if len(recent_rewards) >= 100 and avg_reward >= -110:
            print(f"在第{episode}回合达到目标性能！平均奖励: {avg_reward:.2f}")
            break
    
    env_wrapper.close()
    
    # 计算最终统计
    final_avg = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
    max_reward = max(episode_rewards)
    success_rate = sum(1 for r in episode_rewards if r >= -110) / len(episode_rewards) * 100
    
    # 记录最终性能到指标跟踪器
    if metrics_tracker is not None:
        metrics_tracker.record_task_performance(task_id, after_perf=final_avg)
    
    print("=" * 60)
    print("训练完成！")
    print(f"最终平均奖励: {final_avg:.2f}")
    print(f"最高奖励: {max_reward:.1f}")
    print(f"成功率: {success_rate:.1f}%")
    print(f"总回合数: {len(episode_rewards)}")
    
    # 保存模型
    save_path = Path("runs") / "trained_model.pth"
    save_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存到: {save_path}")
    
    return save_path, final_avg

def demonstrate(model_path, episodes=3, max_steps=200, delay=0.05, task_id=0, config_path="config/mountaincar_config.yaml"):
    """演示训练好的模型"""
    print("\n开始演示训练好的智能体...")
    print(f"加载模型: {model_path}")
    
    # 加载配置和任务信息
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    task_scheduler = TaskScheduler(config_path)
    task = task_scheduler.get_task_by_id(task_id)
    
    print(f"演示任务 {task_id}: {task['name']}")
    print(f"任务参数: gravity={task['gravity']}, force_mag={task['force_mag']}")
    
    # 加载模型
    model = DQNNetwork()
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        print("模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 创建演示环境
    env_wrapper = MountainCarCL()
    env = env_wrapper.make_env(render=True)
    env_wrapper.set_variant(
        gravity=task['gravity'],
        force_mag=task['force_mag']
    )
    
    if episodes == 1:
        print("开始单次演示...")
        print("观察智能体如何爬上山坡！")
    else:
        print(f"开始{episodes}个回合的演示...")
        print("观察智能体如何爬上山坡！")
    print("=" * 50)
    
    total_demo_rewards = []
    
    for episode in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        
        if episodes == 1:
            print("\n智能体演示中...")
        else:
            print(f"\n回合 {episode + 1}/{episodes}")
        
        for step in range(max_steps):
            # 使用训练好的模型选择动作
            state = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                action = model(state).max(1)[1].item()
            
            # 执行动作
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            
            # 显示当前状态 - 单次演示时更频繁显示
            if episodes == 1:
                if step % 25 == 0:  # 单次演示时每25步显示一次
                    print(f"  步数 {step:3d}: 位置={obs[0]:6.2f}, 速度={obs[1]:6.2f}, 奖励={total_reward:3.0f}")
            else:
                if step % 50 == 0:  # 多次演示时每50步显示一次
                    print(f"  步数 {step:3d}: 位置={obs[0]:6.2f}, 速度={obs[1]:6.2f}, 奖励={total_reward:3.0f}")
            
            time.sleep(delay)
            
            if terminated or truncated:
                break
        
        total_demo_rewards.append(total_reward)
        if episodes == 1:
            print(f"演示完成: 总奖励={total_reward:6.1f}, 步数={steps:3d}")
        else:
            print(f"回合 {episode + 1} 完成: 总奖励={total_reward:6.1f}, 步数={steps:3d}")
        
        if episode < episodes - 1:  # 不是最后一个回合
            print("准备下一回合...")
            time.sleep(2)  # 回合之间暂停2秒
    
    # 演示统计
    avg_demo_reward = np.mean(total_demo_rewards)
    max_demo_reward = max(total_demo_rewards)
    
    print("\n" + "=" * 50)
    print("演示完成！")
    if episodes == 1:
        print(f"演示奖励: {avg_demo_reward:.1f}")
        print(f"演示步数: {total_demo_rewards[0]:.0f}")
    else:
        print(f"平均奖励: {avg_demo_reward:.1f}")
        print(f"最高奖励: {max_demo_reward:.1f}")
        print(f"演示回合数: {episodes}")
    
    env_wrapper.close()
    print("演示结束！")

def evaluate_task_performance(model_path, task_id, config_path, episodes=10):
    """评估任务性能"""
    try:
        # 加载模型
        model = DQNNetwork()
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        # 获取任务信息
        task_scheduler = TaskScheduler(config_path)
        task = task_scheduler.get_task_by_id(task_id)
        
        # 创建评估环境
        env_wrapper = MountainCarCL()
        env = env_wrapper.make_env(render=False)
        env_wrapper.set_variant(
            gravity=task['gravity'],
            force_mag=task['force_mag']
        )
        
        total_rewards = []
        
        for episode in range(episodes):
            obs, _ = env.reset()
            total_reward = 0
            
            for step in range(200):  # 最大200步
                state = torch.FloatTensor(obs).unsqueeze(0)
                with torch.no_grad():
                    action = model(state).max(1)[1].item()
                
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                
                if terminated or truncated:
                    break
            
            total_rewards.append(total_reward)
        
        env_wrapper.close()
        
        # 返回平均性能
        avg_performance = np.mean(total_rewards)
        return avg_performance
        
    except Exception as e:
        print(f"评估任务 {task_id} 性能时出错: {e}")
        return None

def train_continual_learning(task_sequence, episodes_per_task=1000, config_path="config/mountaincar_config.yaml"):
    """持续学习训练 - 支持多任务和指标跟踪"""
    print("开始持续学习训练...")
    print(f"任务序列: {task_sequence}")
    print(f"每个任务训练回合数: {episodes_per_task}")
    print("=" * 60)
    
    # 初始化指标跟踪器
    metrics_tracker = MetricsTracker()
    
    # 创建运行目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"runs/continual_learning_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    
    task_scheduler = TaskScheduler(config_path)
    model_path = None
    previous_task_performance = {}
    
    for i, task_id in enumerate(task_sequence):
        print(f"\n{'='*20} 训练任务 {task_id} {'='*20}")
        
        # 获取任务信息
        task = task_scheduler.get_task_by_id(task_id)
        print(f"任务: {task['name']}")
        print(f"参数: gravity={task['gravity']}, force_mag={task['force_mag']}")
        
        # 训练当前任务
        model_path, final_performance = train(
            episodes=episodes_per_task,
            task_id=task_id,
            config_path=config_path,
            metrics_tracker=metrics_tracker,
            before_performance=previous_task_performance.get(task_id)
        )
        
        # 记录当前任务性能
        previous_task_performance[task_id] = final_performance
        
        # 计算灾难性遗忘（如果有之前的任务）
        if i > 0:
            print(f"\n计算灾难性遗忘...")
            for prev_task_id in task_sequence[:i]:
                # 先评估旧任务的当前性能
                print(f"评估任务 {prev_task_id} 的当前性能...")
                old_performance = evaluate_task_performance(model_path, prev_task_id, config_path)
                
                # 计算CF（使用减法而非百分比）
                if old_performance is not None and previous_task_performance.get(prev_task_id) is not None:
                    # 使用单纯减法：性能变化量 = 新性能 - 旧性能
                    # 正数表示性能提升，负数表示性能下降
                    cf = old_performance - previous_task_performance[prev_task_id]
                    metrics_tracker.cf_data[f"{prev_task_id}_after_{task_id}"] = {
                        'task_id': prev_task_id,
                        'new_task_id': task_id,
                        'before_performance': previous_task_performance[prev_task_id],
                        'after_performance': old_performance,
                        'cf': cf
                    }
                    if cf >= 0:
                        print(f"任务 {prev_task_id} 在训练任务 {task_id} 后的性能变化: +{cf:.2f} (正向迁移)")
                    else:
                        print(f"任务 {prev_task_id} 在训练任务 {task_id} 后的性能变化: {cf:.2f} (灾难性遗忘)")
        
        print(f"任务 {task_id} 训练完成，性能: {final_performance:.2f}")
    
    # 保存指标
    metrics_tracker.save_metrics(run_dir)
    print(f"\n指标报告已保存到: {run_dir}")
    
    return run_dir, metrics_tracker

def main():
    parser = argparse.ArgumentParser(description='MountainCar训练和演示')
    parser.add_argument('--train', action='store_true', help='训练新模型')
    parser.add_argument('--demo', action='store_true', help='演示已训练的模型')
    parser.add_argument('--model-path', type=str, default='runs/trained_model.pth', help='模型文件路径')
    parser.add_argument('--episodes', type=int, default=1, help='演示回合数')
    parser.add_argument('--train-episodes', type=int, default=300, help='训练回合数')
    parser.add_argument('--task-id', type=int, default=0, help='任务ID (0-4)')
    parser.add_argument('--config', type=str, default='config/mountaincar_config.yaml', help='配置文件路径')
    parser.add_argument('--continual', action='store_true', help='持续学习模式')
    parser.add_argument('--task-sequence', type=int, nargs='+', default=[0, 1, 2, 3, 4], help='任务序列')
    parser.add_argument('--episodes-per-task', type=int, default=300, help='每个任务的训练回合数')
    args = parser.parse_args()
    
    print("MountainCar强化学习演示")
    print("=" * 40)
    
    if args.train:
        if args.continual:
            print("开始持续学习模式...")
            run_dir, metrics_tracker = train_continual_learning(
                task_sequence=args.task_sequence,
                episodes_per_task=args.episodes_per_task,
                config_path=args.config
            )
            print(f"\n持续学习训练完成！")
            print(f"结果保存在: {run_dir}")
            print("查看指标报告: metrics_report.txt")
        else:
            print("开始单任务训练模式...")
            model_path, final_performance = train(
                episodes=args.train_episodes, 
                task_id=args.task_id, 
                config_path=args.config
            )
            print(f"\n训练完成！模型已保存到: {model_path}")
            print(f"最终性能: {final_performance:.2f}")
            
            # 询问是否立即演示
            try:
                demo_choice = input("\n是否立即演示训练好的模型？(y/n): ").strip().lower()
                if demo_choice in ['y', 'yes', '是']:
                    demonstrate(model_path, args.episodes, task_id=args.task_id, config_path=args.config)
            except KeyboardInterrupt:
                print("\n再见！")
            
    elif args.demo:
        print("开始演示模式...")
        if not Path(args.model_path).exists():
            print(f"模型文件不存在: {args.model_path}")
            print("请先运行训练: python run_mountaincar.py --train")
            return
        demonstrate(args.model_path, args.episodes, task_id=args.task_id, config_path=args.config)
    else:
        print("请指定操作模式！")
        print("使用方法:")
        print("  单任务训练: python run_mountaincar.py --train --task-id 1")
        print("  持续学习: python run_mountaincar.py --train --continual")
        print("  演示: python run_mountaincar.py --demo --task-id 1")
        print("  自定义持续学习: python run_mountaincar.py --train --continual --task-sequence 0 1 2 --episodes-per-task 500")
        print("  任务列表:")
        print("    0: 标准MountainCar")
        print("    1: T1 - 标准重力+标准推力")
        print("    2: T2 - 增强重力+标准推力")
        print("    3: T3 - 标准重力+增强推力")
        print("    4: T4 - 增强重力+增强推力")
        print("  指标跟踪:")
        print("    - 收敛速度: 记录每个任务达到目标性能的episode数")
        print("    - 平均回报: 记录每个任务的平均奖励")
        print("    - 灾难性遗忘: 计算新任务训练后旧任务性能下降程度")

if __name__ == '__main__':
    main()
