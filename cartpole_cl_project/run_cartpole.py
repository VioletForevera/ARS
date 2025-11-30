#!/usr/bin/env python3
"""
CartPole训练和演示
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
from typing import Dict
from environments import CartPoleCL, TaskScheduler, DynamicScenario
import collections
import random

class MetricsTracker:
    """指标跟踪器"""
    
    def __init__(self):
        self.task_metrics = {}  # 存储每个任务的指标
        self.convergence_data = {}  # 收敛数据
        self.cf_data = {}  # 灾难性遗忘数据
        self.anytime_performance = {}  # 随时性能曲线 {task_id: [(step, reward), ...]}
        
    def record_task_start(self, task_id, task_name):
        """Record task start"""
        self.task_metrics[task_id] = {
            'task_name': task_name,
            'episode_rewards': [],
            'convergence_episode': None,
            'final_performance': None,
            'before_performance': None,
            'after_performance': None,
            'cumulative_regret': 0.0,
            'ewma_reward': 0.0,
            'smoothness_tv': 0.0,
            '_last_reward': None
        }
        
    def record_episode(self, task_id, episode, reward, epsilon):
        """Record round data"""
        if task_id in self.task_metrics:
            self.task_metrics[task_id]['episode_rewards'].append(reward)
            
            # Check convergence（连续10个回合平均奖励>=200）
            if len(self.task_metrics[task_id]['episode_rewards']) >= 10:
                recent_avg = np.mean(self.task_metrics[task_id]['episode_rewards'][-10:])
                if recent_avg >= 200 and self.task_metrics[task_id]['convergence_episode'] is None:
                    self.task_metrics[task_id]['convergence_episode'] = episode
                    
            # === 动态遗憾，平滑度，EWMA ===
            m = self.task_metrics[task_id]
            m['cumulative_regret'] += (500 - reward)  # 如 max_steps 变量有变，可传参
            if m['_last_reward'] is not None:
                m['smoothness_tv'] += abs(reward - m['_last_reward'])
            m['_last_reward'] = reward
            alpha = 0.1
            m['ewma_reward'] = alpha * reward + (1 - alpha) * m['ewma_reward']
                
    def record_task_performance(self, task_id, before_perf=None, after_perf=None):
        """Record task performance"""
        if task_id in self.task_metrics:
            if before_perf is not None:
                self.task_metrics[task_id]['before_performance'] = before_perf
            if after_perf is not None:
                self.task_metrics[task_id]['after_performance'] = after_perf
                self.task_metrics[task_id]['final_performance'] = after_perf
    
    def record_anytime_performance(self, global_step: int, task_rewards: Dict[int, float]):
        """
        记录随时性能评估结果
        
        Args:
            global_step: 全局训练步数
            task_rewards: {task_id: average_reward} 字典
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
            f.write("CartPole持续学习指标报告\n")
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
                avg_reward = np.mean(rewards) if rewards else 0.0
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
                
            # 熵门控统计 (EGP)
            f.write("4. 熵门控统计 (EGP)\n")
            f.write("-" * 30 + "\n")
            for task_id, metrics in self.task_metrics.items():
                trg = metrics.get('egp_triggers', 0)
                pst = metrics.get('egp_paused_steps', 0)
                f.write(f"任务 {task_id} ({metrics.get('task_name','')}): 触发次数={trg}, 暂停步数={pst}\n")
            f.write("\n")
                
            # 动态遗憾报告
            f.write("5. 动态遗憾 (Dynamic Regret)\n")
            f.write("-" * 30 + "\n")
            for task_id, metrics in self.task_metrics.items():
                regret_avg = metrics.get('cumulative_regret', 0.0) / max(1, len(metrics['episode_rewards']))
                smooth = metrics.get('smoothness_tv', 0.0)
                ewma = metrics.get('ewma_reward', 0.0)
                f.write(f"任务 {task_id} ({metrics['task_name']}):\n")
                f.write(f"  平均动态遗憾: {regret_avg:.2f}\n")
                f.write(f"  奖励曲线总变差(平滑度): {smooth:.2f}\n")
                f.write(f"  最终奖励EWMA: {ewma:.2f}\n")
            f.write("\n")
                
    def _generate_visualization(self, save_dir):
        """生成可视化图表"""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('CartPole Continual Learning Metrics', fontsize=16, fontweight='bold')
        
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
        avg_rewards = [np.mean(metrics['episode_rewards']) if metrics['episode_rewards'] else 0.0 for metrics in self.task_metrics.values()]
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
            # 对称色轴&标注（新实现）
            v = np.nanmax(np.abs(cf_matrix)) if cf_matrix.size else 1.0
            im = ax3.imshow(cf_matrix, cmap='coolwarm', vmin=-v, vmax=v, aspect='auto')
            ax3.set_xlabel('Subsequent Task')
            ax3.set_ylabel('Forgotten Task')
            ax3.set_title('Catastrophic Forgetting Heatmap')
            ax3.set_xticks(range(len(task_ids)))
            ax3.set_yticks(range(len(task_ids)))
            ax3.set_xticklabels([f'T{tid}' for tid in task_ids])
            ax3.set_yticklabels([f'T{tid}' for tid in task_ids])
            # 数值标注，正负均标出
            for i in range(len(task_ids)):
                for j in range(len(task_ids)):
                    val = cf_matrix[i, j]
                    if abs(val) > 1e-9:
                        txt_color = 'white' if abs(val) > 0.6 * v else 'black'
                        ax3.text(j, i, f'{val:.2f}', ha='center', va='center', color=txt_color, fontweight='bold')
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
            
            # [修复] 检查列表是否为空
            if len(rewards) > 0:
                avg_rewards.append(np.mean(rewards))
                max_rewards.append(np.max(rewards))
                min_rewards.append(np.min(rewards))
            else:
                # 如果没有数据，填充默认值 0
                avg_rewards.append(0.0)
                max_rewards.append(0.0)
                min_rewards.append(0.0)
        
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
        avg_rewards = [np.mean(self.task_metrics[tid]['episode_rewards']) if self.task_metrics[tid]['episode_rewards'] else 0.0 for tid in task_ids]
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
    
    def _generate_anytime_performance_plot(self, save_dir):
        """生成随时性能曲线图"""
        if not self.anytime_performance:
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 为每个任务绘制性能曲线
        for task_id in sorted(self.anytime_performance.keys()):
            data = self.anytime_performance[task_id]
            if not data:
                continue
            
            steps, rewards = zip(*data)
            task_name = self.task_metrics.get(task_id, {}).get('task_name', f'T{task_id}')
            ax.plot(steps, rewards, label=f'{task_name}', linewidth=2, alpha=0.7, marker='o', markersize=3)
        
        ax.set_xlabel('Global Training Step', fontsize=12)
        ax.set_ylabel('Average Reward', fontsize=12)
        ax.set_title('Anytime Performance Curves (Unknown Task Boundaries)', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'anytime_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 同时保存数据到CSV
        import pandas as pd
        all_data = []
        for task_id, data in self.anytime_performance.items():
            for step, reward in data:
                all_data.append({
                    'task_id': task_id,
                    'task_name': self.task_metrics.get(task_id, {}).get('task_name', f'T{task_id}'),
                    'global_step': step,
                    'average_reward': reward
                })
        
        if all_data:
            df = pd.DataFrame(all_data)
            df.to_csv(os.path.join(save_dir, 'anytime_performance.csv'), index=False)

class EntropyGateController:
    """高/低熵门控暂停控制器，支持 high(low) 模式+迟滞"""
    def __init__(self, window_size=20, pause_steps=10, mode='high', z_hi=1.8, z_lo=1.5, z_threshold=-1.5):
        self.window_size = window_size
        self.pause_steps = pause_steps
        self.mode = mode
        self.z_hi = z_hi
        self.z_lo = z_lo
        self.z_threshold = z_threshold  # 保证兼容 low
        self.entropy_window = collections.deque(maxlen=window_size)
        self.pause_count = 0
        self.trigger_count = 0
        self.total_paused_steps = 0
    def step(self, H):
        self.entropy_window.append(H)
        if len(self.entropy_window) < self.window_size:
            return False, 0, 0
        mean = np.mean(self.entropy_window)
        std = np.std(self.entropy_window) + 1e-8
        Z = (H - mean) / std
        pause = False
        if self.mode == 'high':
            if self.pause_count > 0:
                self.pause_count -= 1
                self.total_paused_steps += 1
                if Z < self.z_lo or self.pause_count == 0:
                    if self.pause_count == 0:
                        print(f"[EGP][RESUME] after {self.pause_steps} steps")
                    return False, Z, self.trigger_count
                return True, Z, self.trigger_count
            if Z > self.z_hi:
                self.pause_count = self.pause_steps
                self.trigger_count += 1
                print(f"[EGP][TRIGGER#{self.trigger_count}] H={H:.3f} Z={Z:.2f} mode={self.mode}")
                self.total_paused_steps += 1
                return True, Z, self.trigger_count
            return False, Z, self.trigger_count
        else:
            if self.pause_count > 0:
                self.pause_count -= 1
                self.total_paused_steps += 1
                if self.pause_count == 0:
                    print(f"[EGP][RESUME] after {self.pause_steps} steps")
                return True, Z, self.trigger_count
            if Z < self.z_threshold:
                self.pause_count = self.pause_steps
                self.trigger_count += 1
                print(f"[EGP][TRIGGER#{self.trigger_count}] H={H:.3f} Z={Z:.2f} mode={self.mode}")
                self.total_paused_steps += 1
                return True, Z, self.trigger_count
            return False, Z, self.trigger_count

class DQNNetwork(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, output_size=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class EWCConsolidator:
    """
    轻量级 EWC (Elastic Weight Consolidation) 实现。
    用于在暂停时刻巩固重要参数，防止遗忘。
    """
    def __init__(self, lambda_ewc=5000.0):
        self.lambda_ewc = lambda_ewc
        self.fisher = {}   # 存储 Fisher 信息矩阵 (对角线近似)
        self.params = {}   # 存储旧任务的参数锚点 (θ*)
        self.device = torch.device("cpu") # CartPole 默认使用 CPU

    def update_fisher(self, model, dataset_samples):
        """
        在暂停时刻更新 Fisher 矩阵和锚点参数。
        Args:
            model: 当前的 DQN 网络
            dataset_samples: 从 ReplayBuffer 中采样的一批数据 [(state, action, reward, next_state, done), ...]
        """
        model.eval()
        temp_fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
        states = torch.FloatTensor(np.array([x[0] for x in dataset_samples])).to(self.device)
        model.zero_grad()
        q_values = model(states)
        log_probs = torch.nn.functional.log_softmax(q_values, dim=1)
        for i in range(q_values.shape[1]):
            loss = -log_probs[:, i].mean()
            loss.backward(retain_graph=True)
            for n, p in model.named_parameters():
                if p.grad is not None:
                    temp_fisher[n] += p.grad.pow(2) * (1.0 / q_values.shape[1])
            model.zero_grad()
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.params[n] = p.detach().clone()
                self.fisher[n] = temp_fisher[n].detach()
        model.train()
        print(f"  [EWC] 知识巩固完成 (Consolidation Complete). 保护了 {len(self.params)} 层参数.")
    def penalty(self, model):
        """计算 EWC 惩罚损失: λ * sum(F * (θ - θ*)^2)"""
        if not self.fisher:
            return 0.0
        loss = 0.0
        for n, p in model.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.params[n]).pow(2)).sum()
        return self.lambda_ewc * loss

def train(episodes=1000, max_steps=500, task_id=0, config_path="config/cartpole_config.yaml", 
          metrics_tracker=None, before_performance=None, entropy_temperature=1.0, egp_mode='high',
          egp_window=20, egp_z_hi=1.8, egp_z_lo=1.5, egp_pause_steps=10, egp_z_threshold=-1.5,
          init_model_path=None, pause_policy='egp', fixed_k=10, seed=0,
          drift_type='none', drift_slope=0.0, drift_delta=0.0, drift_amp=0.0, drift_freq=0.0):
    """训练模型 - 真正的DQN实现"""
    # 加载配置和任务信息
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 获取任务信息
    task_scheduler = TaskScheduler(config_path)
    task = task_scheduler.get_task_by_id(task_id)
    print(f"训练任务 {task_id}: {task['name']}")
    base_length = task['length']
    base_wind = task['wind']
    print(f"任务参数: 杆长={base_length}, 风力={base_wind}")
    if drift_type != 'none':
        print(f"[Drift] 启用未知任务边界漂移: type={drift_type}, slope={drift_slope}, delta={drift_delta}, "
              f"amp={drift_amp}, freq={drift_freq}")
    
    # 初始化指标跟踪
    if metrics_tracker is not None:
        metrics_tracker.record_task_start(task_id, task['name'])
        if before_performance is not None:
            metrics_tracker.record_task_performance(task_id, before_perf=before_performance)
    
    # Creating a training environment
    env_wrapper = CartPoleCL()
    env_wrapper.set_variant(length=base_length, wind=base_wind)
    env = env_wrapper.make_env(render=False, seed=seed)
    
    # Creating models and optimizers
    model = DQNNetwork()
    target_model = DQNNetwork()  # 目标网络
    if init_model_path and Path(init_model_path).exists():
        sd = torch.load(init_model_path, map_location='cpu')
        model.load_state_dict(sd)
        target_model.load_state_dict(sd)
        print(f"[Warmstart] 从 {init_model_path} 继续训练")
    else:
        target_model.load_state_dict(model.state_dict())  # 初始化目标网络
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # 经验回放缓冲区
    from collections import deque
    replay_buffer = deque(maxlen=10000)
    batch_size = 32
    gamma = 0.99
    
    # 加入熵门控暂停控制器
    entropy_gate = EntropyGateController(
        window_size=egp_window,
        pause_steps=egp_pause_steps,
        mode=egp_mode,
        z_hi=egp_z_hi,
        z_lo=egp_z_lo,
        z_threshold=egp_z_threshold
    )
    
    print("开始训练CartPole智能体...")
    print(f"训练参数: {episodes}回合, 每回合最多{max_steps}步")
    print(f"暂停策略: {pause_policy}")
    if pause_policy == 'fixed':
        print(f"固定间隔: 每 {fixed_k} 步触发, 暂停 {egp_pause_steps} 步")
    print(f"EGP温度: {entropy_temperature}")
    print("=" * 60)
    
    # 记录训练统计
    episode_rewards = []
    episode_steps = []
    recent_rewards = []
    losses = []
    
    # Fixed策略的状态变量
    fixed_global_step = 0
    fixed_countdown = 0
    fixed_triggers = 0
    fixed_paused_steps = 0
    
    for episode in range(episodes):
        if drift_type == 'progressive':
            cur_wind = base_wind + drift_slope * episode
        elif drift_type == 'abrupt':
            cur_wind = base_wind + (drift_delta if np.random.rand() < 0.03 else 0.0)
        elif drift_type == 'periodic':
            cur_wind = base_wind + drift_amp * np.sin(2 * np.pi * drift_freq * episode)
        else:
            cur_wind = base_wind
        if drift_type != 'none':
            env_wrapper.set_variant(length=base_length, wind=cur_wind)
            if episode % 100 == 0 or episode < 3:
                print(f"[Drift] Episode {episode} wind→{cur_wind:.3f}")
        obs, _ = env.reset(seed=seed)
        total_reward = 0
        steps = 0
        
        for step in range(max_steps):
            # 将观察转换为张量
            state = torch.FloatTensor(obs).unsqueeze(0)
            
            # epsilon贪婪策略选择动作
            epsilon = max(0.01, 0.9 - 0.8 * episode / 1000)  # 更合理的epsilon衰减
            # DQN动作选择：epsilon贪婪 + Q_softmax 熵
            if np.random.random() < epsilon:
                action = env.action_space.sample()
                with torch.no_grad():
                    q_values = model(state).squeeze(0)  # shape: [n_actions]
            else:
                with torch.no_grad():
                    q_values = model(state).squeeze(0)
                action = q_values.argmax().item()

            # ==== Softmax 熵计算 ====
            T = entropy_temperature
            probs = torch.softmax(q_values / T, dim=-1)
            H = -(probs * torch.log(probs + 1e-12)).sum().item()
            
            if pause_policy == 'egp':
                pause, Z, trig_id = entropy_gate.step(H)
            elif pause_policy == 'fixed':
                fixed_global_step += 1
                if fixed_countdown > 0:
                    fixed_countdown -= 1
                    fixed_paused_steps += 1
                    pause, Z, trig_id = True, 0.0, fixed_triggers
                elif fixed_global_step % fixed_k == 0:
                    fixed_countdown = egp_pause_steps
                    fixed_triggers += 1
                    fixed_paused_steps += 1
                    pause, Z, trig_id = True, 0.0, fixed_triggers
                else:
                    pause, Z, trig_id = False, 0.0, fixed_triggers
            else:  # 'none'
                pause, Z, trig_id = False, 0.0, 0

            if pause:
                next_obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                steps += 1
                done = terminated or truncated
                replay_buffer.append((obs, action, reward, next_obs, done))
                obs = next_obs
                if terminated or truncated:
                    break
                continue  # EGP日志已在控制器输出，这里不再打印
            
            # 执行动作
            next_obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            
            # 存储经验
            done = terminated or truncated
            replay_buffer.append((obs, action, reward, next_obs, done))
            
            # 训练网络
            if len(replay_buffer) >= batch_size:
                # 采样批次数据
                batch = np.random.choice(len(replay_buffer), batch_size, replace=False)
                states = torch.FloatTensor([replay_buffer[i][0] for i in batch])
                actions = torch.LongTensor([replay_buffer[i][1] for i in batch])
                rewards = torch.FloatTensor([replay_buffer[i][2] for i in batch])
                next_states = torch.FloatTensor([replay_buffer[i][3] for i in batch])
                dones = torch.BoolTensor([replay_buffer[i][4] for i in batch])
                
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
        
        # 更新目标网络
        if episode % 10 == 0:
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
        
        # 检查是否达到目标性能（当前阈值200）
        if len(recent_rewards) >= 100 and avg_reward >= 200:
            print(f"在第{episode}回合达到目标性能！平均奖励: {avg_reward:.2f}")
            break
    
    env_wrapper.close()
    
    # Calculate the final statistics
    final_avg = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
    max_reward = max(episode_rewards)
    success_rate = sum(1 for r in episode_rewards if r >= 200) / len(episode_rewards) * 100
    
    # Record final performance to the metrics tracker
    if metrics_tracker is not None:
        metrics_tracker.record_task_performance(task_id, after_perf=final_avg)
        # 写回暂停策略统计
        if task_id in metrics_tracker.task_metrics:
            if pause_policy == 'egp':
                metrics_tracker.task_metrics[task_id]['egp_triggers'] = getattr(entropy_gate, 'trigger_count', 0)
                metrics_tracker.task_metrics[task_id]['egp_paused_steps'] = getattr(entropy_gate, 'total_paused_steps', 0)
            elif pause_policy == 'fixed':
                metrics_tracker.task_metrics[task_id]['egp_triggers'] = fixed_triggers
                metrics_tracker.task_metrics[task_id]['egp_paused_steps'] = fixed_paused_steps
            else:  # 'none'
                metrics_tracker.task_metrics[task_id]['egp_triggers'] = 0
                metrics_tracker.task_metrics[task_id]['egp_paused_steps'] = 0
    
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

def demonstrate(model_path, episodes=3, max_steps=500, delay=0.05, task_id=0, config_path="config/cartpole_config.yaml"):
    """演示训练好的模型"""
    print("\n开始演示训练好的智能体...")
    print(f"加载模型: {model_path}")
    
    # 加载配置和任务信息
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    task_scheduler = TaskScheduler(config_path)
    task = task_scheduler.get_task_by_id(task_id)
    
    print(f"演示任务 {task_id}: {task['name']}")
    print(f"任务参数: 杆长={task['length']}, 风力={task['wind']}")
    
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
    env_wrapper = CartPoleCL()
    env_wrapper.set_variant(length=task['length'], wind=task['wind'])
    env = env_wrapper.make_env(render=True)
    
    if episodes == 1:
        print("开始单次演示...")
        print("观察智能体如何保持杆子平衡！")
    else:
        print(f"开始{episodes}个回合的演示...")
        print("观察智能体如何保持杆子平衡！")
    print("=" * 50)
    
    total_demo_rewards = []
    demo_steps = []
    
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
                    print(f"  步数 {step:3d}: 位置={obs[0]:6.2f}, 角度={obs[2]:6.2f}, 奖励={total_reward:3.0f}")
            else:
                if step % 50 == 0:  # 多次演示时每50步显示一次
                    print(f"  步数 {step:3d}: 位置={obs[0]:6.2f}, 角度={obs[2]:6.2f}, 奖励={total_reward:3.0f}")
            
            time.sleep(delay)
            
            if terminated or truncated:
                break
        
        total_demo_rewards.append(total_reward)
        demo_steps.append(steps)
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
        print(f"演示步数: {steps}")  # 单回合实际步数
    else:
        print(f"平均奖励: {avg_demo_reward:.1f}")
        print(f"最高奖励: {max_demo_reward:.1f}")
        print(f"平均步数: {np.mean(demo_steps):.1f}")
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
        env_wrapper = CartPoleCL()
        env_wrapper.set_variant(length=task['length'], wind=task['wind'])
        env = env_wrapper.make_env(render=False)
        
        total_rewards = []
        
        for episode in range(episodes):
            obs, _ = env.reset()
            total_reward = 0
            
            for step in range(500):  # Maximum 500 steps
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

def evaluate_on_all_tasks(model, task_scheduler, config_path, episodes_per_task=5, seed=0):
    """
    在所有任务上评估模型性能（随时评估）
    
    Args:
        model: 训练好的DQN模型
        task_scheduler: 任务调度器
        config_path: 配置文件路径
        episodes_per_task: 每个任务的评估回合数
        seed: 随机种子
        
    Returns:
        Dict[int, float]: {task_id: average_reward} 字典
    """
    model.eval()
    task_rewards = {}
    
    # 获取所有任务
    all_tasks = task_scheduler.get_task_sequence()
    
    for task in all_tasks:
        task_id = task.get('id')
        if task_id is None:
            continue
            
        # 创建评估环境
        env_wrapper = CartPoleCL()
        env_wrapper.set_variant(length=task['length'], wind=task['wind'])
        env = env_wrapper.make_env(render=False, seed=seed)
        
        total_rewards = []
        
        for episode in range(episodes_per_task):
            obs, _ = env.reset(seed=seed + episode)
            total_reward = 0
            
            for step in range(500):  # Maximum 500 steps
                state = torch.FloatTensor(obs).unsqueeze(0)
                with torch.no_grad():
                    action = model(state).max(1)[1].item()
                
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                
                if terminated or truncated:
                    break
            
            total_rewards.append(total_reward)
        
        env_wrapper.close()
        task_rewards[task_id] = np.mean(total_rewards)
    
    model.train()  # 恢复训练模式
    return task_rewards

def train_online_stream(total_steps=100000, max_steps_per_episode=500, config_path="config/cartpole_config.yaml",
                       metrics_tracker=None, entropy_temperature=1.0, egp_mode='high',
                       egp_window=20, egp_z_hi=1.8, egp_z_lo=1.5, egp_pause_steps=10, egp_z_threshold=-1.5,
                       pause_policy='egp', fixed_k=10, seed=0, steps_per_task=20000,
                       drift_type='none', drift_slope=0.0, drift_delta=0.0, drift_amp=0.0, drift_freq=0.0,
                       eval_freq=1000, eval_episodes=5, epsilon_decay=0.995,
                       enable_ewc=False, ewc_lambda=5000.0):
    """
    完全在线持续学习训练（未知任务边界）
    
    使用单一循环，持久化 replay buffer，环境参数根据全局步数动态变化
    
    Args:
        total_steps: 总训练步数
        max_steps_per_episode: 每个episode最大步数
        config_path: 配置文件路径
        metrics_tracker: 指标跟踪器
        entropy_temperature: 熵温度参数
        egp_mode: EGP模式 ('high' or 'low')
        egp_window: EGP滑动窗口大小
        egp_z_hi: 高模式触发Z阈值
        egp_z_lo: 高模式恢复Z阈值
        egp_pause_steps: EGP暂停步数
        egp_z_threshold: 低模式Z阈值
        pause_policy: 暂停策略 ('egp', 'fixed', 'none')
        fixed_k: 固定间隔策略的步数间隔
        seed: 随机种子
        steps_per_task: 每个任务持续的训练步数
        drift_type: 漂移类型
        drift_slope: progressive漂移斜率
        drift_delta: abrupt漂移增量
        drift_amp: periodic漂移振幅
        drift_freq: periodic漂移频率
        eval_freq: 评估频率（每多少步评估一次）
        eval_episodes: 每次评估的回合数
        epsilon_decay: Epsilon衰减率（每步衰减）
        
    Returns:
        (model_path, metrics_tracker): 模型路径和指标跟踪器
    """
    print("=" * 60)
    print("开始完全在线持续学习训练（未知任务边界）")
    print(f"总训练步数: {total_steps}")
    print(f"每个任务持续步数: {steps_per_task}")
    print(f"评估频率: 每 {eval_freq} 步")
    print("=" * 60)
    
    # 初始化任务调度器和动态场景
    task_scheduler = TaskScheduler(config_path)
    scenario = DynamicScenario(
        task_scheduler=task_scheduler,
        steps_per_task=steps_per_task,
        drift_type=drift_type,
        drift_slope=drift_slope,
        drift_delta=drift_delta,
        drift_amp=drift_amp,
        drift_freq=drift_freq
    )
    
    # 初始化指标跟踪器
    if metrics_tracker is None:
        metrics_tracker = MetricsTracker()
    
    # 初始化所有任务的指标
    for task in task_scheduler.get_task_sequence():
        task_id = task.get('id')
        if task_id is not None:
            metrics_tracker.record_task_start(task_id, task.get('name', f'T{task_id}'))
    
    # 创建环境
    env_wrapper = CartPoleCL()
    # 初始配置
    initial_config, _ = scenario.get_config(0)
    env_wrapper.set_variant(length=initial_config['length'], wind=initial_config['wind'])
    env = env_wrapper.make_env(render=False, seed=seed)
    
    # 创建模型和优化器（只创建一次，持续使用）
    model = DQNNetwork()
    target_model = DQNNetwork()
    target_model.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # 经验回放缓冲区（持久化，不重置）
    from collections import deque
    replay_buffer = deque(maxlen=10000)
    batch_size = 32
    gamma = 0.99
    
    # 熵门控暂停控制器
    entropy_gate = EntropyGateController(
        window_size=egp_window,
        pause_steps=egp_pause_steps,
        mode=egp_mode,
        z_hi=egp_z_hi,
        z_lo=egp_z_lo,
        z_threshold=egp_z_threshold
    )
    
    # Fixed策略的状态变量
    fixed_global_step = 0
    fixed_countdown = 0
    fixed_triggers = 0
    fixed_paused_steps = 0
    
    # 训练循环状态
    global_step = 0
    episode_count = 0
    current_episode_reward = 0
    current_episode_steps = 0
    
    # Epsilon 初始化
    epsilon = 1.0  # 初始探索率
    min_epsilon = 0.01
    
    # 当前环境状态
    obs, _ = env.reset(seed=seed)
    current_config, current_task_id = scenario.get_config(global_step)
    
    print(f"\n开始训练...")
    print(f"初始任务: {current_config['task_name']} (length={current_config['length']}, wind={current_config['wind']:.3f})")
    print(f"暂停策略: {pause_policy}")
    if pause_policy == 'fixed':
        print(f"固定间隔: 每 {fixed_k} 步触发, 暂停 {egp_pause_steps} 步")
    print("-" * 60)
    
    # == 上帝视角评估变量初始化 ==
    task_baselines = {}
    seen_tasks = set()
    seen_tasks.add(current_task_id)
    
    # 初始化 EWC
    ewc = EWCConsolidator(lambda_ewc=ewc_lambda)
    last_trig_id = 0
    
    # 主训练循环
    while global_step < total_steps:
        # 检查是否需要更新环境配置（任务切换）
        new_config, new_task_id = scenario.get_config(global_step)
        if new_task_id != current_task_id or new_config['length'] != current_config['length'] or abs(new_config['wind'] - current_config['wind']) > 1e-6:
            # == 上帝视角评估逻辑 BEGIN ==
            task_rewards = evaluate_on_all_tasks(model, task_scheduler, config_path, episodes_per_task=eval_episodes, seed=seed)
            # 记录当前任务的baseline
            if current_task_id not in task_baselines:
                baseline = task_rewards.get(current_task_id)
                if baseline is not None:
                    task_baselines[current_task_id] = baseline
                    metrics_tracker.record_task_performance(current_task_id, after_perf=baseline)
            # 计算已见过任务的CF
            for t in seen_tasks:
                if t == current_task_id:
                    continue
                cf = task_rewards.get(t, 0.0) - task_baselines.get(t, 0.0)
                cf_key = f"{t}_after_{current_task_id}"
                metrics_tracker.cf_data[cf_key] = {
                    'task_id': t,
                    'new_task_id': current_task_id,
                    'before_performance': task_baselines.get(t, 0.0),
                    'after_performance': task_rewards.get(t, 0.0),
                    'cf': cf
                }
                print(f"[CF] 任务 {t} 在训练任务 {current_task_id} 后性能变化: {cf:.2f}")
            # 记录forward transfer
            zero_shot = task_rewards.get(new_task_id)
            if zero_shot is not None:
                print(f"[Forward Transfer] 任务 {new_task_id} Zero-shot 性能: {zero_shot:.2f}")
                metrics_tracker.record_task_performance(new_task_id, before_perf=zero_shot)
            seen_tasks.add(new_task_id)
            # == 上帝视角评估逻辑 END ==
            if new_task_id != current_task_id:
                print(f"\n[任务切换] Step {global_step}: {current_config['task_name']} -> {new_config['task_name']}")
            env_wrapper.set_variant(length=new_config['length'], wind=new_config['wind'])
            current_config = new_config
            current_task_id = new_task_id
        
        # 将观察转换为张量
        state = torch.FloatTensor(obs).unsqueeze(0)
        
        # DQN动作选择
        if np.random.random() < epsilon:
            action = env.action_space.sample()
            with torch.no_grad():
                q_values = model(state).squeeze(0)
        else:
            with torch.no_grad():
                q_values = model(state).squeeze(0)
            action = q_values.argmax().item()
        
        # Softmax 熵计算
        T = entropy_temperature
        probs = torch.softmax(q_values / T, dim=-1)
        H = -(probs * torch.log(probs + 1e-12)).sum().item()
        
        # 暂停策略判断
        if pause_policy == 'egp':
            pause, Z, trig_id = entropy_gate.step(H)
            # === EWC 触发逻辑 (新增) ===
            if enable_ewc and pause and trig_id > last_trig_id:
                print(f"\n[EWC] 触发巩固! Step {global_step} (Entropy Trigger #{trig_id})")
                if len(replay_buffer) >= batch_size:
                    sample_size = min(len(replay_buffer), 128)
                    fisher_samples = random.sample(replay_buffer, sample_size)
                    ewc.update_fisher(model, fisher_samples)
                last_trig_id = trig_id
        elif pause_policy == 'fixed':
            fixed_global_step += 1
            if fixed_countdown > 0:
                fixed_countdown -= 1
                fixed_paused_steps += 1
                pause, Z, trig_id = True, 0.0, fixed_triggers
            elif fixed_global_step % fixed_k == 0:
                fixed_countdown = egp_pause_steps
                fixed_triggers += 1
                fixed_paused_steps += 1
                pause, Z, trig_id = True, 0.0, fixed_triggers
            else:
                pause, Z, trig_id = False, 0.0, fixed_triggers
        else:  # 'none'
            pause, Z, trig_id = False, 0.0, 0
        
        # 执行动作
        next_obs, reward, terminated, truncated, _ = env.step(action)
        current_episode_reward += reward
        current_episode_steps += 1
        global_step += 1
        done = terminated or truncated
        
        # 存储经验（无论是否暂停都存储）
        replay_buffer.append((obs, action, reward, next_obs, done))
        
        # === Epsilon 衰减（每一步都衰减） ===
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        # 如果不是暂停状态，进行网络更新
        if not pause:
            # 训练网络
            if len(replay_buffer) >= batch_size:
                # 采样批次数据
                batch = np.random.choice(len(replay_buffer), batch_size, replace=False)
                states = torch.FloatTensor([replay_buffer[i][0] for i in batch])
                actions = torch.LongTensor([replay_buffer[i][1] for i in batch])
                rewards = torch.FloatTensor([replay_buffer[i][2] for i in batch])
                next_states = torch.FloatTensor([replay_buffer[i][3] for i in batch])
                dones = torch.BoolTensor([replay_buffer[i][4] for i in batch])
                
                # 计算当前Q值
                current_q_values = model(states).gather(1, actions.unsqueeze(1))
                
                # 计算目标Q值
                with torch.no_grad():
                    next_q_values = target_model(next_states).max(1)[0]
                    target_q_values = rewards + (gamma * next_q_values * ~dones)
                
                # 计算损失
                loss = criterion(current_q_values.squeeze(), target_q_values)
                
                # === EWC 损失修正 (新增) ===
                if enable_ewc:
                    ewc_loss = ewc.penalty(model)
                    loss += ewc_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # 更新目标网络
        if global_step % 10 == 0:
            target_model.load_state_dict(model.state_dict())
        
        # Episode结束处理
        if done or current_episode_steps >= max_steps_per_episode:
            episode_count += 1
            
            # 记录episode指标（使用当前任务ID）
            if metrics_tracker is not None:
                metrics_tracker.record_episode(current_task_id, episode_count, current_episode_reward, epsilon)
            
            # 打印进度
            if episode_count % 50 == 0 or episode_count < 5:
                print(f"Step {global_step:6d} | Episode {episode_count:4d} | "
                      f"Task {current_task_id} | Reward: {current_episode_reward:6.1f} | "
                      f"ε: {epsilon:.3f} | H: {H:.3f}")
            
            # 重置episode
            obs, _ = env.reset(seed=seed + episode_count)
            current_episode_reward = 0
            current_episode_steps = 0
        else:
            obs = next_obs
        
        # 定期评估所有任务
        if global_step % eval_freq == 0 and global_step > 0:
            print(f"\n[评估] Step {global_step}: 评估所有任务性能...")
            task_rewards = evaluate_on_all_tasks(model, task_scheduler, config_path, 
                                                episodes_per_task=eval_episodes, seed=seed)
            
            if metrics_tracker is not None:
                metrics_tracker.record_anytime_performance(global_step, task_rewards)
            
            # 打印评估结果
            for tid, avg_reward in sorted(task_rewards.items()):
                task_name = task_scheduler.get_task_by_id(tid).get('name', f'T{tid}')
                print(f"  任务 {tid} ({task_name}): {avg_reward:.2f}")
            print()
    
    env_wrapper.close()
    
    # 记录最终统计
    if metrics_tracker is not None:
        if pause_policy == 'egp':
            # 为所有任务记录EGP统计（使用最后一个任务的值作为示例）
            for task_id in scenario.get_all_task_ids():
                if task_id in metrics_tracker.task_metrics:
                    metrics_tracker.task_metrics[task_id]['egp_triggers'] = getattr(entropy_gate, 'trigger_count', 0)
                    metrics_tracker.task_metrics[task_id]['egp_paused_steps'] = getattr(entropy_gate, 'total_paused_steps', 0)
        elif pause_policy == 'fixed':
            for task_id in scenario.get_all_task_ids():
                if task_id in metrics_tracker.task_metrics:
                    metrics_tracker.task_metrics[task_id]['egp_triggers'] = fixed_triggers
                    metrics_tracker.task_metrics[task_id]['egp_paused_steps'] = fixed_paused_steps
    
    # ===========================
    # [修复] 训练结束后的最终评估
    # ===========================
    print(f"\n[训练结束] 正在进行最终全任务评估 (Step {global_step})...")
    final_rewards = evaluate_on_all_tasks(model, task_scheduler, config_path, 
                                        episodes_per_task=eval_episodes, seed=seed)
    # 1. 记录最后一个任务的 Baseline (如果之前没记过)
    if current_task_id not in task_baselines:
        if current_task_id in final_rewards:
            task_baselines[current_task_id] = final_rewards[current_task_id]
            metrics_tracker.record_task_performance(current_task_id, after_perf=final_rewards[current_task_id])
    # 2. 计算最终的 CF (所有旧任务)
    for t in seen_tasks:
        if t == current_task_id:
            continue # 跳过当前刚结束的任务
        cf = final_rewards.get(t, 0.0) - task_baselines.get(t, 0.0)
        metrics_tracker.cf_data[f"{t}_after_{current_task_id}"] = {
            'task_id': t,
            'new_task_id': current_task_id,
            'before_performance': task_baselines.get(t, 0.0),
            'after_performance': final_rewards.get(t, 0.0),
            'cf': cf
        }
        print(f"[Final CF] 任务 {t} 最终性能变化: {cf:.2f}")
    
    return None, metrics_tracker

def train_continual_learning(task_sequence, episodes_per_task=1000, config_path="config/cartpole_config.yaml",
                             entropy_temperature=1.0, egp_mode='high', egp_window=20, egp_z_hi=1.8, egp_z_lo=1.5,
                             egp_pause_steps=10, egp_z_threshold=-1.5, pause_policy='egp', fixed_k=10, seed=0,
                             drift_type='none', drift_slope=0.0, drift_delta=0.0, drift_amp=0.0, drift_freq=0.0):
    """Continuous learning and training - supports multi-task and metric tracking"""
    print("Begin continuous learning and training...")
    print(f"Task sequence: {task_sequence}")
    print(f"Number of training rounds per task: {episodes_per_task}")
    print("=" * 60)
    
    # Initialize indicator tracker
    metrics_tracker = MetricsTracker()
    
    # Create running directory (使用绝对路径)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, "runs", f"continual_learning_{timestamp}_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    
    task_scheduler = TaskScheduler(config_path)
    model_path = None
    previous_task_performance = {}
    
    for i, task_id in enumerate(task_sequence):
        print(f"\n{'='*20} Training task {task_id} {'='*20}")
        
        # Get task information
        task = task_scheduler.get_task_by_id(task_id)
        print(f"task: {task['name']}")
        print(f"config: length={task['length']}, wind={task['wind']}")
        
        # Training current task
        init_path = model_path  # 首轮None，后续为当前最新权重
        model_path, final_performance = train(
            episodes=episodes_per_task,
            task_id=task_id,
            config_path=config_path,
            metrics_tracker=metrics_tracker,
            before_performance=previous_task_performance.get(task_id),
            entropy_temperature=entropy_temperature,
            egp_mode=egp_mode,
            egp_window=egp_window,
            egp_z_hi=egp_z_hi,
            egp_z_lo=egp_z_lo,
            egp_pause_steps=egp_pause_steps,
            egp_z_threshold=egp_z_threshold,
            pause_policy=pause_policy,
            fixed_k=fixed_k,
            seed=seed,
            drift_type=drift_type,
            drift_slope=drift_slope,
            drift_delta=drift_delta,
            drift_amp=drift_amp,
            drift_freq=drift_freq
        )
        
        # Record current task performance
        previous_task_performance[task_id] = final_performance
        
        # Calculate catastrophic forgetting (if there was a previous task).
        if i > 0:
            print(f"\nCalculating CF...")
            for prev_task_id in task_sequence[:i]:
                # First, assess the current performance of the old task.
                print(f"Assessment Task {prev_task_id} Current performance...")
                old_performance = evaluate_task_performance(model_path, prev_task_id, config_path)
                
                # Calculate CF
                if old_performance is not None and previous_task_performance.get(prev_task_id) is not None:
                    # Using subtraction: Performance change = New performance - Old performance
                    # Positive numbers indicate improved performance, while negative numbers indicate decreased performance.
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
    parser = argparse.ArgumentParser(description='CartPole训练和演示')
    parser.add_argument('--train', action='store_true', help='训练新模型')
    parser.add_argument('--demo', action='store_true', help='演示已训练的模型')
    parser.add_argument('--model-path', type=str, default='runs/trained_model.pth', help='模型文件路径')
    parser.add_argument('--episodes', type=int, default=1, help='演示回合数')
    parser.add_argument('--train-episodes', type=int, default=300, help='训练回合数')
    parser.add_argument('--task-id', type=int, default=0, help='任务ID (0-4)')
    parser.add_argument('--config', type=str, default='config/cartpole_config.yaml', help='配置文件路径')
    parser.add_argument('--continual', action='store_true', help='持续学习模式')
    parser.add_argument('--task-sequence', type=int, nargs='+', default=[0, 1, 2, 3, 4], help='任务序列')
    parser.add_argument('--episodes-per-task', type=int, default=300, help='每个任务的训练回合数')
    parser.add_argument('--entropy-temperature', type=float, default=1.0, help='Softmax策略熵的温度系数')
    parser.add_argument('--egp-mode', type=str, default="high", choices=["high", "low"], help='EGP触发模式（high=高熵, low=低熵）')
    parser.add_argument('--egp-window', type=int, default=20, help='EGP滑动窗口长度')
    parser.add_argument('--egp-z-hi', type=float, default=1.8, help='高模式触发Z阈值')
    parser.add_argument('--egp-z-lo', type=float, default=1.5, help='高模式恢复Z阈值（迟滞）')
    parser.add_argument('--egp-pause-steps', type=int, default=10, help='EGP暂停步数')
    parser.add_argument('--egp-z-threshold', type=float, default=-1.5, help='低模式启动Z阈值')
    parser.add_argument('--pause-policy', type=str, default='egp', choices=['egp', 'fixed', 'none'], help='暂停策略 (egp: 熵门控, fixed: 固定间隔, none: 无暂停)')
    parser.add_argument('--fixed-k', type=int, default=10, help='固定间隔策略的步数间隔')
    parser.add_argument('--drift-type', type=str, default='none', choices=['none', 'progressive', 'abrupt', 'periodic'], help='任务内部风力漂移类型')
    parser.add_argument('--drift-slope', type=float, default=0.0, help='progressive 漂移的每回合增量')
    parser.add_argument('--drift-delta', type=float, default=0.0, help='abrupt 漂移的突变增量')
    parser.add_argument('--drift-amp', type=float, default=0.0, help='periodic 漂移的振幅')
    parser.add_argument('--drift-freq', type=float, default=0.0, help='periodic 漂移的频率')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--online-stream', action='store_true', help='启用完全在线流式训练（未知任务边界）')
    parser.add_argument('--total-steps', type=int, default=100000, help='在线流式训练的总步数')
    parser.add_argument('--steps-per-task', type=int, default=20000, help='每个任务持续的训练步数')
    parser.add_argument('--eval-freq', type=int, default=1000, help='评估频率（每多少步评估一次）')
    parser.add_argument('--eval-episodes', type=int, default=5, help='每次评估的回合数')
    parser.add_argument('--epsilon-decay', type=float, default=0.995, help='Epsilon decay rate (per step or episode)')
    parser.add_argument('--enable-ewc', action='store_true', help='启用 EWC (Elastic Weight Consolidation) 机制')
    parser.add_argument('--ewc-lambda', type=float, default=5000.0, help='EWC 正则化系数 lambda')
    args = parser.parse_args()
    
    print("CartPole强化学习演示")
    print("=" * 40)
    
    if args.train:
        if args.online_stream:
            print("开始完全在线流式训练模式（未知任务边界）...")
            # 获取当前脚本的绝对目录
            base_dir = os.path.dirname(os.path.abspath(__file__))
            
            # 根据是否启用EWC选择不同的保存目录
            if args.enable_ewc:
                # 带EWC的实验数据保存在 runs 目录下
                run_dir = os.path.join(base_dir, "runs", args.drift_type, args.pause_policy, f"seed_{args.seed}")
            else:
                # 不带EWC的实验数据保存在 runs_cartpole_No_EWC 目录下
                run_dir = os.path.join(base_dir, "runs_cartpole_No_EWC", args.drift_type, args.pause_policy, f"seed_{args.seed}")
            os.makedirs(run_dir, exist_ok=True)
            
            # 初始化指标跟踪器
            metrics_tracker = MetricsTracker()
            
            # 执行在线流式训练
            model_path, metrics_tracker = train_online_stream(
                total_steps=args.total_steps,
                max_steps_per_episode=500,
                config_path=args.config,
                metrics_tracker=metrics_tracker,
                entropy_temperature=args.entropy_temperature,
                egp_mode=args.egp_mode,
                egp_window=args.egp_window,
                egp_z_hi=args.egp_z_hi,
                egp_z_lo=args.egp_z_lo,
                egp_pause_steps=args.egp_pause_steps,
                egp_z_threshold=args.egp_z_threshold,
                pause_policy=args.pause_policy,
                fixed_k=args.fixed_k,
                seed=args.seed,
                steps_per_task=args.steps_per_task,
                drift_type=args.drift_type,
                drift_slope=args.drift_slope,
                drift_delta=args.drift_delta,
                drift_amp=args.drift_amp,
                drift_freq=args.drift_freq,
                eval_freq=args.eval_freq,
                eval_episodes=args.eval_episodes,
                epsilon_decay=args.epsilon_decay,
                enable_ewc=args.enable_ewc,
                ewc_lambda=args.ewc_lambda
            )
            
            # 保存指标
            metrics_tracker.save_metrics(run_dir)
            print(f"\n在线流式训练完成！")
            print(f"结果保存在: {run_dir}")
            print("查看指标报告: metrics_report.txt")
            print("查看随时性能曲线: anytime_performance.png")
        elif args.continual:
            print("开始持续学习模式...")
            run_dir, metrics_tracker = train_continual_learning(
                task_sequence=args.task_sequence,
                episodes_per_task=args.episodes_per_task,
                config_path=args.config,
                entropy_temperature=args.entropy_temperature,
                egp_mode=args.egp_mode,
                egp_window=args.egp_window,
                egp_z_hi=args.egp_z_hi,
                egp_z_lo=args.egp_z_lo,
                egp_pause_steps=args.egp_pause_steps,
                egp_z_threshold=args.egp_z_threshold,
                pause_policy=args.pause_policy,
                fixed_k=args.fixed_k,
                seed=args.seed,
                drift_type=args.drift_type,
                drift_slope=args.drift_slope,
                drift_delta=args.drift_delta,
                drift_amp=args.drift_amp,
                drift_freq=args.drift_freq
            )
            print(f"\n持续学习训练完成！")
            print(f"结果保存在: {run_dir}")
            print("查看指标报告: metrics_report.txt")
        else:
            print("开始单任务训练模式...")
            model_path, final_performance = train(
                episodes=args.train_episodes, 
                task_id=args.task_id, 
                config_path=args.config,
                entropy_temperature=args.entropy_temperature,
                egp_mode=args.egp_mode,
                egp_window=args.egp_window,
                egp_z_hi=args.egp_z_hi,
                egp_z_lo=args.egp_z_lo,
                egp_pause_steps=args.egp_pause_steps,
                egp_z_threshold=args.egp_z_threshold,
                pause_policy=args.pause_policy,
                fixed_k=args.fixed_k,
                seed=args.seed,
                drift_type=args.drift_type,
                drift_slope=args.drift_slope,
                drift_delta=args.drift_delta,
                drift_amp=args.drift_amp,
                drift_freq=args.drift_freq
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
            print("请先运行训练: python run_cartpole.py --train")
            return
        demonstrate(args.model_path, args.episodes, task_id=args.task_id, config_path=args.config)
    else:
        print("请指定操作模式！")
        print("使用方法:")
        print("  单任务训练: python run_cartpole.py --train --task-id 1")
        print("  持续学习: python run_cartpole.py --train --continual")
        print("  在线流式训练: python run_cartpole.py --train --online-stream --total-steps 100000 --steps-per-task 20000")
        print("  演示: python run_cartpole.py --demo --task-id 1")
        print("  自定义持续学习: python run_cartpole.py --train --continual --task-sequence 0 1 2 --episodes-per-task 500")
        print("  任务列表:")
        print("    0: 标准CartPole")
        print("    1: T1 - 短杆+无风")
        print("    2: T2 - 短杆+有风")
        print("    3: T3 - 长杆+无风")
        print("    4: T4 - 长杆+有风")
        print("  指标跟踪:")
        print("    - 收敛速度: 记录每个任务达到目标性能的episode数")
        print("    - 平均回报: 记录每个任务的平均奖励")
        print("    - 灾难性遗忘: 计算新任务训练后旧任务性能下降程度")
        print("    - 随时性能: 在线流式训练模式下，定期评估所有任务的性能曲线")

if __name__ == '__main__':
    main()
