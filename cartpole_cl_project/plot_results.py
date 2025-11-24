#!/usr/bin/env python3
"""
结果可视化脚本 - 生成论文级别的图表
从 runs/ 目录加载数据，生成带均值和标准差的性能曲线
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

# 配置
RUNS_DIR = "runs"
DRIFT_TYPES = ['progressive', 'abrupt', 'periodic']
POLICIES = ['egp', 'fixed', 'none']
SEEDS = [0, 1, 2]
POLICY_LABELS = {
    'egp': 'EGP',
    'fixed': 'Fixed',
    'none': 'None'
}
POLICY_COLORS = {
    'egp': '#2E86AB',      # 蓝色
    'fixed': '#A23B72',    # 紫红色
    'none': '#F18F01'      # 橙色
}

# 任务边界（步数）
TASK_BOUNDARIES = [25000, 50000, 75000]


def load_anytime_performance_csv(run_dir: Path) -> pd.DataFrame:
    """
    加载单个实验的随时性能CSV文件
    
    Args:
        run_dir: 运行目录路径
        
    Returns:
        DataFrame with columns: task_id, task_name, global_step, average_reward
    """
    csv_path = run_dir / "anytime_performance.csv"
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found, skipping...")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None


def aggregate_across_seeds(drift_type: str, policy: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    聚合所有种子的数据，计算均值和标准差
    
    对于每个 global_step，计算所有任务的平均性能，然后跨种子聚合
    
    Args:
        drift_type: 漂移类型
        policy: 暂停策略
        
    Returns:
        (steps, mean_rewards, std_rewards): 全局步数数组，平均奖励数组，标准差数组
    """
    all_step_data = {}  # {global_step: [reward_values_across_seeds]}
    
    # 加载所有种子的数据
    for seed in SEEDS:
        run_dir = Path(RUNS_DIR) / drift_type / policy / f"seed_{seed}"
        df = load_anytime_performance_csv(run_dir)
        if df is None:
            continue
        
        # 对于每个 global_step，计算所有任务的平均奖励（任务级聚合）
        step_averages = df.groupby('global_step')['average_reward'].mean()
        
        # 收集到字典中
        for step, avg_reward in step_averages.items():
            if step not in all_step_data:
                all_step_data[step] = []
            all_step_data[step].append(avg_reward)
    
    if not all_step_data:
        print(f"Warning: No data found for drift={drift_type}, policy={policy}")
        return None, None, None
    
    # 对每个步数计算均值和标准差
    steps = sorted(all_step_data.keys())
    mean_rewards = []
    std_rewards = []
    
    for step in steps:
        rewards = np.array(all_step_data[step])
        mean_rewards.append(np.mean(rewards))
        std_rewards.append(np.std(rewards))
    
    steps = np.array(steps)
    mean_rewards = np.array(mean_rewards)
    std_rewards = np.array(std_rewards)
    
    # 处理 NaN（当只有一个数据点时）
    std_rewards = np.nan_to_num(std_rewards, nan=0.0)
    
    return steps, mean_rewards, std_rewards


def plot_benchmark_figure(drift_type: str, save_path: Path):
    """
    为特定漂移类型生成基准测试图表
    
    Args:
        drift_type: 漂移类型
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 为每个策略绘制曲线
    for policy in POLICIES:
        steps, mean_rewards, std_rewards = aggregate_across_seeds(drift_type, policy)
        
        if steps is None:
            continue
        
        label = POLICY_LABELS[policy]
        color = POLICY_COLORS[policy]
        
        # 绘制均值曲线
        ax.plot(steps, mean_rewards, label=label, color=color, linewidth=2.5, alpha=0.9)
        
        # 绘制标准差阴影
        ax.fill_between(
            steps,
            mean_rewards - std_rewards,
            mean_rewards + std_rewards,
            color=color,
            alpha=0.2,
            label=f'{label} ±1σ'
        )
    
    # 添加任务边界垂直线
    for boundary in TASK_BOUNDARIES:
        ax.axvline(x=boundary, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # 设置标签和标题
    ax.set_xlabel('Global Training Step', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Reward', fontsize=14, fontweight='bold')
    ax.set_title(f'Anytime Performance: {drift_type.capitalize()} Drift', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # 设置图例
    ax.legend(loc='best', fontsize=12, framealpha=0.9)
    
    # 网格
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # 设置坐标轴范围
    ax.set_xlim(0, 100000)
    ax.set_ylim(bottom=0)
    
    # 美化
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_per_task_performance(drift_type: str, save_dir: Path):
    """
    为每个任务分别绘制性能曲线（可选，用于详细分析）
    
    Args:
        drift_type: 漂移类型
        save_dir: 保存目录
    """
    # 获取所有任务ID
    run_dir = Path(RUNS_DIR) / drift_type / POLICIES[0] / f"seed_{SEEDS[0]}"
    df = load_anytime_performance_csv(run_dir)
    if df is None:
        return
    
    task_ids = sorted(df['task_id'].unique())
    
    # 为每个任务创建一个子图
    fig, axes = plt.subplots(len(task_ids), 1, figsize=(12, 4 * len(task_ids)))
    if len(task_ids) == 1:
        axes = [axes]
    
    fig.suptitle(f'Per-Task Performance: {drift_type.capitalize()} Drift', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    for idx, task_id in enumerate(task_ids):
        ax = axes[idx]
        
        for policy in POLICIES:
            # 收集该任务的所有数据
            all_rewards = []
            all_steps = None
            
            for seed in SEEDS:
                run_dir = Path(RUNS_DIR) / drift_type / policy / f"seed_{seed}"
                df = load_anytime_performance_csv(run_dir)
                if df is not None:
                    task_df = df[df['task_id'] == task_id].sort_values('global_step')
                    if all_steps is None:
                        all_steps = task_df['global_step'].values
                    rewards = task_df['average_reward'].values
                    all_rewards.append(rewards)
            
            if not all_rewards:
                continue
            
            # 计算均值和标准差
            all_rewards = np.array(all_rewards)
            mean_rewards = np.mean(all_rewards, axis=0)
            std_rewards = np.std(all_rewards, axis=0)
            
            label = POLICY_LABELS[policy]
            color = POLICY_COLORS[policy]
            
            ax.plot(all_steps, mean_rewards, label=label, color=color, linewidth=2, alpha=0.9)
            ax.fill_between(
                all_steps,
                mean_rewards - std_rewards,
                mean_rewards + std_rewards,
                color=color,
                alpha=0.2
            )
        
        # 任务边界
        for boundary in TASK_BOUNDARIES:
            ax.axvline(x=boundary, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        task_name = df[df['task_id'] == task_id]['task_name'].iloc[0]
        ax.set_ylabel(f'Task {task_id}\n({task_name})', fontsize=11, fontweight='bold')
        ax.set_xlim(0, 100000)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        
        if idx == len(task_ids) - 1:
            ax.set_xlabel('Global Training Step', fontsize=12, fontweight='bold')
        else:
            ax.set_xticklabels([])
    
    plt.tight_layout()
    save_path = save_dir / f'per_task_{drift_type}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    """主函数"""
    print("=" * 80)
    print("RQ1 Results Plotting Script")
    print("=" * 80)
    
    # 检查运行目录
    runs_path = Path(RUNS_DIR)
    if not runs_path.exists():
        print(f"Error: {RUNS_DIR} directory not found!")
        print("Please run batch_runner.py first to generate experimental data.")
        return
    
    # 创建输出目录
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    # 为每个漂移类型生成基准图表
    print("\nGenerating benchmark figures...")
    for drift_type in DRIFT_TYPES:
        save_path = plots_dir / f"benchmark_{drift_type}.png"
        
        # 检查是否有数据
        has_data = False
        for policy in POLICIES:
            run_dir = runs_path / drift_type / policy / f"seed_{SEEDS[0]}"
            if run_dir.exists() and (run_dir / "anytime_performance.csv").exists():
                has_data = True
                break
        
        if not has_data:
            print(f"Warning: No data found for drift_type={drift_type}, skipping...")
            continue
        
        plot_benchmark_figure(drift_type, save_path)
    
    # 可选：生成每个任务的详细图表
    print("\nGenerating per-task performance figures...")
    for drift_type in DRIFT_TYPES:
        plot_per_task_performance(drift_type, plots_dir)
    
    print("\n" + "=" * 80)
    print("Plotting completed!")
    print(f"Results saved in: {plots_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
