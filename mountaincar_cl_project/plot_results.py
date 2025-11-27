#!/usr/bin/env python3
"""
MountainCar RQ1 结果可视化
读取 runs/{drift}/{policy}/seed_X/anytime_performance.csv，生成对比图
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RUNS_DIR = "runs"
DRIFT_TYPES = ['progressive', 'abrupt', 'periodic']
POLICIES = ['egp', 'fixed', 'none']
SEEDS = [0, 1, 2]

POLICY_LABELS = {
    'egp': 'EGP',
    'fixed': 'Fixed',
    'none': 'None',
}
POLICY_COLORS = {
    'egp': '#2E86AB',
    'fixed': '#A23B72',
    'none': '#F18F01',
}

TASK_BOUNDARIES = [25000, 50000, 75000]


def load_anytime_csv(run_dir: Path) -> pd.DataFrame | None:
    csv_path = run_dir / "anytime_performance.csv"
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found, skip.")
        return None
    try:
        return pd.read_csv(csv_path)
    except Exception as exc:
        print(f"Failed to read {csv_path}: {exc}")
        return None


def aggregate_policy(drift_type: str, policy: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按 global_step 聚合 (均值 / 标准差)"""
    step_to_rewards: Dict[int, List[float]] = {}
    
    for seed in SEEDS:
        run_dir = Path(RUNS_DIR) / drift_type / policy / f"seed_{seed}"
        df = load_anytime_csv(run_dir)
        if df is None:
            continue
        step_avg = df.groupby('global_step')['average_reward'].mean()
        for step, reward in step_avg.items():
            step_to_rewards.setdefault(int(step), []).append(float(reward))
    
    if not step_to_rewards:
        return None, None, None
    
    steps = sorted(step_to_rewards.keys())
    means = []
    stds = []
    for step in steps:
        vals = np.array(step_to_rewards[step])
        means.append(vals.mean())
        stds.append(vals.std())
    
    return np.array(steps), np.array(means), np.array(stds)


def plot_benchmark(drift_type: str, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    
    plotted = False
    for policy in POLICIES:
        steps, means, stds = aggregate_policy(drift_type, policy)
        if steps is None:
            continue
        plotted = True
        label = POLICY_LABELS[policy]
        color = POLICY_COLORS[policy]
        ax.plot(steps, means, label=label, color=color, linewidth=2.5)
        ax.fill_between(steps, means - stds, means + stds, color=color, alpha=0.2)
    
    if not plotted:
        print(f"Warning: no data for drift={drift_type}, skip figure.")
        plt.close(fig)
        return
    
    for boundary in TASK_BOUNDARIES:
        ax.axvline(boundary, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Global Training Step', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Reward', fontsize=14, fontweight='bold')
    ax.set_title(f'MountainCar Anytime Performance ({drift_type.capitalize()})', fontsize=16, fontweight='bold')
    ax.set_xlim(0, 100000)
    ax.set_ylim(-200, 0)  # Mountain Car rewards are negative, adjusted for better visibility
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_per_task(drift_type: str, save_dir: Path) -> None:
    sample_dir = Path(RUNS_DIR) / drift_type / POLICIES[0] / f"seed_{SEEDS[0]}"
    df = load_anytime_csv(sample_dir)
    if df is None:
        return
    
    task_ids = sorted(df['task_id'].unique())
    fig, axes = plt.subplots(len(task_ids), 1, figsize=(12, 4 * len(task_ids)))
    if len(task_ids) == 1:
        axes = [axes]
    
    for idx, task_id in enumerate(task_ids):
        ax = axes[idx]
        for policy in POLICIES:
            all_rewards = []
            steps_ref = None
            for seed in SEEDS:
                run_dir = Path(RUNS_DIR) / drift_type / policy / f"seed_{seed}"
                df_policy = load_anytime_csv(run_dir)
                if df_policy is None:
                    continue
                task_df = df_policy[df_policy['task_id'] == task_id].sort_values('global_step')
                if task_df.empty:
                    continue
                if steps_ref is None:
                    steps_ref = task_df['global_step'].values
                all_rewards.append(task_df['average_reward'].values)
            if not all_rewards or steps_ref is None:
                continue
            arr = np.vstack(all_rewards)
            mean_rewards = arr.mean(axis=0)
            std_rewards = arr.std(axis=0)
            label = POLICY_LABELS[policy]
            color = POLICY_COLORS[policy]
            ax.plot(steps_ref, mean_rewards, label=label, color=color, linewidth=2)
            ax.fill_between(steps_ref, mean_rewards - std_rewards, mean_rewards + std_rewards, color=color, alpha=0.2)
        
        for boundary in TASK_BOUNDARIES:
            ax.axvline(boundary, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        task_name = df[df['task_id'] == task_id]['task_name'].iloc[0]
        ax.set_ylabel(f'Task {task_id}\n({task_name})', fontsize=11, fontweight='bold')
        ax.set_xlim(0, 100000)
        ax.set_ylim(-200, 0)  # Mountain Car rewards are negative, adjusted for better visibility
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        if idx == len(task_ids) - 1:
            ax.set_xlabel('Global Training Step', fontsize=12, fontweight='bold')
        else:
            ax.set_xticklabels([])
    
    plt.tight_layout()
    out_path = save_dir / f'per_task_{drift_type}.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    print("=" * 80)
    print("MountainCar RQ1 Plotting")
    print("=" * 80)
    
    runs_path = Path(RUNS_DIR)
    if not runs_path.exists():
        print(f"Error: {RUNS_DIR} not found. Run experiments first.")
        return
    
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    print("\nGenerating benchmark figures...")
    for drift in DRIFT_TYPES:
        save_path = plots_dir / f'benchmark_{drift}.png'
        plot_benchmark(drift, save_path)
    
    print("\nGenerating per-task figures...")
    for drift in DRIFT_TYPES:
        plot_per_task(drift, plots_dir)
    
    print("\nAll figures saved to:", plots_dir)


if __name__ == "__main__":
    main()


