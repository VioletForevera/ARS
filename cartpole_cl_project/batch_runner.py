#!/usr/bin/env python3
"""
批量实验运行器 - RQ1 实验矩阵
执行 27 个实验: 3 Drifts × 3 Policies × 3 Seeds
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

# 实验配置
SEEDS = [0, 1, 2]
DRIFT_TYPES = ['progressive', 'abrupt', 'periodic']
POLICIES = ['egp', 'fixed', 'none']

# Drift 参数 (根据 smoke tests 调优后的值)
DRIFT_PARAMS = {
    'progressive': {
        'drift_type': 'progressive',
        'drift_slope': 0.0001,
        'drift_delta': 0.0,
        'drift_amp': 0.0,
        'drift_freq': 0.0
    },
    'abrupt': {
        'drift_type': 'abrupt',
        'drift_slope': 0.0,
        'drift_delta': 0.4,  # 增加到 0.4 确保影响可见
        'drift_amp': 0.0,
        'drift_freq': 0.0
    },
    'periodic': {
        'drift_type': 'periodic',
        'drift_slope': 0.0,
        'drift_delta': 0.0,
        'drift_amp': 0.5,  # 增加到 0.5 增加振幅
        'drift_freq': 0.001  # 频率 0.001
    }
}

# Fixed 策略参数
FIXED_K = 100

# 通用参数 (Golden Settings)
COMMON_PARAMS = {
    'total_steps': 100000,
    'steps_per_task': 25000,
    'eval_freq': 1000,
    'eval_episodes': 5,
    'epsilon_decay': 0.99995,
    'egp_z_hi': 3.5,
    'egp_window': 50
}

def build_command(drift_type, policy, seed):
    """构建命令行参数"""
    drift = DRIFT_PARAMS[drift_type]
    
    cmd = [
        sys.executable,  # Python 解释器
        'run_cartpole.py',
        '--train',
        '--online-stream',
        '--total-steps', str(COMMON_PARAMS['total_steps']),
        '--steps-per-task', str(COMMON_PARAMS['steps_per_task']),
        '--eval-freq', str(COMMON_PARAMS['eval_freq']),
        '--eval-episodes', str(COMMON_PARAMS['eval_episodes']),
        '--epsilon-decay', str(COMMON_PARAMS['epsilon_decay']),
        '--egp-z-hi', str(COMMON_PARAMS['egp_z_hi']),
        '--egp-window', str(COMMON_PARAMS['egp_window']),
        '--drift-type', drift['drift_type'],
        '--drift-slope', str(drift['drift_slope']),
        '--drift-delta', str(drift['drift_delta']),
        '--drift-amp', str(drift['drift_amp']),
        '--drift-freq', str(drift['drift_freq']),
        '--pause-policy', policy,
        '--seed', str(seed)
    ]
    
    # Fixed 策略需要额外参数
    if policy == 'fixed':
        cmd.extend(['--fixed-k', str(FIXED_K)])
    
    return cmd

def run_experiment(drift_type, policy, seed, experiment_num, total_experiments):
    """运行单个实验"""
    print("\n" + "=" * 80)
    print(f"Running [{experiment_num}/{total_experiments}]: "
          f"Drift={drift_type}, Policy={policy}, Seed={seed}")
    print("=" * 80)
    
    cmd = build_command(drift_type, policy, seed)
    
    # 打印命令（用于调试）
    print(f"Command: {' '.join(cmd)}")
    print("-" * 80)
    
    try:
        # 运行命令
        result = subprocess.run(
            cmd,
            check=True,
            cwd=Path(__file__).parent,
            capture_output=False  # 显示实时输出
        )
        print(f"\n✓ Completed [{experiment_num}/{total_experiments}]: "
              f"Drift={drift_type}, Policy={policy}, Seed={seed}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed [{experiment_num}/{total_experiments}]: "
              f"Drift={drift_type}, Policy={policy}, Seed={seed}")
        print(f"Error: {e}")
        return False

def main():
    """主函数：运行所有实验"""
    print("=" * 80)
    print("RQ1 Batch Experiment Runner")
    print("=" * 80)
    print(f"Total Experiments: {len(DRIFT_TYPES) * len(POLICIES) * len(SEEDS)}")
    print(f"Drift Types: {DRIFT_TYPES}")
    print(f"Policies: {POLICIES}")
    print(f"Seeds: {SEEDS}")
    print("=" * 80)
    
    # 确保运行目录存在
    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)
    
    # 记录开始时间
    start_time = datetime.now()
    
    # 运行所有实验
    experiment_num = 0
    successful = 0
    failed = 0
    
    for drift_type in DRIFT_TYPES:
        for policy in POLICIES:
            for seed in SEEDS:
                experiment_num += 1
                success = run_experiment(
                    drift_type, policy, seed, 
                    experiment_num, 
                    len(DRIFT_TYPES) * len(POLICIES) * len(SEEDS)
                )
                if success:
                    successful += 1
                else:
                    failed += 1
    
    # 打印总结
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 80)
    print("Batch Experiment Summary")
    print("=" * 80)
    print(f"Total Experiments: {experiment_num}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total Duration: {duration}")
    print(f"Average per Experiment: {duration / experiment_num if experiment_num > 0 else 0}")
    print("=" * 80)
    
    if failed > 0:
        print(f"\nWarning: {failed} experiment(s) failed. Please check the logs above.")
        sys.exit(1)
    else:
        print("\nAll experiments completed successfully!")
        print("\nNext steps:")
        print("1. Run 'python plot_results.py' to generate paper-ready figures")
        print("2. Check individual results in runs/{drift_type}/{pause_policy}/seed_{seed}/")

if __name__ == '__main__':
    main()
