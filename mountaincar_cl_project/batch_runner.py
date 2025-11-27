#!/usr/bin/env python3
"""
MountainCar RQ1 批量实验运行器
执行 27 个组合：3 种漂移 × 3 种暂停策略 × 3 种随机种子
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from datetime import datetime

# 实验网格
SEEDS = [0, 1, 2]
DRIFT_TYPES = ['progressive', 'abrupt', 'periodic']
POLICIES = ['egp', 'fixed', 'none']

# 漂移参数（MountainCar 物理量较小，精调后更易观察差异）
DRIFT_PARAMS = {
    'progressive': {
        'drift_type': 'progressive',
        'drift_slope': 0.00001,
        'drift_delta': 0.0,
        'drift_amp': 0.0,
        'drift_freq': 0.0,
    },
    'abrupt': {
        'drift_type': 'abrupt',
        'drift_slope': 0.0,
        'drift_delta': 0.0005,
        'drift_amp': 0.0,
        'drift_freq': 0.0,
    },
    'periodic': {
        'drift_type': 'periodic',
        'drift_slope': 0.0,
        'drift_delta': 0.0,
        'drift_amp': 0.0005,
        'drift_freq': 0.001,
    },
}

# 固定策略配置
FIXED_K = 100

# 通用黄金参数
COMMON_PARAMS = {
    'total_steps': 100000,
    'steps_per_task': 25000,
    'eval_freq': 1000,
    'eval_episodes': 5,
    'epsilon_decay': 0.99995,
    'entropy_temperature': 1.0,
    'egp_z_hi': 3.5,
    'egp_window': 50,
}

PROJECT_ROOT = Path(__file__).resolve().parent
RUN_DIR = PROJECT_ROOT / "mountaincar_cl"


def build_command(drift_type: str, policy: str, seed: int) -> list[str]:
    drift = DRIFT_PARAMS[drift_type]
    cmd = [
        sys.executable,
        'run_mountaincar.py',
        '--train',
        '--online-stream',
        '--total-steps', str(COMMON_PARAMS['total_steps']),
        '--steps-per-task', str(COMMON_PARAMS['steps_per_task']),
        '--eval-freq', str(COMMON_PARAMS['eval_freq']),
        '--eval-episodes', str(COMMON_PARAMS['eval_episodes']),
        '--epsilon-decay', str(COMMON_PARAMS['epsilon_decay']),
        '--entropy-temperature', str(COMMON_PARAMS['entropy_temperature']),
        '--egp-z-hi', str(COMMON_PARAMS['egp_z_hi']),
        '--egp-window', str(COMMON_PARAMS['egp_window']),
        '--drift-type', drift['drift_type'],
        '--drift-slope', str(drift['drift_slope']),
        '--drift-delta', str(drift['drift_delta']),
        '--drift-amp', str(drift['drift_amp']),
        '--drift-freq', str(drift['drift_freq']),
        '--pause-policy', policy,
        '--seed', str(seed),
    ]
    if policy == 'fixed':
        cmd.extend(['--fixed-k', str(FIXED_K)])
    return cmd


def run_experiment(drift_type: str, policy: str, seed: int,
                   idx: int, total: int) -> bool:
    print("\n" + "=" * 90)
    print(f"Running [{idx}/{total}] Drift={drift_type} Policy={policy} Seed={seed}")
    print("=" * 90)
    cmd = build_command(drift_type, policy, seed)
    print("Command:", " ".join(cmd))
    print("-" * 90)
    try:
        subprocess.run(
            cmd,
            check=True,
            cwd=RUN_DIR,
            capture_output=False,
        )
        print(f"\n✓ Completed [{idx}/{total}] Drift={drift_type} Policy={policy} Seed={seed}")
        return True
    except subprocess.CalledProcessError as exc:
        print(f"\n✗ Failed [{idx}/{total}] Drift={drift_type} Policy={policy} Seed={seed}")
        print(f"Exit code: {exc.returncode}")
        return False


def main() -> None:
    print("=" * 90)
    print("MountainCar Online Stream Batch Runner")
    print("=" * 90)
    total = len(DRIFT_TYPES) * len(POLICIES) * len(SEEDS)
    print(f"Total Experiments: {total}")
    print(f"Drift Types: {DRIFT_TYPES}")
    print(f"Policies: {POLICIES}")
    print(f"Seeds: {SEEDS}")
    print("=" * 90)
    
    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)
    
    start_time = datetime.now()
    idx = 0
    successes = 0
    failures = 0
    
    for drift in DRIFT_TYPES:
        for policy in POLICIES:
            for seed in SEEDS:
                idx += 1
                if run_experiment(drift, policy, seed, idx, total):
                    successes += 1
                else:
                    failures += 1
    
    duration = datetime.now() - start_time
    print("\n" + "=" * 90)
    print("Batch Summary")
    print("=" * 90)
    print(f"Total: {total}")
    print(f"Success: {successes}")
    print(f"Failed: {failures}")
    print(f"Duration: {duration}")
    print(f"Avg per run: {duration / total if total else 0}")
    print("=" * 90)
    
    if failures:
        print(f"\nWARNING: {failures} experiment(s) failed. 请检查日志。")
        sys.exit(1)
    
    print("\nAll experiments finished successfully!")
    print("下一步：运行 `python plot_results.py` 生成对比图。")


if __name__ == "__main__":
    main()

