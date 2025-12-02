import subprocess
import sys
import time
import os
from itertools import product

# ================= 实验参数配置 (3×3 完整矩阵) =================
# 1. 随机种子：跑 3 个种子取平均
SEEDS = [0, 1, 2] 

# 2. 漂移类型：3种漂移方式
DRIFT_TYPES = ["abrupt", "progressive", "periodic"]

# 3. 暂停策略：3种暂停方式
PAUSE_POLICIES = ["egp", "fixed", "none"]

# 4. 训练参数 (快速实验设置)
TOTAL_STEPS = "25000"      
STEPS_PER_TASK = "5000"    
EVAL_FREQ = "250"

# 5. EWC 参数
EWC_LAMBDA = "5000.0"

# 6. 漂移参数配置
DRIFT_PARAMS = {
    "abrupt": {
        "drift_delta": "0.5",    # 突变增量
    },
    "progressive": {
        "drift_slope": "0.0001", # 渐进斜率
    },
    "periodic": {
        "drift_amp": "0.3",      # 周期振幅
        "drift_freq": "0.0002",  # 周期频率
    }
}

# 7. Fixed EGP 参数
FIXED_K = "1000"  # 固定间隔步数

# 脚本定位
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_PATH = os.path.join(CURRENT_DIR, "run_cartpole.py")

def run_single_experiment(seed, drift_type, pause_policy, enable_ewc):
    """执行单个实验命令"""
    
    cmd = [
        sys.executable, SCRIPT_PATH,
        "--train",
        "--online-stream",
        "--total-steps", TOTAL_STEPS,
        "--steps-per-task", STEPS_PER_TASK,
        "--eval-freq", EVAL_FREQ,
        "--drift-type", drift_type,
        "--pause-policy", pause_policy,
        "--seed", str(seed)
    ]
    
    # 添加漂移参数
    if drift_type in DRIFT_PARAMS:
        for param_name, param_value in DRIFT_PARAMS[drift_type].items():
            cmd.extend([f"--{param_name.replace('_', '-')}", param_value])
    
    # 添加 Fixed EGP 参数
    if pause_policy == "fixed":
        cmd.extend(["--fixed-k", FIXED_K])
    
    # 区分实验组
    method_name = f"{drift_type.capitalize()} + {pause_policy.upper()}"
    if enable_ewc:
        method_name += " + EWC"
        cmd.append("--enable-ewc")
        cmd.extend(["--ewc-lambda", EWC_LAMBDA])
    
    print(f"\n{'-'*70}")
    print(f"▶ 正在运行: {method_name} (Seed {seed})")
    print(f"  漂移类型: {drift_type} | 暂停策略: {pause_policy} | EWC: {enable_ewc}")
    print(f"  参数: Steps={TOTAL_STEPS}, TaskSteps={STEPS_PER_TASK}, EvalFreq={EVAL_FREQ}")
    print(f"{'-'*70}")
    
    start_time = time.time()
    try:
        # 在脚本所在目录运行命令，确保相对路径正确
        subprocess.run(cmd, check=True, cwd=CURRENT_DIR)
        duration = (time.time() - start_time) / 60
        print(f"✔ 完成! 耗时: {duration:.2f} 分钟")
    except subprocess.CalledProcessError as e:
        print(f"✘ 失败! 退出代码: {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 实验已手动终止")
        sys.exit(0)

def main():
    total_experiments = len(DRIFT_TYPES) * len(PAUSE_POLICIES) * 2 * len(SEEDS)  # 2 = Baseline + EWC
    print(f"{'='*70}")
    print(f"=== 开始 RQ2 完整批量实验 (3×3 矩阵) ===")
    print(f"{'='*70}")
    print(f"漂移类型: {', '.join(DRIFT_TYPES)}")
    print(f"暂停策略: {', '.join(PAUSE_POLICIES)}")
    print(f"随机种子: {SEEDS}")
    print(f"总实验数: {total_experiments} 组 (9 组合 × 2 方法 × {len(SEEDS)} 种子)")
    print(f"{'='*70}\n")
    
    experiment_count = 0
    
    # 遍历所有组合：漂移类型 × 暂停策略 × Baseline/EWC
    for drift_type, pause_policy in product(DRIFT_TYPES, PAUSE_POLICIES):
        print(f"\n{'#'*70}")
        print(f"### 实验组合: {drift_type.upper()} × {pause_policy.upper()} ###")
        print(f"{'#'*70}")
        
        # 1. 跑 Baseline 组（无 EWC）
        print(f"\n>>> [Baseline] {drift_type} + {pause_policy} (无巩固)...")
        for seed in SEEDS:
            experiment_count += 1
            print(f"\n进度: [{experiment_count}/{total_experiments}]")
            run_single_experiment(seed, drift_type, pause_policy, enable_ewc=False)
        
        # 2. 跑 EWC 组（有 EWC）
        print(f"\n>>> [EWC Method] {drift_type} + {pause_policy} (有巩固)...")
        for seed in SEEDS:
            experiment_count += 1
            print(f"\n进度: [{experiment_count}/{total_experiments}]")
            run_single_experiment(seed, drift_type, pause_policy, enable_ewc=True)

    print("\n" + "="*70)
    print("🎉 实验全部完成！")
    print(f"总共完成 {experiment_count} 个实验")
    print(f"\n结果目录：")
    print(f"  - Baseline: {os.path.join(CURRENT_DIR, 'runs_cartpole_No_EWC')}")
    print(f"  - EWC Data: {os.path.join(CURRENT_DIR, 'runs')}")
    print("="*70)

if __name__ == "__main__":
    main()
