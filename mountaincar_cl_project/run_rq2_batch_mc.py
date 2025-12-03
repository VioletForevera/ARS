import subprocess
import sys
import time
import os

# ================= MountainCar 实验配置 (极速版) =================
# 1. 随机种子：跑 3 个种子取平均
SEEDS = [0, 1, 2] 

# 2. 漂移类型：RQ2 主要关注 'abrupt' (突变) 场景
DRIFT_TYPE = "abrupt" 

# 3. 训练参数 (完整训练 - 100K步)
# 总步数 100,000 (5个任务 * 20,000步)
TOTAL_STEPS = "100000"      
STEPS_PER_TASK = "20000"    
EVAL_FREQ = "1000"          # 评估频率（每1000步评估一次）
PAUSE_POLICY = "egp"        # 暂停策略

# 4. EWC 参数
EWC_LAMBDA = "5000.0"       # 正则化强度

# 脚本路径定位 (使用项目根目录的包装脚本)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_PATH = os.path.join(CURRENT_DIR, "run_mountaincar.py")

def run_single_experiment(seed, enable_ewc):
    """执行单个实验命令"""
    
    cmd = [
        sys.executable, SCRIPT_PATH,
        "--train",
        "--online-stream",
        "--total-steps", TOTAL_STEPS,
        "--steps-per-task", STEPS_PER_TASK,
        "--eval-freq", EVAL_FREQ,
        "--drift-type", DRIFT_TYPE,
        "--pause-policy", PAUSE_POLICY,
        "--seed", str(seed)
    ]
    
    # 区分实验组
    if enable_ewc:
        group_name = "Method: MC + EWC"
        cmd.append("--enable-ewc")
        cmd.extend(["--ewc-lambda", EWC_LAMBDA])
        # 结果将存入 runs/
    else:
        group_name = "Baseline: MC (No EWC)"
        # 不加 EWC 参数，代码会自动存入 runs_mountaincar_No_EWC/
    
    print(f"\n{'-'*60}")
    print(f"▶ 正在运行: {group_name} (Seed {seed})")
    print(f"  参数: Steps={TOTAL_STEPS}, TaskSteps={STEPS_PER_TASK}")
    print(f"{'-'*60}")
    
    start_time = time.time()
    try:
        # 这里的 cwd 设置很重要，确保 python 能够正确解析包路径
        # 我们在 mountaincar_cl_project 根目录下运行
        subprocess.run(cmd, check=True, cwd=CURRENT_DIR)
        duration = (time.time() - start_time) / 60
        print(f"✔ 完成! 耗时: {duration:.2f} 分钟")
    except subprocess.CalledProcessError as e:
        print(f"✘ 失败! 退出代码: {e.returncode}")
        # sys.exit(1) # 可以选择报错继续或退出
    except KeyboardInterrupt:
        print("\n🛑 实验已手动终止")
        sys.exit(0)

def main():
    print(f"=== MountainCar RQ2 极速批量实验 (Total Seeds: {len(SEEDS)}) ===")
    
    # 1. 跑 Baseline 组
    print("\n>>> [Phase 1] Running Baseline (无巩固)...")
    for seed in SEEDS:
        run_single_experiment(seed, enable_ewc=False)
        
    # 2. 跑 EWC 组
    print("\n>>> [Phase 2] Running EWC Method (有巩固)...")
    for seed in SEEDS:
        run_single_experiment(seed, enable_ewc=True)

    print("\n" + "="*60)
    print("🎉 实验全部完成！请检查数据：")
    print(f"1. Baseline: {os.path.join(CURRENT_DIR, 'mountaincar_cl', 'runs_mountaincar_No_EWC')}")
    print(f"2. EWC Data: {os.path.join(CURRENT_DIR, 'mountaincar_cl', 'runs')}")
    print("="*60)

if __name__ == "__main__":
    main()
