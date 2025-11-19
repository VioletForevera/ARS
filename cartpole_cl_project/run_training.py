#!/usr/bin/env python3
"""
简化的训练脚本 - 自动训练并演示
"""

import subprocess
import sys
import os

def main():
    print("CartPole自动训练和演示")
    print("=" * 40)
    
    # 训练模型
    print("开始训练...")
    try:
        result = subprocess.run([
            sys.executable, "run_cartpole.py", "--train", "--train-episodes", "200"
        ], capture_output=True, text=True, input="n\n")  # 自动回答"否"避免交互
        print("训练输出:")
        print(result.stdout)
        if result.stderr:
            print("错误信息:")
            print(result.stderr)
    except Exception as e:
        print(f"训练失败: {e}")
        return
    
    print("\n" + "=" * 40)
    print("开始演示训练好的模型...")
    
    # 演示模型
    try:
        result = subprocess.run([
            sys.executable, "run_cartpole.py", "--demo", "--episodes", "3"
        ], capture_output=True, text=True)
        print("演示输出:")
        print(result.stdout)
        if result.stderr:
            print("错误信息:")
            print(result.stderr)
    except Exception as e:
        print(f"演示失败: {e}")
        return
    
    print("\n完成！")

if __name__ == "__main__":
    main()












