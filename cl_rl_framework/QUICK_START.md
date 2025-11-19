# Quick Start Guide

## MountainCar Continuous Learning

### 快速导航
```powershell
# 从任何位置
cd "C:\Users\panha\Desktop\ARS CW\cl_rl_framework\mountaincar_cl"
```

### 运行训练
```powershell
python run_mountaincar.py --train --continual --task-sequence 0 1 2 3 4 --episodes-per-task 500
```

### 查看帮助
```powershell
python run_mountaincar.py
```

## 输出位置
训练完成后，结果保存在：
```
runs/continual_learning_YYYYMMDD_HHMMSS/
```

包含：
- 收敛速度图表
- 平均奖励图表  
- 训练曲线
- 灾难性遗忘矩阵和表格
- 文本报告

## CartPole对比
CartPole在`cartpole_cl_project/`，功能相同








