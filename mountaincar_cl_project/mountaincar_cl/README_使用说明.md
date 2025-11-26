# MountainCar Continuous Learning Framework

MountainCar环境持续学习框架，完全镜像CartPole的结构和功能。

## 项目结构

```
mountaincar_cl/
├── config/
│   └── mountaincar_config.yaml      # 配置文件（定义5个任务）
├── environments/
│   ├── mountaincar_cl.py            # MountainCar环境包装器
│   └── task_scheduler.py            # 任务调度器
├── agents/
│   └── dqn_agent.py                 # DQN智能体（与CartPole共用）
├── core/
│   └── metrics.py                   # 指标追踪（与CartPole共用）
├── utils/
│   ├── logger.py
│   └── visualization.py
├── run_mountaincar.py               # 主训练脚本
└── requirements.txt
```

## 关键差异

### 环境参数
- **CartPole**: length（杆长）、wind（风力）、force_mag（推力）
- **MountainCar**: gravity（重力）、force_mag（推力）

### 网络配置
- **输入维度**: 2 (position, velocity)
- **输出维度**: 3 (left, noop, right)

### 收敛标准
- **CartPole**: 平均奖励 >= 200
- **MountainCar**: 平均奖励 >= -110

### 任务定义（5个任务）
- T0: 标准MountainCar
- T1: 标准重力+标准推力
- T2: 增强重力+标准推力（更难）
- T3: 标准重力+增强推力（更容易）
- T4: 增强重力+增强推力

## 使用方法

### 1. 持续学习训练（推荐）
```bash
cd cl_rl_framework/mountaincar_cl
python run_mountaincar.py --train --continual --task-sequence 0 1 2 3 4 --episodes-per-task 500
```

### 2. 单任务训练
```bash
python run_mountaincar.py --train --task-id 1 --train-episodes 500
```

### 3. 演示训练好的模型
```bash
python run_mountaincar.py --demo --task-id 1
```

## 生成的报告

训练完成后，会在`runs/continual_learning_YYYYMMDD_HHMMSS/`目录下生成：

1. **metrics_report.txt** - 文本报告（收敛速度、平均奖励、CF）
2. **metrics_visualization.png** - 2x2综合图表
3. **summary_metrics.png** - 综合指标图表
4. **convergence_speed.png** - 收敛速度图表
5. **reward_statistics.png** - 奖励统计图表
6. **catastrophic_forgetting_table.png** - CF矩阵表格
7. **catastrophic_forgetting_values.png** - CF数值表格（含颜色编码）
8. **metrics.json** - 详细数据
9. **catastrophic_forgetting.json** - CF数据

## 指标说明

### Convergence Speed
- 记录每个任务达到目标性能（平均奖励>=-110）所需的episode数
- 跨任务平均收敛速度

### Average Reward
- 每个任务的平均奖励
- 跨任务平均奖励

### Catastrophic Forgetting
- 使用减法计算：`CF = after_performance - before_performance`
- 负数：灾难性遗忘（性能下降）
- 正数：正向迁移（性能提升）

## 配置文件

修改`config/mountaincar_config.yaml`来定义新的任务变体：

```yaml
tasks:
  - id: 5
    name: "T5: 自定义任务"
    gravity: 0.004    # 调整重力
    force_mag: 0.002  # 调整推力
    description: "描述"
```








