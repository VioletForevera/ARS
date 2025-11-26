# MountainCar Continuous Learning Design

## 环境信息
- **环境名**: MountainCar-v1
- **状态空间**: 2维 (position, velocity)
- **动作空间**: 3个离散动作 (0: left, 1: noop, 2: right)
- **目标**: 让小车克服重力达到右侧山顶（目标位置0.5）

## 任务定义

基于两个参数的组合定义4个任务（类似CartPole的杆长+风力）：

### 参数1: 重力系数 (gravity)
- **标准**: 0.0025
- **增强**: 0.0035 (增加难度)

### 参数2: 推力因子 (force_mag)  
- **标准**: 0.001
- **增强**: 0.0015 (增加推力，降低难度)

### 任务列表

| ID | 名称 | gravity | force_mag | 难度 | 说明 |
|----|------|---------|-----------|------|------|
| 0 | 标准MountainCar | 0.0025 | 0.001 | 标准 | 原始环境 |
| 1 | T1: 标准重力+标准推力 | 0.0025 | 0.001 | 标准 | 参考任务 |
| 2 | T2: 增强重力+标准推力 | 0.0035 | 0.001 | 困难 | 重力更强，更难爬坡 |
| 3 | T3: 标准重力+增强推力 | 0.0025 | 0.0015 | 容易 | 推力更大，更容易 |
| 4 | T4: 增强重力+增强推力 | 0.0035 | 0.0015 | 中等 | 推力补偿部分重力 |

## 指标定义

### 1. Convergence Speed
- **目标性能**: 连续10回合平均奖励 >= -110
- **注**: MountainCar标准奖励是负数，好成绩接近-110（较短的episode完成）
- **测量**: 每个任务达到目标性能所需的episode数

### 2. Average Reward
- **固定episodes**: 500
- **测量**: 每个任务的平均奖励（跨所有episodes）
- **注**: 奖励通常为负，更接近0越好

### 3. Catastrophic Forgetting (CF)
- **计算方式**: CF = after_performance - before_performance
- **before_performance**: 任务训练完成时的性能
- **after_performance**: 训练新任务后，该任务的当前性能
- **解释**:
  - 正数 (+): 正向迁移（性能提升）
  - 负数 (-): 灾难性遗忘（性能下降）

## 推荐结构

```
cl_rl_framework/
├── mountaincar_cl/          # MountainCar持续学习代码
│   ├── config/
│   │   └── mountaincar_config.yaml
│   ├── run_mountaincar.py   # 主训练脚本
│   ├── environments/
│   │   ├── mountaincar_cl.py  # MountainCar变体实现
│   │   └── task_scheduler.py  # 任务调度器
│   ├── agents/              # RL算法
│   │   └── dqn_agent.py    # (可复用CartPole的DQN)
│   ├── core/                # 持续学习机制
│   │   ├── metrics.py       # 指标跟踪
│   │   ├── ewc_consolidator.py
│   │   └── entropy_trigger.py
│   ├── utils/
│   │   ├── logger.py
│   │   └── visualization.py
│   └── requirements.txt
```

## 关键实现点

### 1. MountainCar环境变体
需要修改重力系数和推力因子：
```python
# 伪代码
def make_mountaincar_variant(gravity=0.0025, force_mag=0.001):
    env = gym.make('MountainCar-v1')
    # 修改env的物理参数
    env.gravity = gravity
    env.force_mag = force_mag
    return env
```

### 2. 训练脚本
复用CartPole的训练框架，只需：
- 修改环境导入
- 调整收敛阈值（-110而不是200）
- 使用相同的DQN算法
- 保持相同的指标收集逻辑

### 3. 配置示例
```yaml
tasks:
  - id: 0
    name: "标准MountainCar"
    gravity: 0.0025
    force_mag: 0.001
  - id: 1
    name: "T1: 标准重力+标准推力"
    gravity: 0.0025
    force_mag: 0.001
  - id: 2
    name: "T2: 增强重力+标准推力"
    gravity: 0.0035
    force_mag: 0.001
  - id: 3
    name: "T3: 标准重力+增强推力"
    gravity: 0.0025
    force_mag: 0.0015
  - id: 4
    name: "T4: 增强重力+增强推力"
    gravity: 0.0035
    force_mag: 0.0015
```

## 预期结果

1. **环境1 (CartPole-v1)**: ✅ 已完成
2. **环境2 (MountainCar-v1)**: 待实现
3. **两个POC**: 都使用相同的DQN算法，但针对不同的环境特性
4. **对比分析**: 两个环境的CF模式、收敛速度等差异

## 下一步

1. 创建 `cl_rl_framework/mountaincar_cl/` 目录结构
2. 实现MountainCar环境变体包装器
3. 配置任务定义（4-5个任务）
4. 复用DQN训练代码
5. 实现指标收集和可视化
6. 运行实验并生成对比报告








