# CartPole强化学习项目使用说明

## 🎯 项目功能

这个项目实现了一个完整的CartPole强化学习系统，包括：
- 智能体训练（使用DQN算法）
- 训练过程详细输出
- 模型自动保存
- 训练完成后动画演示

## 🚀 快速开始

### 方法1：一键运行（推荐）
```bash
python run_training.py
```
这会自动完成训练和演示，无需手动输入。

### 方法2：分步运行

#### 1. 训练模型
```bash
python run_cartpole.py --train --train-episodes 200
```

#### 2. 演示训练好的模型
```bash
python run_cartpole.py --demo
```

#### 3. 训练后立即演示
```bash
python run_cartpole.py --train --demo
```

## 📊 训练输出说明

训练时会显示：
- **回合信息**：回合数、奖励、步数、epsilon值、平均奖励
- **训练统计**：最终平均奖励、最高奖励、成功率、总回合数
- **模型保存**：确认模型已保存到指定路径

## 🎬 演示功能

演示时会显示：
- **模型加载**：确认训练好的模型加载成功
- **实时状态**：位置、角度、奖励信息
- **演示统计**：平均奖励、最高奖励、演示回合数

## ⚙️ 参数说明

### 训练参数
- `--train-episodes`：训练回合数（默认1000）
- `--train`：开始训练模式

### 演示参数
- `--episodes`：演示回合数（默认1，因为相同模型结果相似）
- `--demo`：开始演示模式
- `--model-path`：模型文件路径（默认runs/trained_model.pth）

## 📁 文件结构

```
cartpole_cl_project/
├── run_cartpole.py          # 主训练和演示脚本
├── run_training.py          # 一键运行脚本
├── runs/                    # 模型保存目录
│   └── trained_model.pth   # 训练好的模型
└── README_使用说明.md       # 本说明文档
```

## 🎮 使用示例

### 基础训练
```bash
# 训练200回合
python run_cartpole.py --train --train-episodes 200

# 演示（默认1个回合，因为相同模型结果相似）
python run_cartpole.py --demo
```

### 高级用法
```bash
# 训练500回合，演示5个回合
python run_cartpole.py --train --train-episodes 500 --demo --episodes 5

# 使用自定义模型路径
python run_cartpole.py --demo --model-path "my_model.pth"
```

## 🔧 故障排除

### 1. 路径问题
确保在正确的目录下运行：
```bash
cd "C:\Users\panha\Desktop\ARS CW\cartpole_cl_project"
```

### 2. 模型文件不存在
如果提示模型文件不存在，请先运行训练：
```bash
python run_cartpole.py --train
```

### 3. 编码问题
如果遇到Unicode编码问题，代码已经移除了emoji字符，应该可以正常运行。

## 📈 性能指标

- **目标性能**：平均奖励达到475以上
- **训练时间**：200回合约需1-2分钟
- **模型大小**：约几MB
- **成功率**：训练完成后通常能达到90%以上

## 🎯 项目特点

✅ **训练时详细输出**：每回合显示奖励、步数、epsilon等统计信息  
✅ **自动模型保存**：训练完成后自动保存到runs目录  
✅ **动画演示**：只在训练完成后展示智能体平衡动画  
✅ **用户友好**：清晰的命令行界面和错误提示  
✅ **无交互依赖**：支持自动化运行，无需手动输入  

## 🎉 完成！

现在您可以：
1. 运行 `python run_training.py` 一键体验完整流程
2. 使用 `python run_cartpole.py --train` 开始训练
3. 使用 `python run_cartpole.py --demo` 观看演示

享受您的CartPole强化学习之旅！🚀

