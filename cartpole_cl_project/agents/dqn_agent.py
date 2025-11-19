"""
DQN智能代理实现
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from typing import Dict, Any, List
from .base import BaseAgent


class DQNNetwork(nn.Module):
    """DQN网络结构"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)


class DQNAgent(BaseAgent):
    """
    DQN智能代理    Deep Q-Network
    
    实现Deep Q-Network算法
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 网络参数
        self.hidden_dim = config.get('hidden_dim', 64)
        self.lr = config.get('lr', 0.001)#Learning Rate 
        self.gamma = config.get('gamma', 0.99)#Discount Factor #控制未来奖励的重要性
        self.epsilon = config.get('epsilon', 0.1)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.batch_size = config.get('batch_size', 32)#Number of samples used in each training session
        self.memory_size = config.get('memory_size', 10000)
        
        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #Create two networks
        # 创建网络#一个是Q网络（用于策略评估和更新）#另一个是目标网络（用于提供稳定的目标Q值）
        self.q_network = DQNNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.target_network = DQNNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        
        # 初始化经验回放缓冲区，存储历史交互数据
        self.memory = deque(maxlen=self.memory_size)
        
        # 更新目标网络
        self.update_target_network()
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """选择动作"""                  """Select Action"""
        if training and np.random.random() < self.epsilon:#AI有10%的机会会"随便试试"
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor) #用Q网络计算每个动作的分数
            action = q_values.argmax().item()       #选择分数最高的那个动作
        
        return action
    
    def observe(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool) -> None:
        """观察经验""" """每次交互 (state, action, reward, next_state, done) 被存入 memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def update(self) -> Dict[str, float]:
        """更新策略"""
        if len(self.memory) < self.batch_size:
            return {}
        
        # 采样批次数据   #Sampling batch data
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states = torch.FloatTensor([self.memory[i][0] for i in batch]).to(self.device)
        actions = torch.LongTensor([self.memory[i][1] for i in batch]).to(self.device)
        rewards = torch.FloatTensor([self.memory[i][2] for i in batch]).to(self.device)
        next_states = torch.FloatTensor([self.memory[i][3] for i in batch]).to(self.device)
        dones = torch.BoolTensor([self.memory[i][4] for i in batch]).to(self.device)
        
        # 计算当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # 计算损失
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # 反向传播       #Back propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return {'loss': loss.item(), 'epsilon': self.epsilon}
    
    def update_target_network(self):
        """更新目标网络""" """使目标Q值在一段时间内保持相对稳定"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, filepath: str) -> None:
        """Save Model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, filepath)
    
    def load(self, filepath: str) -> None:
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def load_model(self, model_path: str) -> None:
        """加载训练好的模型

        Args:
            model_path: 模型文件路径
        """
        try:
            checkpoint = torch.load(model_path)
            if isinstance(checkpoint, dict):
                # 检查不同的保存格式
                if 'q_network_state_dict' in checkpoint:
                    self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
                elif 'model_state_dict' in checkpoint:
                    self.q_network.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.q_network.load_state_dict(checkpoint)
            else:
                self.q_network = checkpoint
            
            self.q_network.eval()  # 设置为评估模式
            print(f"成功加载模型: {model_path}")
        
        except Exception as e:
            print(f"加载模型失败: {str(e)}")
            if isinstance(checkpoint, dict):
                print("可用的键:", list(checkpoint.keys()))
            raise















