"""
CartPole持续学习环境封装
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple, Optional


class CartPoleCL:
    """
    CartPole持续学习环境
    
    支持不同变体的CartPole任务：
    - 不同杆长
    - 不同风力
    - 不同推力
    """
    
    def __init__(self, render_mode: Optional[str] = None):
        self.render_mode = render_mode
        self.env = None
        self.current_variant = None
        
    def make_env(self, render: bool = False, seed: Optional[int] = None) -> gym.Env:
        """创建环境实例，可选指定种子"""
        render_mode = "human" if render else None
        self.env = gym.make("CartPole-v1", render_mode=render_mode)
        if seed is not None:
            self.env.reset(seed=seed)
        # 如果之前已经设置了variant，现在应用它
        if self.current_variant is not None:
            self._apply_variant()
        return self.env
    
    def _enable_wind_patch(self):
        """为当前env补丁，使原始动力学加入风力扰动。风力大小: wind * force_mag"""
        if self.env is None:
            return
        envu = self.env.unwrapped
        # 防止重复打补丁
        if hasattr(envu, '__wind_patched') and envu.__wind_patched:
            return
        def patched_step(action):
            # 用 length 替代 force_mag
            length = getattr(envu, 'length', 0.5)
            wind = 0.0
            variant = getattr(self, 'current_variant', None)
            if variant is not None:
                wind = variant.get('wind', 0.0)
            wind_force = wind * length
            # 判断动作决定的原始力方向
            force = length if int(action) == 1 else -length
            total_force = force + wind_force
            # == 拷贝 CartPole 的动力学 ==
            x, x_dot, theta, theta_dot = envu.state
            costheta = np.cos(theta)
            sintheta = np.sin(theta)
            total_mass = envu.total_mass
            polemass_length = envu.polemass_length
            gravity = envu.gravity
            l = envu.length
            masspole = envu.masspole
            tau = getattr(envu, 'tau', 0.02)
            temp = (total_force + polemass_length * theta_dot ** 2 * sintheta) / total_mass
            thetaacc = (gravity * sintheta - costheta * temp) / (
                l * (4.0 / 3.0 - masspole * costheta ** 2 / total_mass)
            )
            xacc = temp - polemass_length * thetaacc * costheta / total_mass
            x = x + tau * x_dot
            x_dot = x_dot + tau * xacc
            theta = theta + tau * theta_dot
            theta_dot = theta_dot + tau * thetaacc
            envu.state = (x, x_dot, theta, theta_dot)
            done = (
                x < -envu.x_threshold
                or x > envu.x_threshold
                or theta < -envu.theta_threshold_radians
                or theta > envu.theta_threshold_radians
            )
            terminated = bool(done)
            truncated = False
            reward = 1.0 if not done else 0.0
            obs = np.array(envu.state, dtype=np.float32)
            info = {}
            return obs, reward, terminated, truncated, info
        envu.step = patched_step
        envu.__wind_patched = True

    def _apply_variant(self):
        """应用已保存的variant参数到环境"""
        if self.env is None or self.current_variant is None:
            return
        length = self.current_variant['length']
        env_unwrapped = self.env.unwrapped
        env_unwrapped.length = length
        env_unwrapped.gravity = 9.8
        env_unwrapped.total_mass = env_unwrapped.masspole + env_unwrapped.masscart
        env_unwrapped.polemass_length = env_unwrapped.masspole * env_unwrapped.length
        self._enable_wind_patch()
    
    def set_variant(self, length: float = 0.5, wind: float = 0.0) -> None:
        """设置环境变体参数(支持风力补丁)"""
        # 先保存参数
        self.current_variant = {
            'length': length,
            'wind': wind
        }
        # 如果环境已创建，立即应用
        if self.env is not None:
            self._apply_variant()
    
    def reset(self, *, seed: Optional[int]=None):
        """重置环境，并可显式指定种子"""
        if self.env is None:
            raise RuntimeError("请先调用make_env()创建环境")
        return self.env.reset(seed=seed)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行动作"""
        if self.env is None:
            raise RuntimeError("请先调用make_env()创建环境")
        return self.env.step(action)
    
    def render(self):
        """渲染环境"""
        if self.env is not None:
            return self.env.render()
    
    def close(self):
        """关闭环境"""
        if self.env is not None:
            self.env.close()
            self.env = None
    
    def get_observation_space(self):
        """获取观测空间"""
        if self.env is None:
            raise RuntimeError("请先调用make_env()创建环境")
        return self.env.observation_space
    
    def get_action_space(self):
        """获取动作空间"""
        if self.env is None:
            raise RuntimeError("请先调用make_env()创建环境")
        return self.env.action_space
