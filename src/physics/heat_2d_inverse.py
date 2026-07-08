"""
PINN V2 二维热传导逆问题物理类 (2D Heat Inverse Problem)

方程: u_t = α(u_xx + u_yy)
域: x, y ∈ [0, 1], t ∈ [0, 1]
待反演参数: α (热扩散系数)

解析解: u(x,y,t) = exp(-2π²αt) sin(πx) sin(πy)
边界条件: Dirichlet u = 0 (四边)
初始条件: u(x,y,0) = sin(πx) sin(πy)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from .base_physics import Physics


class Heat2DInverse(Physics):
    """
    二维热传导逆问题 (2D Heat Inverse Problem)

    PDE: u_t = α(u_xx + u_yy)
    待反演参数: α (热扩散系数)

    使用 Softplus 约束确保 α > 0 (LBFGS 友好)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        初始化物理模型

        Args:
            config: 配置字典，包含:
                - params: 初始参数猜测 {'alpha': 初始值}
                - true_params: 真实参数 {'alpha': 真实值}
                - domain: 空间域 [min, max]
        """
        super().__init__()

        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 域定义
        self.x_min, self.x_max = 0.0, 1.0
        self.y_min, self.y_max = 0.0, 1.0
        self.t_min, self.t_max = 0.0, 1.0
        self.domain = [0.0, 1.0]

        # 真实参数 (用于误差计算)
        self.true_params = config.get('true_params', {'alpha': 0.05})

        # 初始化可学习参数 (使用逆 Softplus 变换)
        init_params = config.get('params', {'alpha': 0.1})
        alpha_init = self._inv_softplus(init_params['alpha'])
        self.alpha = nn.Parameter(
            torch.tensor([alpha_init], dtype=torch.float32, device=self.device)
        )

    def _inv_softplus(self, x: float) -> float:
        """逆 Softplus 变换: x = log(exp(y) - 1)"""
        if x <= 0:
            raise ValueError(f"初始参数必须为正数 (Softplus 约束). 当前值: {x}")
        return np.log(np.exp(x) - 1)

    def residual(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        计算 PDE 残差: u_t - α(u_xx + u_yy)

        Args:
            model: 神经网络模型
            x, y, t: 坐标张量

        Returns:
            PDE 残差张量
        """
        # 启用梯度
        x.requires_grad_(True)
        y.requires_grad_(True)
        t.requires_grad_(True)

        u = model(x, y, t)

        # 计算导数
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]

        # 可学习参数 (Softplus 确保正数)
        alpha_eff = F.softplus(self.alpha)

        # PDE 残差
        res = u_t - alpha_eff * (u_xx + u_yy)
        return res

    def analytical_solution(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        解析解: u(x,y,t) = exp(-2π²αt) sin(πx) sin(πy)

        注意: 使用真实参数 α_true
        """
        alpha_true = self.true_params['alpha']
        pi = math.pi
        return torch.exp(-2 * (pi**2) * alpha_true * t) * torch.sin(pi * x) * torch.sin(pi * y)

    def sample_pde(self, n: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """在 PDE 域内采样配点"""
        x = (self.x_max - self.x_min) * torch.rand(n, 1, device=device) + self.x_min
        y = (self.y_max - self.y_min) * torch.rand(n, 1, device=device) + self.y_min
        t = (self.t_max - self.t_min) * torch.rand(n, 1, device=device) + self.t_min
        return x, y, t

    def sample_ic(self, n: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """采样初始条件点 (t=0)"""
        x = (self.x_max - self.x_min) * torch.rand(n, 1, device=device) + self.x_min
        y = (self.y_max - self.y_min) * torch.rand(n, 1, device=device) + self.y_min
        t = torch.zeros(n, 1, device=device)
        return x, y, t

    def sample_bc(self, n: int, device: torch.device) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        采样边界条件点 (四边 Dirichlet)

        边界: left(x=0), right(x=1), bottom(y=0), top(y=1)
        """
        n_side = n // 4

        # 1. Left (x=0)
        t1 = torch.rand(n_side, 1, device=device)
        y1 = torch.rand(n_side, 1, device=device)
        x1 = torch.zeros(n_side, 1, device=device)

        # 2. Right (x=1)
        t2 = torch.rand(n_side, 1, device=device)
        y2 = torch.rand(n_side, 1, device=device)
        x2 = torch.ones(n_side, 1, device=device) * self.x_max

        # 3. Bottom (y=0)
        t3 = torch.rand(n_side, 1, device=device)
        x3 = torch.rand(n_side, 1, device=device)
        y3 = torch.zeros(n_side, 1, device=device)

        # 4. Top (y=1)
        t4 = torch.rand(n_side, 1, device=device)
        x4 = torch.rand(n_side, 1, device=device)
        y4 = torch.ones(n_side, 1, device=device) * self.y_max

        return [(x1, y1, t1), (x2, y2, t2), (x3, y3, t3), (x4, y4, t4)]

    def sample_data(self, n: int, device: torch.device) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
        """
        采样数据点用于反演训练

        Returns:
            ((x, y, t), u_true): 输入坐标元组和真实解
        """
        x = (self.x_max - self.x_min) * torch.rand(n, 1, device=device) + self.x_min
        y = (self.y_max - self.y_min) * torch.rand(n, 1, device=device) + self.y_min
        t = (self.t_max - self.t_min) * torch.rand(n, 1, device=device) + self.t_min

        u_true = self.analytical_solution(x, y, t)

        # 添加观测噪声
        noise_level = self.config.get('noise_level', 0.05)
        if noise_level > 0:
            noise = noise_level * torch.randn_like(u_true) * torch.abs(u_true).mean()
            u_noisy = u_true + noise
            return (x, y, t), u_noisy

        return (x, y, t), u_true

    def get_param_errors(self) -> Dict[str, float]:
        """计算参数误差"""
        alpha_val = F.softplus(self.alpha).item()
        alpha_true = self.true_params['alpha']

        return {
            'alpha': alpha_val,
            'alpha_error': abs(alpha_val - alpha_true) / abs(alpha_true) * 100
        }
