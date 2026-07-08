"""
================================================================================
PINN V2 二维对流-扩散逆问题 (2D Convection-Diffusion Inverse Problem)
================================================================================

物理背景:
---------
对流-扩散方程描述了物质在流体中的传输过程，同时考虑了:
1. 扩散作用 (Diffusion): 物质从高浓度向低浓度移动 (α∇²u)
2. 对流作用 (Convection): 物质被流体携带移动 (v·∇u)

应用场景:
- 污染物在河流中的传播
- 热量在流动介质中的传递
- 化学物质在反应器中的分布

PDE 公式:
---------
    ∂u/∂t - α∇²u + v·∇u = Q

其中:
    - u(x,y,t): 温度/浓度场
    - α: 扩散系数 (Diffusion coefficient)
    - v = (vx, vy): 对流速度向量 (Convection velocity)
    - Q: 源项 (Source term, 从制造解推导)

待反演参数 (3个):
-----------------
    - α (alpha): 扩散系数
    - vx: x方向对流速度
    - vy: y方向对流速度

难点说明:
---------
1. 参数耦合: α 和 (vx, vy) 对解的影响相互交织
2. 方向性: 对流项引入了流动方向，打破了扩散的各向同性
3. 梯度复杂: 需要同时计算 u_x 和 u_y 用于对流项
4. 三参数同时反演: 搜索空间维度更高，优化更困难

制造解:
-------
    u(x,y,t) = exp(-t) sin(πx) sin(πy)

边界条件: Dirichlet u = 0 (四边)
初始条件: u(x,y,0) = sin(πx) sin(πy)

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Tuple, Any
from .base_physics import Physics


class Heat2DConvection(Physics):
    """
    二维对流-扩散逆问题 (2D Convection-Diffusion Inverse Problem)

    PDE: u_t - α∇²u + vx*u_x + vy*u_y = Q

    待反演参数:
        - α (alpha): 扩散系数
        - vx: x方向对流速度
        - vy: y方向对流速度

    这是 2D 任务中最难的逆问题:
    - 3个参数同时反演
    - 对流项引入方向性
    - 参数之间存在耦合效应

    特别适合展示迁移学习在复杂问题上的优势。
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        初始化物理模型

        Args:
            config: 配置字典，包含:
                - params: 初始参数猜测 {'alpha': 初始值, 'vx': 初始值, 'vy': 初始值}
                - true_params: 真实参数 {'alpha': 真实值, 'vx': 真实值, 'vy': 真实值}
        """
        super().__init__()

        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 域定义: [0,1]³ (x, y, t)
        self.x_min, self.x_max = 0.0, 1.0
        self.y_min, self.y_max = 0.0, 1.0
        self.t_min, self.t_max = 0.0, 1.0
        self.domain = [0.0, 1.0]

        # 真实参数 (用于误差计算和制造解)
        # 默认值: α=0.05, vx=0.5, vy=0.3
        self.true_params = config.get('true_params', {
            'alpha': 0.05,
            'vx': 0.5,
            'vy': 0.3
        })

        # 初始化可学习参数
        # 使用逆 Softplus 变换确保 α > 0
        # vx, vy 可正可负，直接使用原值
        init_params = config.get('params', {
            'alpha': 0.1,
            'vx': 0.1,
            'vy': 0.1
        })

        # α 使用 Softplus 约束 (必须 > 0)
        alpha_init = self._inv_softplus(init_params['alpha'])
        self.alpha = nn.Parameter(
            torch.tensor([alpha_init], dtype=torch.float32, device=self.device)
        )

        # vx, vy 无约束 (可正可负，表示流动方向)
        self.vx = nn.Parameter(
            torch.tensor([init_params['vx']], dtype=torch.float32, device=self.device)
        )
        self.vy = nn.Parameter(
            torch.tensor([init_params['vy']], dtype=torch.float32, device=self.device)
        )

    def _inv_softplus(self, x: float) -> float:
        """逆 Softplus 变换: y = log(exp(x) - 1)"""
        if x <= 0:
            raise ValueError(f"初始参数必须为正数 (Softplus 约束): {x}")
        return np.log(np.exp(x) - 1)

    def residual(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        计算 PDE 残差

        PDE: u_t - α(u_xx + u_yy) + vx*u_x + vy*u_y - Q = 0

        Args:
            model: 神经网络模型
            x, y, t: 坐标张量

        Returns:
            PDE 残差张量
        """
        # 启用梯度计算
        x.requires_grad_(True)
        y.requires_grad_(True)
        t.requires_grad_(True)

        # 神经网络预测
        u = model(x, y, t)

        # 计算一阶导数
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        # 计算二阶导数 (扩散项)
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]

        # 可学习参数
        alpha_eff = F.softplus(self.alpha)  # α > 0 (Softplus 约束)
        vx_eff = self.vx  # vx 无约束
        vy_eff = self.vy  # vy 无约束

        # 源项 Q (从制造解推导)
        Q = self._compute_source(x, y, t)

        # PDE 残差: u_t - α∇²u + v·∇u - Q = 0
        # 展开: u_t - α(u_xx + u_yy) + vx*u_x + vy*u_y - Q = 0
        laplacian_u = u_xx + u_yy
        convection = vx_eff * u_x + vy_eff * u_y

        res = u_t - alpha_eff * laplacian_u + convection - Q
        return res

    def _compute_source(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        计算源项 Q (从制造解推导)

        制造解: u = exp(-t) sin(πx) sin(πy)

        导数计算:
            u_t = -exp(-t) sin(πx) sin(πy) = -u
            u_x = π exp(-t) cos(πx) sin(πy)
            u_y = π exp(-t) sin(πx) cos(πy)
            u_xx = -π² exp(-t) sin(πx) sin(πy) = -π² u
            u_yy = -π² exp(-t) sin(πx) sin(πy) = -π² u

        PDE: u_t - α(u_xx + u_yy) + vx*u_x + vy*u_y = Q

        代入:
            Q = -u - α(-π²u - π²u) + vx * π exp(-t) cos(πx) sin(πy)
                                   + vy * π exp(-t) sin(πx) cos(πy)
              = -u + 2απ²u + π exp(-t) [vx cos(πx) sin(πy) + vy sin(πx) cos(πy)]
        """
        alpha_true = self.true_params['alpha']
        vx_true = self.true_params['vx']
        vy_true = self.true_params['vy']
        pi = math.pi

        # 制造解
        u = self.analytical_solution(x, y, t)
        exp_t = torch.exp(-t)

        # 一阶导数 (用于对流项)
        u_x = pi * exp_t * torch.cos(pi * x) * torch.sin(pi * y)
        u_y = pi * exp_t * torch.sin(pi * x) * torch.cos(pi * y)

        # 时间导数
        u_t = -u

        # 拉普拉斯算子
        laplacian_u = -2 * (pi**2) * u

        # 源项 Q = u_t - α∇²u + v·∇u
        Q = u_t - alpha_true * laplacian_u + vx_true * u_x + vy_true * u_y

        return Q.detach()  # 作为常数处理，不参与梯度计算

    def analytical_solution(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        制造解 (Manufactured Solution)

        u(x,y,t) = exp(-t) sin(πx) sin(πy)

        特点:
        - 满足 Dirichlet 边界条件 (四边 u=0)
        - 初始条件为 sin(πx)sin(πy)
        - 随时间指数衰减
        """
        pi = math.pi
        return torch.exp(-t) * torch.sin(pi * x) * torch.sin(pi * y)

    def sample_pde(self, n: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """在 PDE 域内随机采样配点 (Collocation Points)"""
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
        采样边界条件点 (四边 Dirichlet BC)

        边界:
        - Left:   x=0, y∈[0,1], t∈[0,1]
        - Right:  x=1, y∈[0,1], t∈[0,1]
        - Bottom: y=0, x∈[0,1], t∈[0,1]
        - Top:    y=1, x∈[0,1], t∈[0,1]
        """
        n_side = n // 4

        # Left (x=0)
        t1 = torch.rand(n_side, 1, device=device)
        y1 = torch.rand(n_side, 1, device=device)
        x1 = torch.zeros(n_side, 1, device=device)

        # Right (x=1)
        t2 = torch.rand(n_side, 1, device=device)
        y2 = torch.rand(n_side, 1, device=device)
        x2 = torch.ones(n_side, 1, device=device)

        # Bottom (y=0)
        t3 = torch.rand(n_side, 1, device=device)
        x3 = torch.rand(n_side, 1, device=device)
        y3 = torch.zeros(n_side, 1, device=device)

        # Top (y=1)
        t4 = torch.rand(n_side, 1, device=device)
        x4 = torch.rand(n_side, 1, device=device)
        y4 = torch.ones(n_side, 1, device=device)

        return [(x1, y1, t1), (x2, y2, t2), (x3, y3, t3), (x4, y4, t4)]

    def sample_data(self, n: int, device: torch.device) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
        """
        采样观测数据点 (用于反演训练)

        Returns:
            ((x, y, t), u_true): 输入坐标元组和真实解
        """
        x = (self.x_max - self.x_min) * torch.rand(n, 1, device=device) + self.x_min
        y = (self.y_max - self.y_min) * torch.rand(n, 1, device=device) + self.y_min
        t = (self.t_max - self.t_min) * torch.rand(n, 1, device=device) + self.t_min

        u_true = self.analytical_solution(x, y, t)
        noise_level = self.config.get('noise_level', 0.0)
        if noise_level > 0:
            noise = noise_level * torch.randn_like(u_true) * torch.abs(u_true).mean()
            u_noisy = u_true + noise
            return (x, y, t), u_noisy

        return (x, y, t), u_true

    def get_param_errors(self) -> Dict[str, float]:
        """
        计算并返回参数反演误差

        Returns:
            字典包含:
            - alpha, alpha_error: 扩散系数及其相对误差
            - vx, vx_error: x方向对流速度及其相对误差
            - vy, vy_error: y方向对流速度及其相对误差
        """
        alpha_val = F.softplus(self.alpha).item()
        vx_val = self.vx.item()
        vy_val = self.vy.item()

        alpha_true = self.true_params['alpha']
        vx_true = self.true_params['vx']
        vy_true = self.true_params['vy']

        # 计算相对误差 (%)
        alpha_err = abs(alpha_val - alpha_true) / abs(alpha_true) * 100
        vx_err = abs(vx_val - vx_true) / abs(vx_true) * 100 if vx_true != 0 else abs(vx_val) * 100
        vy_err = abs(vy_val - vy_true) / abs(vy_true) * 100 if vy_true != 0 else abs(vy_val) * 100

        return {
            'alpha': alpha_val,
            'alpha_error': alpha_err,
            'vx': vx_val,
            'vx_error': vx_err,
            'vy': vy_val,
            'vy_error': vy_err
        }
