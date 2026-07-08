import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from ..utils.plotter import Plotter
from ..utils.logger import Logger

class Trainer:
    """
    PINN V2 核心训练器 (Core Trainer)
    负责调度模型训练、物理采样、损失计算及优化过程。
    严格遵守 V2 架构规范。
    """
    def __init__(self, model: nn.Module, physics: nn.Module, config: Dict[str, Any], logger: Logger):
        self.model = model
        self.physics = physics
        self.config = config
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # [Fix] Explicitly set device context for Windows/CUDA 12.8
        if self.device.type == 'cuda':
            torch.cuda.set_device(0)

        # 移动到设备
        self.model.to(self.device)
        self.physics.to(self.device)

        # 训练配置
        self.train_config = config['training']
        self.domain = config['physics']['domain'] # [min, max]

        # 优化器参数 (将会在 train() 中初始化)
        # 注意: 我们需要同时优化网络参数和物理参数 (Inverse Problem)
        self.params_to_optimize = list(self.model.parameters()) + list(self.physics.parameters())

        # 初始化绘图器
        self.plotter = Plotter(logger.get_save_dir())
        self._extra_loss_hooks = []
        self._fixed_data_pack = None
        self._fixed_data_pool_size = None

    def _infer_batch_size(self, batch_like: Any) -> Optional[int]:
        """Infer the leading batch dimension from a tensor-like structure."""
        if batch_like is None:
            return None
        if torch.is_tensor(batch_like):
            return int(batch_like.shape[0]) if batch_like.ndim > 0 else None
        if isinstance(batch_like, (list, tuple)):
            for item in batch_like:
                inferred = self._infer_batch_size(item)
                if inferred is not None:
                    return inferred
            return None
        if isinstance(batch_like, dict):
            for value in batch_like.values():
                inferred = self._infer_batch_size(value)
                if inferred is not None:
                    return inferred
            return None
        return None

    def _index_batch_like(self, batch_like: Any, indices: torch.Tensor) -> Any:
        """Slice a tensor-like structure on the leading dimension."""
        if batch_like is None:
            return None
        if torch.is_tensor(batch_like):
            return batch_like.index_select(0, indices.to(batch_like.device))
        if isinstance(batch_like, tuple):
            return tuple(self._index_batch_like(item, indices) for item in batch_like)
        if isinstance(batch_like, list):
            return [self._index_batch_like(item, indices) for item in batch_like]
        if isinstance(batch_like, dict):
            return {key: self._index_batch_like(value, indices) for key, value in batch_like.items()}
        raise TypeError(f'Unsupported batch container type: {type(batch_like)}')

    def _resolve_fixed_data_pool_size(self, batch_size_data: int) -> int:
        configured = int(self.train_config.get('fixed_data_size', batch_size_data))
        return max(batch_size_data, configured)

    def _sample_data_points(self, batch_size_data: int) -> Tuple[Any, Any]:
        """Sample data points, optionally from a cached fixed noisy observation pool."""
        if not self.train_config.get('fixed_data_loss', False):
            return self.physics.sample_data(batch_size_data, self.device)

        pool_size = self._resolve_fixed_data_pool_size(batch_size_data)
        if self._fixed_data_pack is None or self._fixed_data_pool_size != pool_size:
            self._fixed_data_pack = self.physics.sample_data(pool_size, self.device)
            self._fixed_data_pool_size = pool_size
            print(f"[INFO] Fixed noisy data enabled: pool_size={pool_size}, batch_size={batch_size_data}")

        data_inputs, data_targets = self._fixed_data_pack
        total_available = self._infer_batch_size(data_targets)
        if total_available is None:
            total_available = self._infer_batch_size(data_inputs)

        if total_available is None or batch_size_data >= total_available:
            return data_inputs, data_targets

        indices = torch.randperm(total_available, device=self.device)[:batch_size_data]
        return (
            self._index_batch_like(data_inputs, indices),
            self._index_batch_like(data_targets, indices),
        )

    def _sample_all(self) -> Tuple[Any, Any, List[Any], Tuple[Any, Any]]:
        """
        全量采样委托给物理引擎 (Delegate sampling to Physics Engine)
        返回: (pde_points, ic_points, bc_points, data_points)
        """
        batch_size_pde = self.train_config['batch_size_pde']
        batch_size_bc = self.train_config['batch_size_bc']
        batch_size_ic = self.train_config['batch_size_ic']

        # 1. PDE 采样
        pde_points = self.physics.sample_pde(batch_size_pde, self.device)

        # 2. IC 采样
        ic_points = self.physics.sample_ic(batch_size_ic, self.device)

        # 3. BC 采样 (返回元组列表，因为可能有多个边界条件)
        bc_points = self.physics.sample_bc(batch_size_bc, self.device)

        # 4. Data 采样 (可选，用于 Inverse Problem)
        data_points = (None, None)
        if self.train_config.get('use_data_loss', False):
            batch_size_data = self.train_config.get('batch_size_data', 500)
            data_points = self._sample_data_points(batch_size_data)

        return pde_points, ic_points, bc_points, data_points

    def register_extra_loss_hook(self, hook, name: Optional[str] = None) -> None:
        """Register a training-time regularization hook."""
        self._extra_loss_hooks.append((name, hook))

    def _invoke_extra_loss_hook(self, hook, points: Tuple):
        try:
            return hook(trainer=self, model=self.model, physics=self.physics, points=points)
        except TypeError as exc:
            try:
                return hook(self.model)
            except TypeError:
                raise exc

    def _compute_extra_losses(self, points: Tuple) -> Dict[str, torch.Tensor]:
        losses = {}

        for item in self._extra_loss_hooks:
            if isinstance(item, tuple):
                default_name, hook = item
            else:
                default_name, hook = None, item

            result = self._invoke_extra_loss_hook(hook, points)
            if result is None:
                continue

            if isinstance(result, dict):
                for name, value in result.items():
                    losses[name] = value if torch.is_tensor(value) else torch.tensor(float(value), device=self.device)
                continue

            if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], str):
                name, value = result
                losses[name] = value if torch.is_tensor(value) else torch.tensor(float(value), device=self.device)
                continue

            name = default_name or getattr(hook, '__name__', 'extra')
            losses[name] = result if torch.is_tensor(result) else torch.tensor(float(result), device=self.device)

        return losses

    def calculate_loss(self, points: Tuple) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算总损失及各分量"""
        pde_inputs, ic_inputs, bc_inputs_list, data_pack = points

        # 1. PDE 损失
        # 通用解包 (x,t) 或 (x,y,t)
        residual = self.physics.residual(self.model, *pde_inputs)
        loss_pde = torch.mean(residual ** 2)

        # 2. IC 损失
        u_ic_pred = self.model(*ic_inputs)
        u_ic_true = self.physics.analytical_solution(*ic_inputs)
        loss_ic = torch.mean((u_ic_pred - u_ic_true) ** 2)

        # 3. BC 损失 (处理多个边界条件列表)
        loss_bc = torch.tensor(0.0, device=self.device)
        for bc_inp in bc_inputs_list:
            # 委托给物理引擎计算边界残差 (支持 Dirichlet, Robin 等)
            res_b = self.physics.bc_residual(self.model, *bc_inp)
            loss_bc += torch.mean(res_b ** 2)

        # 4. Data 损失 (数据驱动/反演)
        loss_data = torch.tensor(0.0, device=self.device)

        # 检查 data_pack 是否存在且有效
        # 新格式: ((x, t), u_true)
        if data_pack and data_pack[0] is not None:
             d_inputs, d_targets = data_pack

             # 如果 d_inputs 是元组 (x, t)，解包传给模型
             if isinstance(d_inputs, (list, tuple)):
                 u_data_pred = self.model(*d_inputs)
             else:
                 # 单张量输入情况
                 u_data_pred = self.model(d_inputs)

             # 如果提供了目标值 (Inverse 逻辑)，直接使用
             if d_targets is not None:
                 loss_data = torch.mean((u_data_pred - d_targets) ** 2)
             else:
                 # 回退: 使用解析解计算 (Forward 验证逻辑)
                 # 注意: 在纯反演问题中，解析解可能不可用，d_targets 必须提供
                 u_data_true = self.physics.analytical_solution(*d_inputs)
                 if u_data_true is not None:
                     loss_data = torch.mean((u_data_pred - u_data_true) ** 2)

        # 加权求和
        w_pde = self.train_config['weights']['pde']
        w_ic = self.train_config['weights']['ic']
        w_bc = self.train_config['weights']['bc']
        w_data = self.train_config['weights'].get('data', 0.0)

        total_loss = w_pde * loss_pde + w_ic * loss_ic + w_bc * loss_bc + w_data * loss_data

        # 5. 参数正则化损失 (如果 physics 提供了该方法)
        loss_reg = torch.tensor(0.0, device=self.device)
        w_reg = self.train_config['weights'].get('reg', 0.0)
        if hasattr(self.physics, 'get_regularization_loss'):
            if w_reg > 0:
                loss_reg = self.physics.get_regularization_loss()
                total_loss = total_loss + w_reg * loss_reg

        extra_losses = self._compute_extra_losses(points)
        if extra_losses:
            total_extra = torch.tensor(0.0, device=self.device)
            for value in extra_losses.values():
                total_extra = total_extra + value
            total_loss = total_loss + total_extra
        else:
            total_extra = torch.tensor(0.0, device=self.device)

        loss_components = {
            'pde': loss_pde.item(),
            'ic': loss_ic.item(),
            'bc': loss_bc.item(),
            'data': loss_data.item(),
        }
        if w_reg > 0:
            loss_components['reg'] = loss_reg.item()
        if extra_losses:
            loss_components['extra'] = total_extra.item()
            for name, value in extra_losses.items():
                loss_components[name] = value.item()

        return total_loss, loss_components

    def train(self):
        """主训练循环 (包含原子性保护)"""
        print(f"[INFO] 开始训练: {self.config['experiment_name']}")

        # 计时开始
        start_time_total = time.time()

        try:
            # --- 第一阶段: Adam (探索) ---
            epochs_adam = self.train_config['epochs_adam']
            base_lr = self.train_config['learning_rate']

            # [可选] 分组学习率: 物理参数用不同的学习率
            # physics_lr_ratio: 物理参数 lr = base_lr * ratio (默认 1.0 = 不分组)
            # Ill-conditioned inverse problems may benefit from a smaller physics learning rate.
            physics_lr_ratio = self.train_config.get('physics_lr_ratio', 1.0)

            if physics_lr_ratio != 1.0:
                physics_param_list = list(self.physics.parameters())
                model_param_list = list(self.model.parameters())
                param_groups = [
                    {'params': model_param_list, 'lr': base_lr},
                    {'params': physics_param_list, 'lr': base_lr * physics_lr_ratio},
                ]
                optimizer_adam = optim.Adam(param_groups)
            else:
                optimizer_adam = optim.Adam(self.params_to_optimize, lr=base_lr)

            # [Optimization] Adam LR Decay
            # 每 100 epoch 衰减 0.9 (或者更平滑的衰减)
            # 这里的逻辑是: 1000 epoch 后从 1e-3 -> 0.3*1e-3 左右
            use_scheduler = True
            if use_scheduler:
                scheduler = optim.lr_scheduler.ExponentialLR(optimizer_adam, gamma=0.998)

            pbar = tqdm(range(epochs_adam), desc="Adam", ascii=True)
            for epoch in pbar:
                optimizer_adam.zero_grad()
                points = self._sample_all()
                loss, loss_components = self.calculate_loss(points)
                loss.backward()
                optimizer_adam.step()

                if use_scheduler:
                    scheduler.step()

                # 日志记录
                log_freq = self.train_config.get('log_freq', 10)
                if epoch % log_freq == 0:
                    elapsed = time.time() - start_time_total
                    param_errors = self.physics.get_param_errors()
                    self.logger.log_metrics(epoch, {'loss': loss.item(), 'components': loss_components, 'params': param_errors}, time_sec=elapsed)

                    # 进度条动态显示误差
                    postfix = {'Loss': f"{loss.item():.4f}"}
                    for k, v in param_errors.items():
                        if "_error" in k:
                            short_k = k.replace("_error", "")
                            postfix[short_k] = f"{v:.1f}%"
                    pbar.set_postfix(postfix)

            # --- 第二阶段: LBFGS (精调) ---
            epochs_lbfgs = self.train_config['epochs_lbfgs']
            lbfgs_lr = self.train_config.get('lbfgs_lr', 1.0) # 默认为 1.0 (如果不指定)

            # [可选] LBFGS 阶段冻结物理参数
            # freeze_physics_in_lbfgs: true freezes physics parameters during LBFGS.
            freeze_physics = self.train_config.get('freeze_physics_in_lbfgs', False)
            physics_params_frozen = []
            if freeze_physics:
                for p in self.physics.parameters():
                    if p.requires_grad:
                        p.requires_grad_(False)
                        physics_params_frozen.append(p)
                lbfgs_params = [p for p in self.model.parameters() if p.requires_grad]
            else:
                lbfgs_params = self.params_to_optimize

            optimizer_lbfgs = optim.LBFGS(lbfgs_params, lr=lbfgs_lr, max_iter=20, history_size=100)

            # [优化] LBFGS 学习率余弦衰减: 从 lbfgs_lr 衰减到 lbfgs_lr/100
            lbfgs_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer_lbfgs, T_max=epochs_lbfgs, eta_min=lbfgs_lr * 0.01
            )

            pbar_lbfgs = tqdm(range(epochs_lbfgs), desc="LBFGS", ascii=True)
            prev_loss_val = float('inf')
            best_loss = float('inf')
            patience_counter = 0
            patience_limit = 10  # 连续 10 次无改善则停止

            # 【重要】LBFGS 稳定性修复:
            # 整个 LBFGS 阶段只采样一次数据 (Fixed Batch).
            # LBFGS 依赖 Hessian 近似历史，如果每步都变数据，历史就会失效导致发散。
            points_lbfgs = self._sample_all()

            for epoch in pbar_lbfgs:
                def closure():
                    optimizer_lbfgs.zero_grad()
                    # 使用固定的 'points_lbfgs'
                    loss, _ = self.calculate_loss(points_lbfgs)

                    if not torch.isfinite(loss):
                        print(f" [警告] LBFGS Loss is NaN/Inf! Stopping step.")
                        return loss

                    loss.backward()
                    return loss

                loss = optimizer_lbfgs.step(closure)
                current_loss_val = loss.item()

                # 日志记录 (使用重新采样的数据看泛化性，或者使用固定数据看优化性？通常用新数据看泛化)
                # 为了曲线一致性，这里记录重新采样的 Loss
                points = self._sample_all()
                _, loss_components = self.calculate_loss(points)
                param_errors = self.physics.get_param_errors()

                elapsed = time.time() - start_time_total
                self.logger.log_metrics(epochs_adam + epoch, {'loss': current_loss_val, 'components': loss_components, 'params': param_errors}, time_sec=elapsed)

                # 进度条更新
                postfix = {'Loss': f"{current_loss_val:.4f}"}
                for k, v in param_errors.items():
                    if "_error" in k:
                        short_k = k.replace("_error", "")
                        postfix[short_k] = f"{v:.1f}%"
                pbar_lbfgs.set_postfix(postfix)

                # 【优化】增强早停机制 (Enhanced Early Stopping)
                # 1. 相对改善阈值 (relative improvement threshold)
                relative_improvement = (prev_loss_val - current_loss_val) / (prev_loss_val + 1e-10)

                # 2. 是否有足够改善
                if current_loss_val < best_loss * 0.9999:  # 至少改善 0.01%
                    best_loss = current_loss_val
                    patience_counter = 0
                else:
                    patience_counter += 1

                # 3. 满足停止条件
                if patience_counter >= patience_limit:
                    print(f"\n[INFO] LBFGS 提前停止于轮次 {epoch} (连续 {patience_limit} 次无改善)")
                    break

                # 4. 绝对收敛条件 (保留原有逻辑)
                if abs(prev_loss_val - current_loss_val) < 1e-8:
                    print(f"\n[INFO] LBFGS 提前停止于轮次 {epoch} (损失变化 < 1e-8)")
                    break

                prev_loss_val = current_loss_val

                # 学习率衰减
                lbfgs_scheduler.step()

            # [关键修复] 恢复物理参数的梯度 (评估和保存时需要)
            for p in physics_params_frozen:
                p.requires_grad_(True)

        except KeyboardInterrupt:
            print("\n[WARN] 用户中断训练 (KeyboardInterrupt)")
        except Exception as e:
            print(f"\n[ERROR] 训练过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 无论是否出错，都执行评估和保存 (原子性保护)
            self._evaluate_model()
            self._save_results()  # 评估后再保存，确保 L2 误差被写入

    def _save_results(self):
        """保存模型和历史记录 (封装为原子操作)"""
        print(f"\n[INFO] 正在保存训练结果 (原子性操作)...")
        self.logger.save_history()
        self.logger.save_model(self.model)
        self.logger.save_model(self.physics, filename="physics_best.pth")
        print(f"[INFO] 训练完成！结果保存在: {self.logger.get_save_dir()}")

    def _evaluate_model(self):
        """模型评估 (神经元质量 + 最终参数报告)"""
        # 1. 神经元评估
        print("\n[INFO] 正在进行神经元级质量评估...")
        try:
            from ..utils.neuro_evaluator import NeuroQualityEvaluator
            evaluator = NeuroQualityEvaluator(self.model, self.device)
            evaluator.evaluate_layers()
        except Exception as e:
            print(f"[WARN] 神经元评估失败: {e}")

        # 2. 最终参数报告
        print("\n" + "="*50)
        print(" [最终参数收敛报告 Final Parameter Report]")
        print("="*50)

        try:
            final_metrics = self.physics.get_param_errors()
            true_params = self.physics.true_params

            print(f"{'参数 (Param)':<15} | {'学习值 (Learned)':<15} | {'真实值 (True)':<15} | {'误差 (Error)':<15}")
            print("-" * 66)

            # 提取参数名
            param_names = set()
            for k in final_metrics.keys():
                if k.endswith('_error'):
                    param_names.add(k.replace('_error', ''))
                else:
                    param_names.add(k)

            for name in sorted(list(param_names)):
                val = final_metrics.get(name, "N/A")
                err = final_metrics.get(f"{name}_error", "N/A")
                true_val = true_params.get(name, "Unknown")

                val_str = f"{val:.4f}" if isinstance(val, (float, int)) else str(val)
                true_str = f"{true_val:.4f}" if isinstance(true_val, (float, int)) else str(true_val)
                err_str = f"{err:.2f}%" if isinstance(err, (float, int)) else str(err)

                print(f"{name:<15} | {val_str:<15} | {true_str:<15} | {err_str:<15}")
            print("-" * 66)

            # 3. L2 误差计算
            self._compute_l2_error()

        except Exception as e:
            print(f"[ERROR] 生成参数报告失败: {e}")

        print("="*50 + "\n")

        # 4. 绘图
        try:
            print("[INFO] 正在生成图表...")
            self.plotter.plot_loss_history(self.logger.history)
            self.plotter.plot_param_history(self.logger.history, self.physics.true_params)
            self.plotter.plot_solution(self.model, self.physics, self.device)
        except Exception as e:
            print(f"[WARN] 绘图失败: {e}")

    def _compute_l2_error(self):
        """计算模型相对 L2 误差并保存 (使用均匀网格)"""
        final_l2_error = None
        try:
            domain = self.config['physics']['domain']
            n_per_dim = 50  # 每个维度 50 个点

            # 通过模型输入维度判断 1D (input=2: x,t) 还是 2D (input=3: x,y,t)
            input_dim = self.model.linears[0].in_features

            if input_dim == 3:
                # 2D 问题: 输入 (x, y, t)
                d_min = domain[0] if isinstance(domain[0], (int, float)) else domain[0][0]
                d_max = domain[1] if isinstance(domain[1], (int, float)) else domain[0][1]
                x_vals = torch.linspace(d_min, d_max, n_per_dim, device=self.device)
                y_vals = torch.linspace(d_min, d_max, n_per_dim, device=self.device)
                t_vals = torch.linspace(0, 1, n_per_dim, device=self.device)
                # 均匀网格 (n_per_dim^3 个点太多, 用二维网格 + 随机时间)
                xx, yy = torch.meshgrid(x_vals, y_vals, indexing='ij')
                x_flat = xx.reshape(-1, 1).repeat(n_per_dim, 1)
                y_flat = yy.reshape(-1, 1).repeat(n_per_dim, 1)
                t_flat = t_vals.repeat_interleave(n_per_dim * n_per_dim).unsqueeze(1)
                inputs = (x_flat, y_flat, t_flat)
            else:
                # 1D 问题: 输入 (x, t)
                x_vals = torch.linspace(domain[0], domain[1], n_per_dim, device=self.device)
                t_vals = torch.linspace(0, 1, n_per_dim, device=self.device)
                xx, tt = torch.meshgrid(x_vals, t_vals, indexing='ij')
                x_flat = xx.reshape(-1, 1)
                t_flat = tt.reshape(-1, 1)
                inputs = (x_flat, t_flat)

            self.model.eval()
            with torch.no_grad():
                u_pred_val = self.model(*inputs)
                u_true_val = self.physics.analytical_solution(*inputs)

            if u_true_val is not None:
                error_l2 = torch.norm(u_pred_val - u_true_val, 2) / (torch.norm(u_true_val, 2) + 1e-6)
                final_l2_error = error_l2.item()
                print(f"{'Model L2 Error':<15} | {'':>15} | {'':>15} | {final_l2_error*100:.4f}%")
            else:
                print(f"{'Model L2 Error':<15} | {'':>15} | {'':>15} | {'N/A'}")

        except Exception as e:
            print(f"[WARN] L2 Error 计算失败: {e}")
            import traceback
            traceback.print_exc()

        # 保存到 history (无论 L2 是否成功)
        try:
            if self.logger.history:
                if final_l2_error is not None:
                    self.logger.history['l2_error'] = final_l2_error
                self.logger.history['final_params'] = self.physics.get_param_errors()
        except Exception as e:
            print(f"[WARN] 保存参数失败: {e}")
