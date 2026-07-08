import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from .mpl_config import configure_matplotlib

def _use_available_style():
    """Use a seaborn-like style across old/new Matplotlib versions."""
    for style_name in ('seaborn-v0_8-darkgrid', 'seaborn-darkgrid', 'ggplot', 'default'):
        if style_name == 'default' or style_name in plt.style.available:
            plt.style.use(style_name)
            return style_name
    return 'default'


class Plotter:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        # Ensure pretty plotting and Chinese Support
        _use_available_style()

        configure_matplotlib(plt)

    def plot_loss_history(self, history):
        """Plot total loss and loss components"""
        epochs = history['epoch']
        loss = history['loss']

        plt.figure(figsize=(12, 5))

        # 1. Total Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, loss, label='总损失 (Total Loss)', color='black', linewidth=2)
        plt.yscale('log')
        plt.xlabel('训练轮次 (Epoch)')
        plt.ylabel('损失 (Log Scale)')
        plt.title('训练损失收敛曲线')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5)

        # 2. Loss Components
        plt.subplot(1, 2, 2)
        for name, values in history['loss_components'].items():
            if len(values) == len(epochs):
                plt.plot(epochs, values, label=f'{name.upper()} 损失', alpha=0.8)

        plt.yscale('log')
        plt.xlabel('训练轮次 (Epoch)')
        plt.title('损失分量分解')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'loss_history.png'), dpi=150)
        plt.close()

    def plot_param_history(self, history, true_params):
        """Plot parameter evolution (Values only, error is annotated)"""
        epochs = history['epoch']
        params_hist = history['params']

        if not params_hist:
            return

        # Filter out keys that are explicit errors (e.g. 'alpha_error')
        # We only want to plot the VALUE evolution (e.g. 'alpha')
        value_keys = [k for k in params_hist.keys() if not k.endswith('_error')]

        if not value_keys:
            # Fallback if only errors are logged (should not happen with new update)
            value_keys = list(params_hist.keys())

        num_params = len(value_keys)
        # Dynamic Width: 6 per param
        plt.figure(figsize=(6 * num_params, 5))

        for idx, param_name in enumerate(value_keys):
            values = params_hist[param_name]
            plt.subplot(1, num_params, idx + 1)

            # Plot Learned Value
            plt.plot(epochs, values, label='学习值 (Learned)', linewidth=2, color='blue')

            # Plot True Value (if exists)
            true_val = true_params.get(param_name)
            if true_val is not None:
                plt.axhline(y=true_val, color='red', linestyle='--', label=f'真实值 (True): {true_val}')

                # Calculate final error based on last value
                # (We could also use the explicit error key from history if we wanted, but calc is fine)
                try:
                    current_val = values[-1]
                    final_err = abs(current_val - true_val) / abs(true_val) * 100
                    plt.title(f'{param_name} (最终误差: {final_err:.2f}%)')
                except:
                    plt.title(f'{param_name}')
            else:
                plt.title(f'{param_name} 演变过程')

            plt.xlabel('训练轮次 (Epoch)')
            plt.ylabel('参数值')
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'param_history.png'), dpi=150)
        plt.close()

        plt.close()

    def plot_solution(self, model, physics, device):
        """Dispatch to 1D or 2D plotter based on physics"""
        if hasattr(physics, 'y_max'):
             self.plot_solution_2d(model, physics, device)
        else:
             self.plot_solution_1d(model, physics, device)

    def plot_solution_1d(self, model, physics, device):
        """Plot predicted solution heatmap, error map, and snapshots (1D)"""
        # Get Domain from Physics (stored in V2)
        if hasattr(physics, 'domain'):
            x_min, x_max = physics.domain
        else:
            # Fallback
            x_min, x_max = -1, 1

        # Create Grid
        x = np.linspace(x_min, x_max, 200)
        t = np.linspace(0, 1, 100)
        X, T = np.meshgrid(x, t)

        X_flat = torch.tensor(X.flatten()[:, None], dtype=torch.float32).to(device)
        T_flat = torch.tensor(T.flatten()[:, None], dtype=torch.float32).to(device)

        # Predict
        model.eval()
        with torch.no_grad():
            u_pred = model(X_flat, T_flat).cpu().numpy().reshape(X.shape)

            # Analytical Solution?
            u_true = None
            try:
                # Need to use analytical_solution from physics
                # Physics.analytical_solution accepts tensor inputs
                u_true_tensor = physics.analytical_solution(X_flat, T_flat)
                if u_true_tensor is not None:
                    u_true = u_true_tensor.cpu().numpy().reshape(X.shape)
            except Exception as e:
                print(f"[WARN] Failed to compute analytical solution for plotting: {e}")

        # --- Plot 1: Heatmaps ---
        if u_true is not None:
            fig, axs = plt.subplots(1, 3, figsize=(18, 5))

            # Pred
            im0 = axs[0].contourf(T, X, u_pred, levels=100, cmap='jet')
            axs[0].set_title('预测解 Predicted u(x,t)')
            axs[0].set_xlabel('时间 t')
            axs[0].set_ylabel('空间 x')
            plt.colorbar(im0, ax=axs[0])

            # True
            im1 = axs[1].contourf(T, X, u_true, levels=100, cmap='jet')
            axs[1].set_title('精确解 Exact u(x,t)')
            axs[1].set_xlabel('时间 t')
            plt.colorbar(im1, ax=axs[1])

            # Error (Absolute)
            error = np.abs(u_pred - u_true)

            im2 = axs[2].contourf(T, X, error, levels=100, cmap='jet')
            axs[2].set_title('绝对误差 Absolute Error')
            axs[2].set_xlabel('时间 t')
            plt.colorbar(im2, ax=axs[2])

            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, 'solution_heatmap.png'), dpi=150)
            plt.close()

        else:
            # Just Pred
            plt.figure(figsize=(8, 6))
            plt.contourf(T, X, u_pred, levels=100, cmap='jet')
            plt.title('预测解 Predicted u(x,t)')
            plt.xlabel('时间 t')
            plt.ylabel('空间 x')
            plt.colorbar()
            plt.savefig(os.path.join(self.save_dir, 'solution_heatmap.png'), dpi=150)
            plt.close()

        # --- Plot 2: Snapshots (t=0.25, 0.5, 0.75) ---
        plt.figure(figsize=(10, 6))

        snapshots = [0.25, 0.50, 0.75]
        t_indices = [int(s * 100) for s in snapshots] # roughly index since t is 100 points

        colors = ['b', 'g', 'r']

        for idx, t_idx in enumerate(t_indices):
            if t_idx >= len(t): t_idx = len(t) - 1

            t_val = t[t_idx] # t is 1D linspace

            # Plot Pred
            plt.plot(x, u_pred[t_idx, :], color=colors[idx], linestyle='--', linewidth=2, label=f'预测 t={t_val:.2f}')

            # Plot True
            if u_true is not None:
                plt.plot(x, u_true[t_idx, :], color=colors[idx], linestyle='-', linewidth=1, alpha=0.6, label=f'真实 t={t_val:.2f}')

        plt.xlabel('空间 x')
        plt.ylabel('u(x,t)')
        plt.title('不同时刻解的快照')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, 'solution_snapshots.png'), dpi=150)
        plt.close()

    def plot_solution_2d(self, model, physics, device):
        """Plot solution u(x,y) at final time t=1.0"""
        x_min, x_max = physics.x_min, physics.x_max
        y_min, y_max = physics.y_min, physics.y_max

        # Grid
        x = np.linspace(x_min, x_max, 100)
        y = np.linspace(y_min, y_max, 100)
        X, Y = np.meshgrid(x, y)

        # At Time t=1.0 (Final state)
        t_val = 1.0
        T = np.ones_like(X) * t_val

        # Flatten
        X_flat = torch.tensor(X.flatten()[:, None], dtype=torch.float32).to(device)
        Y_flat = torch.tensor(Y.flatten()[:, None], dtype=torch.float32).to(device)
        T_flat = torch.tensor(T.flatten()[:, None], dtype=torch.float32).to(device)

        model.eval()
        with torch.no_grad():
            u_pred = model(X_flat, Y_flat, T_flat).cpu().numpy().reshape(X.shape)

            u_true = None
            try:
                u_true_tensor = physics.analytical_solution(X_flat, Y_flat, T_flat)
                if u_true_tensor is not None:
                    u_true = u_true_tensor.cpu().numpy().reshape(X.shape)
            except Exception as e:
                print(f"[WARN] No analytical solution (2D): {e}")

        # Plot 3-Panel
        if u_true is not None:
            fig, axs = plt.subplots(1, 3, figsize=(18, 5))

            im0 = axs[0].contourf(X, Y, u_pred, levels=100, cmap='jet')
            axs[0].set_title(f'预测解 Predicted u(x,y) at t={t_val}')
            axs[0].set_xlabel('x')
            axs[0].set_ylabel('y')
            plt.colorbar(im0, ax=axs[0])

            im1 = axs[1].contourf(X, Y, u_true, levels=100, cmap='jet')
            axs[1].set_title(f'精确解 Exact u(x,y) at t={t_val}')
            axs[1].set_xlabel('x')
            plt.colorbar(im1, ax=axs[1])

            # Error (Absolute)
            error = np.abs(u_pred - u_true)

            im2 = axs[2].contourf(X, Y, error, levels=100, cmap='jet')
            axs[2].set_title('绝对误差 Absolute Error')
            axs[2].set_xlabel('x')
            plt.colorbar(im2, ax=axs[2])

            plt.savefig(os.path.join(self.save_dir, 'solution_2d_heatmap.png'), dpi=150)
            plt.close()
        else:
             plt.figure(figsize=(6,5))
             plt.contourf(X, Y, u_pred, levels=100, cmap='jet')
             plt.title(f'预测解 Predicted u(x,y) at t={t_val}')
             plt.colorbar()
             plt.savefig(os.path.join(self.save_dir, 'solution_2d_heatmap.png'), dpi=150)
             plt.close()

    @staticmethod
    def compare_loss_histories(history1, history2, name1, name2, title, save_path):
        """Standardized comparison plot for two runs"""
        # Ensure pretty plotting
        _use_available_style()
        configure_matplotlib(plt)

        epochs1 = history1['epoch']
        loss1 = history1['loss']

        epochs2 = history2['epoch']
        loss2 = history2['loss']

        plt.figure(figsize=(10, 6))

        plt.plot(epochs1, loss1, label=f'{name1}', color='red', linestyle='--', alpha=0.7, linewidth=2)
        plt.plot(epochs2, loss2, label=f'{name2}', color='blue', linestyle='-', linewidth=2)

        plt.yscale('log')
        plt.xlabel('训练轮次 (Epoch)')
        plt.ylabel('损失 Result (Log Scale)')
        plt.title(title)
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[INFO] 对比图已生成: {save_path}")

    @staticmethod
    def compare_multiple_loss_histories(history_list, title, save_path, x_axis='epoch'):
        """
        Plot multiple loss curves.
        history_list: List of (history_dict, label_str)
        x_axis: 'epoch' or 'time'
        """
        _use_available_style()
        configure_matplotlib(plt)

        plt.figure(figsize=(10, 6))

        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
        linestyles = ['--', '-', '-.', ':', '--', '-']

        for i, (hist, label) in enumerate(history_list):
            if x_axis == 'time':
                x_data = hist.get('time', [])
                if not x_data:
                    print(f"[WARN] History for {label} missing 'time' data. Fallback to epoch/index.")
                    x_data = range(len(hist['loss']))
                xlabel = '训练时间 (Seconds)'
            else:
                x_data = hist['epoch']
                xlabel = '训练轮次 (Epoch)'

            loss = hist['loss']

            # Ensure shape match (sometimes time might have one less/more usage depending on implementation)
            min_len = min(len(x_data), len(loss))
            x_data = x_data[:min_len]
            loss = loss[:min_len]

            color = colors[i % len(colors)]
            ls = linestyles[i % len(linestyles)]

            plt.plot(x_data, loss, label=label, color=color, linestyle=ls, linewidth=2, alpha=0.8)

        plt.yscale('log')
        plt.xlabel(xlabel)
        plt.ylabel('损失 Result (Log Scale)')
        plt.title(title)
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[INFO] 多重对比图已生成 ({x_axis}): {save_path}")
