# TGSR-PINN

本仓库提供 TGSR-PINN（Target-Guided Selective Reweighting PINN）的最小可复现实现，用于运行 High-Peclet 反问题迁移实验。

## 实验内容

实验将预训练的二维扩散 PINN 迁移到强对流占优的二维对流扩散反问题。

- 源任务：在 0.5% 观测噪声下反演扩散系数。
- 目标任务：同时反演扩散系数 `alpha` 和速度参数 `vx`、`vy`。
- 目标参数真值：`alpha=0.001`、`vx=2.0`、`vy=1.0`。
- 网络结构：6 个隐藏层，每层 100 个神经元，激活函数为 `tanh`。
- 复现种子：`77`。

seed 77 的参考结果约为：

- 物理场相对 L2 误差：`0.06867%`
- Alpha 相对误差：`0.38058%`

## 运行方法

推荐使用 Python 3.10 或更高版本。首先安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

在 Windows 下运行完整的源任务训练和目标任务迁移实验：

```powershell
.\run_transfer_experiment.bat
```

等价的 Python 命令：

```powershell
python experiments/runners/run_tgsr_high_peclet.py --seed 77 --experiment-name tgsr_high_peclet_seed77
```

脚本会依次训练源任务 PINN、执行 TGSR-PINN 迁移、训练目标反问题并检查最终结果。生成的模型、训练记录和图片保存在 `results/` 目录中。

## 快速检查

```powershell
python scripts/smoke_test_repository.py
python -m pytest -q
```
