# TGSR-PINN 中文说明

这是 TGSR-PINN 的最小公开演示仓库，只保留一种方法和一个实验，方便别人从干净仓库直接试跑。

本版本只包含：

- 方法：TGSR-PINN
- 实验：High-Peclet 二维 diffusion 源任务到 advection-diffusion 目标任务迁移
- 推荐演示种子：77

本仓库不包含对比方法、不包含 checkpoint、不包含生成结果、不包含论文草稿、不包含 IDE 配置，也不包含任何本地私有路径。

## 主要文件

- `main.py`：单任务 PINN 训练入口。
- `configs/task2d_diffusion_source_noise005.json`：source diffusion 任务配置。
- `configs/task2d_high_peclet.json`：High-Peclet target 任务配置。
- `experiments/runners/run_tgsr_high_peclet.py`：TGSR source-to-target 端到端运行脚本。
- `run_transfer_experiment.bat`：Windows 下一键运行 seed 77 demo。
- `src/`：模型、物理方程、训练器、TGSR 迁移策略和工具函数。
- `scripts/smoke_test_repository.py`：上传前快速检查脚本。
- `tests/`：公开版本的轻量测试。

## 快速检查

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/smoke_test_repository.py
python -m pytest -q
```

## 运行公开实验

最简单的运行方式：

```powershell
.\run_transfer_experiment.bat
```

它会依次完成：

1. 使用 seed 77 训练 source diffusion PINN。
2. 使用 TGSR-PINN 从 source model 迁移到 High-Peclet target PINN。

等价 Python 命令：

```powershell
python experiments/runners/run_tgsr_high_peclet.py --seed 77 --experiment-name tgsr_high_peclet_seed77
```

新结果会写入 `results/`，该目录已被 Git 忽略，不会上传到 GitHub。

## 数据说明

该实验使用 `src/physics/` 中实现的制造解和采样器，不需要外部数据集。

完整训练耗时取决于硬件，CPU 上会比较慢。`smoke_test_repository.py` 和单元测试只做轻量检查，不会跑完整实验。
