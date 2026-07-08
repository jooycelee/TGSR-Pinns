# Reproduction Notes

This simplified release is designed for one reproducible public demo: TGSR-PINN on the High-Peclet source-to-target transfer experiment.

## Environment

Recommended:

- Python 3.10+
- PyTorch with CUDA for complete training

Install:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Sanity Checks

```powershell
python scripts/smoke_test_repository.py
python -m compileall src main.py experiments/runners scripts
python -m pytest -q
```

## Public Configurations

- `configs/task2d_diffusion_source_noise005.json`: source 2D diffusion inverse task with 0.5% observation noise.
- `configs/task2d_high_peclet.json`: target High-Peclet 2D advection-diffusion inverse task.

## Main Demo Command

```powershell
.\run_transfer_experiment.bat
```

Equivalent Python command:

```powershell
python experiments/runners/run_tgsr_high_peclet.py --seed 77 --experiment-name tgsr_high_peclet_seed77
```

The runner first trains the source model, then runs the target task with:

```text
transfer_mode=tgsr
seed=77
pruning_mode=gmm
```

If the matching source model already exists under `results/tgsr_high_peclet_seed77/`, the runner reuses it.

## Outputs

New runs write to `results/`. The directory is ignored by Git so generated artifacts do not enter the code repository.
