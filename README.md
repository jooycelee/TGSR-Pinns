# TGSR-PINN

This is a minimal public demo repository for **Target-Guided Selective Reweighting PINN (TGSR-PINN)**.

The release intentionally exposes only one method and one experiment:

- Method: TGSR-PINN
- Experiment: High-Peclet 2D diffusion source to advection-diffusion target transfer
- Demo seed: 77

No alternate method implementations, checkpoint files, generated results, paper drafts, IDE files, or local workspace paths are included.

## Repository Layout

- `main.py`: single-task PINN training entry point.
- `configs/task2d_diffusion_source_noise005.json`: source diffusion task.
- `configs/task2d_high_peclet.json`: High-Peclet target task.
- `experiments/runners/run_tgsr_high_peclet.py`: end-to-end TGSR source-to-target runner.
- `run_transfer_experiment.bat`: one-command Windows demo for seed 77.
- `src/`: model, physics, trainer, TGSR transfer strategy, and utilities.
- `scripts/smoke_test_repository.py`: quick release sanity check.
- `tests/`: lightweight tests for the public TGSR release.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/smoke_test_repository.py
python -m pytest -q
```

## Run The Public Experiment

The simplest path is:

```powershell
.\run_transfer_experiment.bat
```

This runs:

1. The source diffusion PINN with seed 77.
2. The High-Peclet target PINN initialized with TGSR-PINN transfer from the source model.

Equivalent Python command:

```powershell
python experiments/runners/run_tgsr_high_peclet.py --seed 77 --experiment-name tgsr_high_peclet_seed77
```

Generated outputs are written under `results/`, which is ignored by Git.

## Notes

The experiment uses manufactured solutions and samplers implemented in `src/physics/`; no external dataset is required.

Complete training can take time, especially on CPU. The smoke test and unit tests are lightweight checks only; they do not run the complete experiment.
