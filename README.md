# TGSR-PINN

This repository provides a minimal reproducible implementation of
**Target-Guided Selective Reweighting PINN (TGSR-PINN)** for a High-Peclet
inverse transfer experiment.

## Experiment

The experiment transfers a pretrained two-dimensional diffusion PINN to a
strongly advection-dominated advection-diffusion inverse problem.

- Source task: infer the diffusion coefficient from observations with 0.5%
  noise.
- Target task: infer the diffusion coefficient `alpha` and velocities `vx`
  and `vy`.
- Target true parameters: `alpha=0.001`, `vx=2.0`, `vy=1.0`.
- Network: six hidden layers with 100 neurons per layer and `tanh` activation.
- Reproduction seed: `77`.

The reference seed-77 result is approximately:

- Relative L2 field error: `0.06867%`
- Alpha relative error: `0.38058%`

## Run

Python 3.10 or newer is recommended. Install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the complete source-to-target experiment on Windows:

```powershell
.\run_transfer_experiment.bat
```

Equivalent Python command:

```powershell
python experiments/runners/run_tgsr_high_peclet.py --seed 77 --experiment-name tgsr_high_peclet_seed77
```

The runner trains the source PINN, performs TGSR-PINN transfer, trains the
target inverse problem, and checks the final seed-77 result. Generated models,
histories, and figures are saved under `results/`.

## Check

```powershell
python scripts/smoke_test_repository.py
python -m pytest -q
```
