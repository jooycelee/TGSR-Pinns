@echo off
setlocal
cd /d "%~dp0"

set "PINN_SEED=77"
set "TGSR_WARMUP_CAPTURE_FRAMES=8"

python experiments\runners\run_tgsr_high_peclet.py --seed 77 --experiment-name tgsr_high_peclet_seed77

endlocal
