"""Fast smoke test for the minimal TGSR-PINN public release.

Run from the repository root:
    python scripts/smoke_test_repository.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _assert_close(name: str, actual: float, expected: float, tol: float = 1e-12) -> None:
    if abs(float(actual) - float(expected)) > tol:
        raise AssertionError(f"{name}: expected {expected}, got {actual}")


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def check_public_layout() -> None:
    required_files = {
        "configs/task2d_diffusion_source_noise005.json",
        "configs/task2d_high_peclet.json",
        "experiments/runners/run_tgsr_high_peclet.py",
        "run_transfer_experiment.bat",
        "main.py",
        "README.md",
        "README_CN.md",
        "REPRODUCE.md",
    }
    missing = sorted(path for path in required_files if not (REPO_ROOT / path).exists())
    if missing:
        raise AssertionError(f"Missing required public files: {missing}")

    runners_dir = REPO_ROOT / "experiments" / "runners"
    allowed_runners = {"__init__.py", "run_tgsr_high_peclet.py"}
    current_runners = {path.name for path in runners_dir.glob("*.py")}
    unexpected = sorted(current_runners.difference(allowed_runners))
    if unexpected:
        raise AssertionError(f"Unexpected public runner files: {unexpected}")

    root_python_runners = sorted(path.name for path in REPO_ROOT.glob("run_*.py"))
    if root_python_runners:
        raise AssertionError(f"Root-level Python runners are not part of this release: {root_python_runners}")


def check_high_peclet_configs() -> None:
    source = _load_json(REPO_ROOT / "configs/task2d_diffusion_source_noise005.json")
    target = _load_json(REPO_ROOT / "configs/task2d_high_peclet.json")

    if source["physics"]["type"] != "heat_2d_inverse":
        raise AssertionError(f"Unexpected source physics: {source['physics']['type']}")
    _assert_close("source noise", source["physics"]["noise_level"], 0.005)
    _assert_close("source true alpha", source["physics"]["true_params"]["alpha"], 0.05)

    physics = target["physics"]
    training = target["training"]
    if physics["type"] != "heat_2d_convection":
        raise AssertionError(f"Unexpected target physics: {physics['type']}")
    _assert_close("target alpha init", physics["params"]["alpha"], 0.005)
    _assert_close("target vx init", physics["params"]["vx"], 1.0)
    _assert_close("target vy init", physics["params"]["vy"], 0.5)
    _assert_close("target alpha true", physics["true_params"]["alpha"], 0.001)
    _assert_close("target vx true", physics["true_params"]["vx"], 2.0)
    _assert_close("target vy true", physics["true_params"]["vy"], 1.0)
    _assert_close("target noise", physics["noise_level"], 0.005)

    expected_weights = {"pde": 10.0, "ic": 10.0, "bc": 10.0, "data": 50.0}
    if training["weights"] != expected_weights:
        raise AssertionError(f"High-Peclet target weights mismatch: {training['weights']}")


def check_registries_and_strategy() -> None:
    from src.physics import PHYSICS_REGISTRY
    from src.transfer import STRATEGY_REGISTRY, get_transfer_strategy
    from src.transfer.tgsr_strategy import TGSRTransferStrategy

    if set(STRATEGY_REGISTRY) != {"tgsr"}:
        raise AssertionError(f"Public release should expose only TGSR: {sorted(STRATEGY_REGISTRY)}")

    expected_physics = {"heat_2d_inverse", "heat_2d_convection"}
    if set(PHYSICS_REGISTRY) != expected_physics:
        raise AssertionError(f"Unexpected public physics registry: {sorted(PHYSICS_REGISTRY)}")

    strategy = get_transfer_strategy("tgsr", torch.device("cpu"))
    if not isinstance(strategy, TGSRTransferStrategy):
        raise AssertionError("get_transfer_strategy('tgsr') returned the wrong type.")


def check_model_forward() -> None:
    from src.models import FCN

    torch.manual_seed(123)
    model = FCN([3, 8, 8, 1]).cpu()
    x = torch.linspace(0.0, 1.0, 5).view(-1, 1)
    y = torch.linspace(1.0, 0.0, 5).view(-1, 1)
    t = torch.zeros_like(x)
    output = model(x, y, t)

    if tuple(output.shape) != (5, 1):
        raise AssertionError(f"Unexpected FCN output shape: {tuple(output.shape)}")
    if not torch.isfinite(output).all():
        raise AssertionError("FCN forward produced non-finite values.")


def check_tgsr_soft_decay_mapping() -> None:
    from src.transfer.tgsr_strategy import TGSRTransferStrategy

    strategy = TGSRTransferStrategy(torch.device("cpu"))
    _, scales = strategy._build_gmm_soft_reset_plan(np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
    expected = np.array([1.0, 0.98125, 0.85, 0.79375, 0.4])
    if not np.allclose(scales, expected):
        raise AssertionError(f"Unexpected TGSR soft-decay scales: {scales}")


def main() -> int:
    checks = [
        ("public repository layout", check_public_layout),
        ("High-Peclet configs", check_high_peclet_configs),
        ("TGSR strategy registry", check_registries_and_strategy),
        ("tiny FCN forward pass", check_model_forward),
        ("TGSR soft-decay mapping", check_tgsr_soft_decay_mapping),
    ]

    print("[TGSR smoke test] repository root:", REPO_ROOT)
    for name, check in checks:
        check()
        print(f"[OK] {name}")

    print("[PASS] Minimal TGSR-PINN release smoke test completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
