import argparse
import glob
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MAIN_PY = ROOT / "main.py"
SOURCE_TASK = "task2d_diffusion_source_noise005"
TARGET_TASK = "task2d_high_peclet"
BEST_SEED = 77
BEST_WARMUP_CAPTURE_FRAMES = 8
EXPECTED_ALPHA_ERROR_PERCENT = 0.38057807832956103
EXPECTED_L2_ERROR = 0.0006866998737677932


def repo_path(*parts: str) -> str:
    return str(ROOT.joinpath(*parts))


def find_latest_model(results_dir: str | None) -> str | None:
    if not results_dir or not os.path.exists(results_dir):
        return None
    models = glob.glob(os.path.join(results_dir, "**", "model_best.pth"), recursive=True)
    if not models:
        return None
    return max(models, key=os.path.getmtime)


def find_existing_source_model(exp_name: str, seed: int) -> tuple[str | None, str | None]:
    search_root = repo_path("results", exp_name)
    if not os.path.exists(search_root):
        return None, None

    candidates = [path for path in glob.glob(os.path.join(search_root, "*")) if os.path.isdir(path)]
    candidates.sort(key=os.path.getmtime, reverse=True)
    for run_dir in candidates:
        model_path = os.path.join(run_dir, "model_best.pth")
        seed_file = os.path.join(run_dir, f"seed_{seed}.txt")
        if os.path.exists(model_path) and os.path.exists(seed_file):
            return model_path, run_dir
    return None, None


def run_main(
    config_path: str,
    exp_name: str,
    seed: int,
    transfer_mode: str = "none",
    load_model: str | None = None,
    pruning_mode: str = "gmm",
) -> str | None:
    cmd = [
        sys.executable,
        str(MAIN_PY),
        "--config",
        config_path,
        "--transfer_mode",
        transfer_mode,
        "--seed",
        str(seed),
        "--experiment_name",
        exp_name,
    ]
    if load_model:
        cmd.extend(["--load_model", load_model])
    if transfer_mode == "tgsr":
        cmd.extend(["--pruning_mode", pruning_mode])

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] RUNNING: {' '.join(cmd)}")
    env = os.environ.copy()
    env["PINN_SEED"] = str(seed)
    if transfer_mode == "tgsr":
        # These diagnostic captures are part of the historical best-run protocol.
        # Each capture samples target points, so omitting them changes the RNG path.
        env["TGSR_WARMUP_CAPTURE_FRAMES"] = str(BEST_WARMUP_CAPTURE_FRAMES)

    known_dirs = {
        path for path in glob.glob(repo_path("results", exp_name, "*"))
        if os.path.isdir(path)
    }
    process = subprocess.Popen(cmd, cwd=str(ROOT), env=env)
    process.wait()

    current_dirs = [
        path for path in glob.glob(repo_path("results", exp_name, "*"))
        if os.path.isdir(path)
    ]
    new_dirs = [path for path in current_dirs if path not in known_dirs]
    out_dir = max(new_dirs, key=os.path.getmtime) if new_dirs else None

    if process.returncode != 0:
        print(f"\n[ERROR] Command failed with exit code {process.returncode}")
        return out_dir
    return out_dir


def verify_best_seed77_result(result_dir: str) -> None:
    history_path = Path(result_dir) / "history.json"
    if not history_path.exists():
        raise RuntimeError(f"Missing target history: {history_path}")

    with history_path.open("r", encoding="utf-8-sig") as handle:
        history = json.load(handle)

    transfer = history.get("tgsr_transfer", {})
    snapshots = transfer.get("warmup_neuron_snapshots", [])
    alpha_error = float(history["final_params"]["alpha_error"])
    l2_error = float(history["l2_error"])

    failures = []
    if len(snapshots) != BEST_WARMUP_CAPTURE_FRAMES:
        failures.append(
            f"warmup snapshots: expected {BEST_WARMUP_CAPTURE_FRAMES}, got {len(snapshots)}"
        )
    if abs(alpha_error - EXPECTED_ALPHA_ERROR_PERCENT) > 0.05:
        failures.append(
            "alpha error: expected approximately "
            f"{EXPECTED_ALPHA_ERROR_PERCENT:.6f}%, got {alpha_error:.6f}%"
        )
    if abs(l2_error - EXPECTED_L2_ERROR) > 5e-5:
        failures.append(
            f"L2 error: expected approximately {EXPECTED_L2_ERROR:.10f}, got {l2_error:.10f}"
        )

    if failures:
        details = "\n  - ".join(failures)
        raise RuntimeError(f"Best-run reproduction verification failed:\n  - {details}")

    print("\n[VERIFIED] Historical best seed-77 trajectory reproduced.")
    print(f"[VERIFIED] alpha error: {alpha_error:.6f}%")
    print(f"[VERIFIED] relative L2 error: {l2_error:.10f}")
    print(f"[VERIFIED] warmup snapshots: {len(snapshots)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the public TGSR High-Peclet transfer experiment.")
    parser.add_argument(
        "--seed",
        type=int,
        default=BEST_SEED,
        help="Random seed for the source and target runs (77 reproduces the paper best run).",
    )
    parser.add_argument(
        "--experiment-name",
        default="tgsr_high_peclet_seed77",
        help="Base results directory name.",
    )
    parser.add_argument("--pruning-mode", default="gmm", choices=["gmm", "kmeans", "percentile"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_exp = os.path.join(args.experiment_name, f"source_{SOURCE_TASK}")
    target_exp = os.path.join(args.experiment_name, f"target_{TARGET_TASK}", "tgsr")

    print("=" * 72)
    print("TGSR-PINN public demo: High-Peclet source-to-target transfer")
    print(f"Seed: {args.seed}")
    print("=" * 72)

    source_model, source_dir = find_existing_source_model(source_exp, args.seed)
    if source_model:
        print(f"[INFO] Reusing source model: {source_model}")
    else:
        source_dir = run_main(
            config_path=repo_path("configs", f"{SOURCE_TASK}.json"),
            exp_name=source_exp,
            seed=args.seed,
            transfer_mode="none",
        )
        source_model = find_latest_model(source_dir)

    if not source_model:
        print("[ERROR] Source model was not found. Target transfer cannot start.")
        return 1

    target_dir = run_main(
        config_path=repo_path("configs", f"{TARGET_TASK}.json"),
        exp_name=target_exp,
        seed=args.seed,
        transfer_mode="tgsr",
        load_model=source_model,
        pruning_mode=args.pruning_mode,
    )
    if not target_dir:
        print("[ERROR] Target TGSR run did not produce an output directory.")
        return 1

    if args.seed == BEST_SEED:
        try:
            verify_best_seed77_result(target_dir)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            print(f"\n[ERROR] {exc}")
            return 2

    print("\n[SUCCESS] TGSR High-Peclet run completed.")
    print(f"[RESULTS] {target_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
