import argparse
import glob
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MAIN_PY = ROOT / "main.py"
SOURCE_TASK = "task2d_diffusion_source_noise005"
TARGET_TASK = "task2d_high_peclet"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the public TGSR High-Peclet transfer experiment.")
    parser.add_argument("--seed", type=int, default=77, help="Random seed for the source and target runs.")
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

    print("\n[SUCCESS] TGSR High-Peclet run completed.")
    print(f"[RESULTS] {target_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
