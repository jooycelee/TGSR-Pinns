import fnmatch
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DOT_CLAUDE = "." + "claude"
DOT_A = "." + "a"
DOT_EA = "." + "ea"
RESULTS_FINAL = "results" + "_final"
PUBLIC_PATTERNS = [
    "README*.md",
    "scripts/*.py",
    "experiments/runners/*.py",
]
PRIVATE_MARKERS = [
    "E:" + "\\",
    "C:" + "\\Users",
    DOT_CLAUDE,
    f'ROOT / "{DOT_A}"',
    f'ROOT / "{DOT_EA}"',
    f'"{DOT_A}"',
    f'"{DOT_EA}"',
    DOT_A + "/",
    DOT_EA + "/",
    "_" + RESULTS_FINAL + "_staging",
    "Paper_Submission_CloseoutMaster",
    "closeout",
]
GENERATED_TRACKED_PATTERNS = [
    "results/*",
    RESULTS_FINAL + "/*",
    "outputs/*",
    "runs/*",
    "logs/*",
    "checkpoints/*",
    "*/__pycache__/*",
    ".pytest_cache/*",
    DOT_CLAUDE + "/*",
    DOT_A + "/*",
    DOT_EA + "/*",
    "B/new.tex",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.npy",
    "*.npz",
    "*.mat",
    "*.h5",
    "*.hdf5",
]


def iter_public_files() -> list[Path]:
    files: set[Path] = set()
    for pattern in PUBLIC_PATTERNS:
        files.update(REPO_ROOT.glob(pattern))
    return sorted(path for path in files if path.is_file())


def test_public_files_do_not_reference_private_artifact_paths():
    offenders: list[str] = []
    for path in iter_public_files():
        text = path.read_text(encoding="utf-8")
        for marker in PRIVATE_MARKERS:
            if marker in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)} contains {marker!r}")

    assert not offenders, "\n".join(offenders)


def git_ls_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def test_tracked_files_do_not_include_generated_or_private_artifacts():
    tracked_files = git_ls_files()
    offenders: list[str] = []

    for file_name in tracked_files:
        normalized = file_name.replace("\\", "/")
        for pattern in GENERATED_TRACKED_PATTERNS:
            if fnmatch.fnmatch(normalized, pattern):
                offenders.append(f"{normalized} matches {pattern!r}")
                break

    assert not offenders, "\n".join(offenders)


def test_no_root_level_experiment_runners_are_tracked():
    root_runners = [
        file_name for file_name in git_ls_files()
        if "/" not in file_name and fnmatch.fnmatch(file_name, "run_*.py")
    ]

    assert not root_runners, f"Root-level experiment runners are tracked: {root_runners}"
