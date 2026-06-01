"""scripts/07_run_dataset.py - Run the full factorial sweep for one dataset.

Fine-tunes the teacher (once), then trains + evaluates all 8 student conditions
for the chosen dataset by invoking the existing per-run scripts as subprocesses
(clean per-run GPU memory isolation; reuses the already-tested orchestration).

Resumable: a teacher or condition whose artifacts already exist is skipped
unless --force. A student condition counts as done only when both its best.pt
and a run_metadata.json carrying test metrics exist.

Usage
-----
    conda activate tinybert-xai
    # from repo root
    python scripts/07_run_dataset.py --dataset imdb
    python scripts/07_run_dataset.py --dataset anli --skip-teacher
    python scripts/07_run_dataset.py --dataset imdb --force
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _config_cli import add_config_flag, add_dataset_override, resolve_run_spec  # noqa: E402
from _student_cli import condition_to_flags  # noqa: E402

from src import ConditionSpec, all_conditions  # noqa: E402
from src.storage.checkpoints import metadata_dir, student_dir, teacher_dir  # noqa: E402

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run teacher + all 8 student conditions for one dataset.")
    add_config_flag(parser)
    add_dataset_override(parser)
    parser.add_argument("--skip-teacher", action="store_true", help="assume the teacher checkpoint already exists")
    parser.add_argument("--force", action="store_true", help="re-run even when artifacts already exist")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = resolve_run_spec(args)
    dataset_name = run.dataset
    # Forward the shared config to every per-run subprocess; --dataset, condition,
    # and --eval are passed explicitly below so the sweep always runs all 8.
    config_args = ["--config", str(args.config)] if args.config else []
    print(f"=== Factorial sweep: {dataset_name} ===")

    run_teacher(dataset_name, config_args, skip=args.skip_teacher, force=args.force)

    failures = run_all_conditions(dataset_name, config_args, force=args.force)

    print("\n=== Sweep complete ===")
    if failures:
        print(f"  {len(failures)} condition(s) FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("  all conditions present.")


def run_teacher(dataset_name: str, config_args: list[str], *, skip: bool, force: bool) -> None:
    if skip:
        print("[teacher] skipped (--skip-teacher)")
        return
    if teacher_done(dataset_name) and not force:
        print("[teacher] skipped (checkpoint exists)")
        return
    print("[teacher] training ...")
    run_script("01_train_teacher.py", [*config_args, "--dataset", dataset_name])


def run_all_conditions(dataset_name: str, config_args: list[str], *, force: bool) -> list[str]:
    failures: list[str] = []
    for cond in all_conditions():
        if condition_done(dataset_name, cond) and not force:
            print(f"[{cond.name}] skipped (results exist)")
            continue
        print(f"[{cond.name}] training + evaluating ...")
        try:
            run_script(
                "02_train_student.py",
                [*config_args, "--dataset", dataset_name, *condition_to_flags(cond), "--eval"],
            )
        except subprocess.CalledProcessError:
            print(f"[{cond.name}] FAILED")
            failures.append(cond.name)
    return failures


def teacher_done(dataset_name: str) -> bool:
    return (REPO_ROOT / teacher_dir(dataset_name) / "best.pt").exists()


def condition_done(dataset_name: str, cond: ConditionSpec) -> bool:
    best_ckpt = REPO_ROOT / student_dir(dataset_name, cond.name) / "best.pt"
    metadata = REPO_ROOT / metadata_dir(dataset_name, "student", cond.name) / "run_metadata.json"
    return best_ckpt.exists() and _has_test_metrics(metadata)


def run_script(script_name: str, script_args: list[str]) -> None:
    subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / script_name), *script_args],
        cwd=REPO_ROOT,
        check=True,
    )


def _has_test_metrics(metadata_path: pathlib.Path) -> bool:
    if not metadata_path.exists():
        return False
    try:
        with open(metadata_path) as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("metrics", {}).get("test", {}).get("macro_f1") is not None


if __name__ == "__main__":
    main()
