"""One-off migration to the dataset-first results/ layout.

Moves artifacts from the old stage-first trees into the new single root:

    checkpoints/teachers/<d>/            -> results/checkpoints/<d>/teacher/
    checkpoints/students/<d>/<c>/        -> results/checkpoints/<d>/student/<c>/
    results/teachers/<d>/run_metadata    -> results/metadata/<d>/teacher/run_metadata
    results/students/<d>/<c>/run_metadata-> results/metadata/<d>/student/<c>/run_metadata

results/analysis/ is already dataset-first and is left untouched. Checkpoints are
gitignored, so they are moved with shutil; metadata is git-tracked, so it is
moved with ``git mv`` (falling back to shutil + ``git add``). Idempotent: targets
that already exist are skipped. Use --dry-run to preview.

Usage
-----
    python scripts/migrate_results_layout.py --dry-run
    python scripts/migrate_results_layout.py
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OLD_CKPT = REPO_ROOT / "checkpoints"
OLD_RESULTS = REPO_ROOT / "results"
NEW_CKPT = OLD_RESULTS / "checkpoints"
NEW_META = OLD_RESULTS / "metadata"


def main() -> None:
    args = _parse_args()
    moves = _planned_moves()
    if not moves:
        print("Nothing to migrate (old trees absent or already migrated).")
        return

    for src, dst, tracked in moves:
        rel_src, rel_dst = src.relative_to(REPO_ROOT), dst.relative_to(REPO_ROOT)
        if dst.exists():
            print(f"  skip (target exists): {rel_dst}")
            continue
        print(f"  {'[dry-run] ' if args.dry_run else ''}{rel_src} -> {rel_dst}")
        if not args.dry_run:
            _move(src, dst, tracked=tracked)

    if not args.dry_run:
        _prune_empty(OLD_CKPT, OLD_RESULTS / "teachers", OLD_RESULTS / "students")
    print("Done." if not args.dry_run else "Dry run complete; no files moved.")


def _planned_moves() -> list[tuple[Path, Path, bool]]:
    """(src, dst, tracked) triples. tracked metadata uses git mv."""
    moves: list[tuple[Path, Path, bool]] = []

    # Checkpoints (untracked): move whole directories.
    for d in _children(OLD_CKPT / "teachers"):
        moves.append((d, NEW_CKPT / d.name / "teacher", False))
    for d in _children(OLD_CKPT / "students"):
        for cond in _children(d):
            moves.append((cond, NEW_CKPT / d.name / "student" / cond.name, False))

    # Metadata (git-tracked): move the run_metadata.json files.
    for d in _children(OLD_RESULTS / "teachers"):
        src = d / "run_metadata.json"
        if src.exists():
            moves.append((src, NEW_META / d.name / "teacher" / "run_metadata.json", True))
    for d in _children(OLD_RESULTS / "students"):
        for cond in _children(d):
            src = cond / "run_metadata.json"
            if src.exists():
                moves.append((src, NEW_META / d.name / "student" / cond.name / "run_metadata.json", True))
    return moves


def _move(src: Path, dst: Path, *, tracked: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if tracked and _git_mv(src, dst):
        return
    shutil.move(str(src), str(dst))
    if tracked:
        subprocess.run(["git", "add", str(dst)], cwd=REPO_ROOT, check=False)


def _git_mv(src: Path, dst: Path) -> bool:
    result = subprocess.run(
        ["git", "mv", str(src.relative_to(REPO_ROOT)), str(dst.relative_to(REPO_ROOT))],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    return result.returncode == 0


def _children(path: Path) -> list[Path]:
    return [p for p in sorted(path.iterdir()) if p.is_dir()] if path.is_dir() else []


def _prune_empty(*roots: Path) -> None:
    """Remove an old root entirely if it holds only (now-empty) directories."""
    for root in roots:
        if root.is_dir() and not any(p.is_file() for p in root.rglob("*")):
            shutil.rmtree(root)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print planned moves without moving anything")
    return parser.parse_args()


if __name__ == "__main__":
    main()
