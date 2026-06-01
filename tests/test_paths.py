import json
from pathlib import Path

from src.storage.checkpoints import (
    analysis_dir,
    cross_dataset_dir,
    metadata_dir,
    student_dir,
    teacher_dir,
)
from src.analysis.loaders import load_all_runs, load_runs, load_teacher


def test_path_helpers_are_dataset_first():
    assert teacher_dir("imdb") == Path("results/checkpoints/imdb/teacher")
    assert student_dir("imdb", "kd_full") == Path("results/checkpoints/imdb/student/kd_full")
    assert metadata_dir("imdb", "teacher") == Path("results/metadata/imdb/teacher")
    assert metadata_dir("imdb", "student", "kd_logit") == Path("results/metadata/imdb/student/kd_logit")
    assert analysis_dir("imdb") == Path("results/analysis/imdb")
    assert cross_dataset_dir() == Path("results/analysis/cross_dataset")


def _write_metadata(path: Path, condition_name: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "test": {"macro_f1": 0.5, "micro_f1": 0.5, "accuracy": 0.5, "ECE": 0.1, "NLL": 0.6, "Brier": 0.3},
        "dev": {"macro_f1": 0.55},
    }
    payload = {"run": {"condition": condition_name}, "metrics": metrics, "training": {"history": []}}
    path.write_text(json.dumps(payload))


def test_loaders_read_new_layout(tmp_path):
    meta_root = tmp_path / "metadata"
    # one dataset with a teacher row and one student condition (ce_only)
    _write_metadata(meta_root / "imdb" / "teacher" / "run_metadata.json", None)
    _write_metadata(meta_root / "imdb" / "student" / "ce_only" / "run_metadata.json", "ce_only")

    runs = load_runs("imdb", meta_root)
    ce_row = runs[runs["condition"] == "ce_only"].iloc[0]
    assert bool(ce_row["valid"]) is True

    all_runs = load_all_runs(meta_root)
    assert set(all_runs["dataset"]) == {"imdb"}  # discovered via the student/ subdir

    teacher = load_teacher("imdb", meta_root)
    assert teacher["test_macro_f1"] == 0.5
