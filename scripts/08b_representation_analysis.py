"""Checkpoint-forward representation + XAI artifacts for the KD ablation.

Reloads the saved teacher and student classifiers and runs forward passes on a
fixed test sample to produce the artifacts that are not in ``run_metadata.json``:

- ``representation/layer_cka.csv``: linear CKA per mapped pair (no projection
  needed; see ``tinybert_xai/analysis/representations.py`` for why).
- ``representation/attention_kl.csv``: head-averaged KL(teacher || student) of
  attention maps per mapped pair.
- ``figures/cka_mean.png``: mean-CKA heatmap (dataset x condition).
- ``representation/attention/*.png``: teacher-vs-student attention heatmaps for
  representative examples (``ce_only`` and ``kd_full``).
- ``representation/efficiency.json`` + ``figures/efficiency.png``: one
  teacher-vs-student size/latency comparison (architecture is fixed across
  conditions).

Needs the conda env ``tinybert-xai`` (torch) and the ``checkpoints/`` tree.

Usage
-----
    conda activate tinybert-xai
    python scripts/08b_representation_analysis.py            # all datasets
    python scripts/08b_representation_analysis.py --sample-size 128
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from dataclasses import asdict, dataclass  # noqa: E402

from tinybert_xai import (  # noqa: E402
    Config,
    build_loader,
    configure_reproducibility,
    dataset_by_name,
    load_classifier,
    load_tokenizer,
    resolve_device,
)
from tinybert_xai.analysis.cross_dataset import _order_axes  # noqa: E402
from tinybert_xai.analysis.plots import (  # noqa: E402
    plot_attention_pair,
    plot_cross_task_heatmap,
    plot_efficiency,
)
from tinybert_xai.analysis.representations import (  # noqa: E402
    LAYER_MAP,
    attention_kl,
    attention_map_for_example,
    collect_forward_outputs,
    input_ids_for_example,
    layer_cka,
    measure_efficiency,
    select_example_indices,
)
from tinybert_xai.storage.checkpoints import load_state_dict, student_dir, teacher_dir  # noqa: E402
from tinybert_xai.distill.conditions import all_conditions  # noqa: E402

ANALYSIS_ROOT = pathlib.Path("results") / "analysis" / "cross_dataset"
RESULTS_ROOT = pathlib.Path("results")
HEATMAP_CONDITIONS = ("ce_only", "kd_full")
TEACHER_HEATMAP_LAYER = 12
STUDENT_HEATMAP_LAYER = 4


def main() -> None:
    args = _parse_args()
    cfg = Config()
    configure_reproducibility(cfg.seed)
    device = resolve_device(cfg)
    print(f"Device: {device}")

    representation_dir = ANALYSIS_ROOT / "representation"
    figures_dir = ANALYSIS_ROOT / "figures"
    attention_dir = representation_dir / "attention"
    representation_dir.mkdir(parents=True, exist_ok=True)

    datasets = _datasets_to_analyze()
    print(f"Datasets with student runs and a teacher checkpoint: {datasets}")

    cka_rows: list[dict] = []
    attn_rows: list[dict] = []
    efficiency: dict | None = None
    skipped: list[tuple[str, str]] = []
    tokenizer = load_tokenizer(cfg.tokenizer_checkpoint)

    for dataset in datasets:
        try:
            result = _analyze_dataset(
                cfg, dataset, tokenizer, device, args.sample_size, attention_dir,
                need_efficiency=efficiency is None,
            )
        except Exception as exc:  # noqa: BLE001 - one broken dataset must not abort the sweep.
            print(f"[{dataset}] SKIPPED: {exc}")
            skipped.append((dataset, str(exc)))
            _empty_cache(device)
            continue
        cka_rows += result.cka_rows
        attn_rows += result.attn_rows
        efficiency = efficiency or result.efficiency

    _write_csv(cka_rows, representation_dir / "layer_cka.csv")
    _write_csv(attn_rows, representation_dir / "attention_kl.csv")
    _write_cka_heatmap(cka_rows, figures_dir)
    if efficiency is not None:
        _write_efficiency(efficiency, representation_dir, figures_dir)

    print("\nRepresentation analysis complete.")
    _print_summary(cka_rows, attn_rows, efficiency, skipped)


@dataclass
class DatasetAnalysis:
    cka_rows: list[dict]
    attn_rows: list[dict]
    efficiency: dict | None


def _analyze_dataset(
    cfg, dataset, tokenizer, device, sample_size, attention_dir, *, need_efficiency
) -> "DatasetAnalysis":
    spec = dataset_by_name(dataset)
    loader = _sample_loader(cfg, spec, tokenizer, sample_size)
    teacher = _load_teacher(cfg, spec, device)
    teacher_out = collect_forward_outputs(teacher, loader, device)
    print(f"[{dataset}] teacher forward done ({len(teacher_out.labels)} examples)")

    cka_rows: list[dict] = []
    attn_rows: list[dict] = []
    efficiency: dict | None = None
    for condition in _conditions_for(dataset):
        student = _load_student(cfg, spec, condition.name, device)
        student_out = collect_forward_outputs(student, loader, device)
        cka_rows += _cka_rows(dataset, condition.name, layer_cka(student_out, teacher_out))
        attn_rows += _attn_rows(dataset, condition.name, attention_kl(student_out, teacher_out))

        if condition.name in HEATMAP_CONDITIONS:
            _write_attention_heatmaps(
                dataset, condition.name, teacher_out, student_out, tokenizer, attention_dir
            )

        if need_efficiency and efficiency is None:
            efficiency = _measure_efficiency(teacher, student, loader, device)
        del student, student_out
        _empty_cache(device)
    del teacher, teacher_out
    _empty_cache(device)
    return DatasetAnalysis(cka_rows=cka_rows, attn_rows=attn_rows, efficiency=efficiency)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=256, help="test examples per dataset")
    return parser.parse_args()


def _datasets_to_analyze() -> list[str]:
    students_root = RESULTS_ROOT / "students"
    datasets = []
    for path in sorted(students_root.iterdir()):
        if path.is_dir() and (teacher_dir(path.name) / "best.pt").exists():
            datasets.append(path.name)
    return datasets


def _conditions_for(dataset: str) -> list:
    return [c for c in all_conditions() if (student_dir(dataset, c.name) / "best.pt").exists()]


def _sample_loader(cfg, spec, tokenizer, sample_size: int) -> DataLoader:
    loader = build_loader(
        spec, "test", tokenizer, max_length=cfg.max_seq_length, batch_size=cfg.eval_batch_size
    )
    size = min(sample_size, len(loader.dataset))
    subset = Subset(loader.dataset, list(range(size)))
    return DataLoader(subset, batch_size=cfg.eval_batch_size, shuffle=False)


def _load_teacher(cfg, spec, device: str):
    model = load_classifier(cfg.teacher_checkpoint, spec.num_labels, device)
    load_state_dict(model, teacher_dir(spec.name) / "best.pt", device)
    return model


def _load_student(cfg, spec, condition_name: str, device: str):
    model = load_classifier(cfg.student_checkpoint, spec.num_labels, device)
    load_state_dict(model, student_dir(spec.name, condition_name) / "best.pt", device)
    return model


def _cka_rows(dataset: str, condition: str, cka: dict[int, float]) -> list[dict]:
    return [
        {
            "dataset": dataset,
            "condition": condition,
            "pair": f"s{layer}-t{LAYER_MAP[layer]}",
            "cka": value,
        }
        for layer, value in cka.items()
    ]


def _attn_rows(dataset: str, condition: str, kl: dict[int, float]) -> list[dict]:
    return [
        {
            "dataset": dataset,
            "condition": condition,
            "pair": f"s{layer}-t{LAYER_MAP[layer]}",
            "attention_kl": value,
        }
        for layer, value in kl.items()
    ]


def _write_attention_heatmaps(dataset, condition, teacher_out, student_out, tokenizer, out_dir):
    for category, index in select_example_indices(student_out, teacher_out).items():
        teacher_map, length = attention_map_for_example(teacher_out, index, TEACHER_HEATMAP_LAYER)
        student_map, _ = attention_map_for_example(student_out, index, STUDENT_HEATMAP_LAYER)
        token_ids = input_ids_for_example(student_out, index, length)
        tokens = tokenizer.convert_ids_to_tokens(token_ids.tolist())
        plot_attention_pair(
            teacher_map.numpy(),
            student_map.numpy(),
            tokens,
            out_dir,
            f"{dataset}__{condition}__{category}",
            f"{dataset} / {condition} / {category}",
        )


def _measure_efficiency(teacher, student, loader: DataLoader, device: str) -> dict:
    sample_batch = next(iter(loader))
    return asdict(measure_efficiency(teacher, student, sample_batch, device))


def _write_cka_heatmap(cka_rows: list[dict], figures_dir: pathlib.Path) -> None:
    if not cka_rows:
        return
    frame = pd.DataFrame(cka_rows)
    matrix = frame.pivot_table(index="dataset", columns="condition", values="cka", aggfunc="mean")
    matrix = _order_axes(matrix)
    plot_cross_task_heatmap(
        matrix,
        figures_dir,
        "cka_mean",
        "Mean layer CKA (student vs teacher) by dataset x condition",
        fmt=".2f",
        cmap="viridis",
    )


def _write_efficiency(efficiency: dict, representation_dir, figures_dir) -> None:
    (representation_dir / "efficiency.json").write_text(json.dumps(efficiency, indent=2) + "\n")
    plot_efficiency(efficiency, figures_dir)


def _write_csv(rows: list[dict], path: pathlib.Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _empty_cache(device: str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()


def _print_summary(
    cka_rows: list[dict],
    attn_rows: list[dict],
    efficiency: dict | None,
    skipped: list[tuple[str, str]],
) -> None:
    if cka_rows:
        cka = pd.DataFrame(cka_rows)
        print(f"  datasets        : {sorted(cka['dataset'].unique())}")
        print(f"  CKA rows        : {len(cka)} (mean {cka['cka'].mean():.3f})")
    if attn_rows:
        attn = pd.DataFrame(attn_rows)
        print(f"  attention-KL    : {len(attn)} rows (mean {attn['attention_kl'].mean():.4f})")
    if efficiency:
        print(
            f"  efficiency      : student {efficiency['parameter_ratio']:.1f}x smaller, "
            f"{efficiency['speedup']:.1f}x faster"
        )
    if skipped:
        print(f"  skipped         : {[name for name, _ in skipped]}")


if __name__ == "__main__":
    main()
