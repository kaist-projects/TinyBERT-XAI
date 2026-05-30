"""Student training/evaluation pipeline contracts."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from tqdm.auto import tqdm

from tinybert_xai.checkpoints import load_state_dict, results_dir, save_state_dict, student_dir, validate_run_artifacts
from tinybert_xai.conditions import ConditionSpec
from tinybert_xai.datasets import build_loader
from tinybert_xai.earlystop import EarlyStopper
from tinybert_xai.eval import (
    EvaluationResult,
    TeacherStudentAnalysis,
    collect_probabilities,
    compute_teacher_student_analysis,
    evaluate,
)
from tinybert_xai.losses import compute_student_losses
from tinybert_xai.models import load_classifier, load_tokenizer
from tinybert_xai.projections import HiddenProjection
from tinybert_xai.runlog import (
    RunMetadata,
    TrainEpochEntry,
    collect_hardware,
    make_run_id,
    optimization_block,
    patch_metadata_file,
    reproducibility_block,
    write_run_metadata,
)
from tinybert_xai.training import log_epoch, run_training_epoch
from tinybert_xai.utils import clone_state_dict_cpu, count_params, training_autocast

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from tinybert_xai.config import Config
    from tinybert_xai.datasets import DatasetSpec


@dataclass(frozen=True)
class StudentData:
    tokenizer: PreTrainedTokenizerBase
    train_loader: DataLoader
    dev_loader: DataLoader
    train_size: int
    dev_size: int


@dataclass(frozen=True)
class StudentModel:
    model: "PreTrainedModel"
    optimizer: torch.optim.Optimizer
    parameter_count: int
    projections: HiddenProjection | None = None
    projection_parameter_count: int | None = None


@dataclass(frozen=True)
class StudentEpochStats:
    loss_total_mean: float
    loss_means: dict[str, float]
    grad_norm_mean: float
    global_step: int
    epoch_time_seconds: float


@dataclass
class StudentTrainingResult:
    best_state: dict[str, torch.Tensor]
    best_epoch: int
    early_stopped: bool
    history: list[dict]
    train_time_seconds: float
    checkpoint_dir: Path


@dataclass(frozen=True)
class StudentEvaluationResult:
    metadata_path: Path
    dev_size: int
    test_size: int
    dev_result: EvaluationResult
    test_result: EvaluationResult
    test_metrics: dict
    teacher_student_analysis: TeacherStudentAnalysis | None = None


def start_student_metadata(
    cfg: "Config",
    spec: "DatasetSpec",
    cond: ConditionSpec,
    device: str,
) -> RunMetadata:
    hardware = collect_hardware(device)
    model = {
        "student_checkpoint": cfg.student_checkpoint,
        "tokenizer": cfg.tokenizer_checkpoint,
    }
    if cond.uses_teacher:
        model["teacher_checkpoint"] = cfg.teacher_checkpoint

    return RunMetadata(
        schema_version="2",
        run={
            "run_id": make_run_id("student", spec.name, cond.name),
            "stage": "student",
            "condition": cond.name,
        },
        dataset={
            "name": spec.hf_path,
            "config": spec.hf_config,
            "num_labels": spec.num_labels,
            "label_names": spec.label_names,
            "splits": {},
            "max_seq_length": cfg.max_seq_length,
            "truncation": True,
            "padding": "max_length",
        },
        model=model,
        optimization=optimization_block(cfg),
        checkpoint_selection={
            "monitor": "dev_macro_f1",
            "mode": "max",
            "patience": cfg.patience,
            "best_epoch": None,
            "early_stopped": None,
            "checkpoint": None,
        },
        reproducibility=reproducibility_block(cfg),
        environment=hardware,
    )


def load_student_data(cfg: "Config", spec: "DatasetSpec") -> StudentData:
    tokenizer = load_tokenizer(cfg.tokenizer_checkpoint)
    train_loader = build_loader(
        spec,
        "train",
        tokenizer,
        max_length=cfg.max_seq_length,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        seed=cfg.seed,
    )
    dev_loader = build_loader(
        spec,
        "validation",
        tokenizer,
        max_length=cfg.max_seq_length,
        batch_size=cfg.eval_batch_size,
    )
    return StudentData(
        tokenizer=tokenizer,
        train_loader=train_loader,
        dev_loader=dev_loader,
        train_size=len(train_loader.dataset),
        dev_size=len(dev_loader.dataset),
    )


def prepare_student_model(cfg: "Config", spec: "DatasetSpec", cond: ConditionSpec, device: str) -> StudentModel:
    model = load_classifier(cfg.student_checkpoint, spec.num_labels, device)
    projections = HiddenProjection().to(torch.device(device)) if cond.hidden else None
    optimizer = torch.optim.AdamW(_trainable_parameters(model, projections), lr=cfg.learning_rate)
    projection_parameter_count = count_params(projections) if projections is not None else None
    parameter_count = count_params(model) + (projection_parameter_count or 0)
    return StudentModel(
        model=model,
        optimizer=optimizer,
        parameter_count=parameter_count,
        projections=projections,
        projection_parameter_count=projection_parameter_count,
    )


def fine_tune_student(
    cfg: "Config",
    spec: "DatasetSpec",
    cond: ConditionSpec,
    data: StudentData,
    student: StudentModel,
    *,
    device: str,
    teacher_model: "PreTrainedModel | None" = None,
) -> StudentTrainingResult:
    if cond.uses_teacher and teacher_model is None:
        raise RuntimeError(f"Condition {cond.name!r} requires a teacher model")
    if cond.hidden and student.projections is None:
        raise RuntimeError(f"Condition {cond.name!r} requires hidden projections")

    stopper = EarlyStopper(patience=cfg.patience, mode="max")
    history: list[dict] = []
    best_state: dict[str, torch.Tensor] | None = None
    early_stopped = False
    global_step = 0

    ckpt_dir = student_dir(spec.name, cond.name)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    total_train_start = time.perf_counter()

    for epoch in range(cfg.num_epochs):
        epoch_stats = train_student_epoch(
            student.model,
            data.train_loader,
            student.optimizer,
            cond,
            projections=student.projections,
            teacher_model=teacher_model,
            device=device,
            seed=cfg.seed,
            epoch=epoch,
            global_step=global_step,
            precision=cfg.precision,
        )
        global_step = epoch_stats.global_step

        dev_result = evaluate(
            student.model,
            data.dev_loader,
            device=device,
            num_classes=spec.num_labels,
        )
        history.append(_student_epoch_entry(epoch_stats, dev_result, epoch))

        log_epoch(
            epoch,
            epoch_stats.loss_total_mean,
            dev_result.macro_f1,
            dev_result.accuracy,
            epoch_stats.epoch_time_seconds,
        )

        save_state_dict(student.model, ckpt_dir / f"epoch_{epoch}.pt")

        is_best, should_stop = stopper.update(dev_result.macro_f1, epoch)
        if is_best:
            best_state = clone_state_dict_cpu(student.model)
        if should_stop:
            early_stopped = True
            print(f"  Early stop triggered after epoch {epoch} (no improvement for {cfg.patience} epochs)")
            break

    if best_state is None:
        raise RuntimeError("No valid epoch completed - check for NaN losses")

    return StudentTrainingResult(
        best_state=best_state,
        best_epoch=stopper.best_step,
        early_stopped=early_stopped,
        history=history,
        train_time_seconds=time.perf_counter() - total_train_start,
        checkpoint_dir=ckpt_dir,
    )


def train_student_epoch(
    model: "PreTrainedModel",
    loader: "DataLoader",
    optimizer: torch.optim.Optimizer,
    cond: ConditionSpec,
    *,
    projections: HiddenProjection | None,
    teacher_model: "PreTrainedModel | None",
    device: str,
    seed: int,
    epoch: int,
    global_step: int,
    precision: str,
) -> StudentEpochStats:
    model.train()
    if projections is not None:
        projections.train()
    if teacher_model is not None:
        teacher_model.eval()

    trainable_params = _trainable_parameters(model, projections)
    result = run_training_epoch(
        loader,
        optimizer,
        batch_loss_fn=lambda batch: _student_batch_losses(
            model,
            batch,
            cond,
            projections=projections,
            teacher_model=teacher_model,
            device=device,
            precision=precision,
        ),
        parameters=trainable_params,
        device=device,
        seed=seed,
        epoch=epoch,
        global_step=global_step,
        progress_factory=tqdm,
    )

    return StudentEpochStats(
        loss_total_mean=result.stats.loss.total,
        loss_means=result.stats.loss.component_means(),
        grad_norm_mean=result.stats.grad_norm_mean,
        global_step=result.global_step,
        epoch_time_seconds=result.epoch_time_seconds,
    )


def save_student_training_result(
    meta: RunMetadata,
    result: StudentTrainingResult,
    spec: "DatasetSpec",
    cond: ConditionSpec,
) -> tuple[Path, Path]:
    best_ckpt_path = result.checkpoint_dir / "best.pt"
    best_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result.best_state, best_ckpt_path)

    meta.checkpoint_selection["best_epoch"] = result.best_epoch
    meta.checkpoint_selection["early_stopped"] = result.early_stopped
    meta.checkpoint_selection["checkpoint"] = str(best_ckpt_path)
    meta.training = {
        "epochs_completed": len(result.history),
        "train_time_seconds": result.train_time_seconds,
        "history": result.history,
    }

    metadata_path = results_dir("student", spec.name, cond.name) / "run_metadata.json"
    write_run_metadata(meta, metadata_path)
    return best_ckpt_path, metadata_path


def evaluate_saved_student(
    cfg: "Config",
    spec: "DatasetSpec",
    cond: ConditionSpec,
    *,
    device: str,
    teacher_model: "PreTrainedModel | None" = None,
) -> StudentEvaluationResult:
    ckpt_path = student_dir(spec.name, cond.name) / "best.pt"
    metadata_path = results_dir("student", spec.name, cond.name) / "run_metadata.json"
    _require_student_artifacts(ckpt_path, metadata_path, cond)

    tokenizer = load_tokenizer(cfg.tokenizer_checkpoint)
    model = load_classifier(cfg.student_checkpoint, spec.num_labels, device)
    load_state_dict(model, ckpt_path, device)

    dev_loader, test_loader = _build_eval_loaders(cfg, spec, tokenizer)

    dev_result = evaluate(model, dev_loader, device=device, num_classes=spec.num_labels)
    test_result = evaluate(model, test_loader, device=device, num_classes=spec.num_labels)
    teacher_student_analysis = _teacher_student_analysis(model, teacher_model, test_loader, device)

    return StudentEvaluationResult(
        metadata_path=metadata_path,
        dev_size=len(dev_loader.dataset),
        test_size=len(test_loader.dataset),
        dev_result=dev_result,
        test_result=test_result,
        test_metrics=asdict(test_result),
        teacher_student_analysis=teacher_student_analysis,
    )


def save_student_evaluation_result(result: StudentEvaluationResult) -> None:
    def mutate(metadata: dict) -> None:
        metadata["dataset"]["splits"]["test"] = result.test_size
        test_metrics = dict(result.test_metrics)
        if result.teacher_student_analysis is not None:
            test_metrics["teacher_student_analysis"] = asdict(result.teacher_student_analysis)

        metadata["metrics"] = {
            "dev": asdict(result.dev_result),
            "test": test_metrics,
        }

    patch_metadata_file(result.metadata_path, mutate)


def format_student_eval_summary(result: StudentEvaluationResult) -> str:
    """Render the dev/test evaluation summary as a printable multi-line string."""
    dev, test = result.dev_result, result.test_result
    lines = [
        f"  dev={result.dev_size}  test={result.test_size}",
        f"  dev macro-F1  : {dev.macro_f1:.4f}",
        f"  dev accuracy  : {dev.accuracy:.4f}",
        f"  dev ECE       : {dev.ECE:.4f}",
        f"  test macro-F1 : {test.macro_f1:.4f}",
        f"  test accuracy : {test.accuracy:.4f}",
        f"  test ECE      : {test.ECE:.4f}",
        f"  per-class F1  : {[f'{v:.3f}' for v in test.per_class_f1]}",
    ]
    if result.teacher_student_analysis is not None:
        analysis = result.teacher_student_analysis
        lines.append(f"  top1 agreement: {analysis.top1_agreement:.4f}")
        lines.append(f"  teacher->student KL: {analysis.teacher_student_kl:.4f}")

    pass_fail = "PASS" if test.macro_f1 >= 0.33 else "FAIL"
    lines.append(f"  DoD check (test macro-F1 >= 0.33): {pass_fail}")
    return "\n".join(lines)


def _student_batch_losses(
    model: "PreTrainedModel",
    batch: dict[str, torch.Tensor],
    cond: ConditionSpec,
    *,
    projections: HiddenProjection | None,
    teacher_model: "PreTrainedModel | None",
    device: str,
    precision: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    teacher_out = None
    if cond.uses_teacher:
        if teacher_model is None:
            raise RuntimeError(f"Condition {cond.name!r} requires a teacher model")
        teacher_batch = {k: v for k, v in batch.items() if k != "labels"}
        with torch.no_grad(), training_autocast(device, precision):
            teacher_out = teacher_model(**teacher_batch)

    with training_autocast(device, precision):
        student_out = model(**batch)
    return compute_student_losses(
        student_out,
        teacher_out,
        cond,
        projections=projections,
        attention_mask=batch["attention_mask"],
    )


def _trainable_parameters(
    model: "PreTrainedModel",
    projections: HiddenProjection | None,
) -> list[torch.nn.Parameter]:
    params = list(model.parameters())
    if projections is not None:
        params.extend(projections.parameters())
    return params


def _build_eval_loaders(
    cfg: "Config",
    spec: "DatasetSpec",
    tokenizer: "PreTrainedTokenizerBase",
) -> tuple["DataLoader", "DataLoader"]:
    dev_loader = build_loader(
        spec,
        "validation",
        tokenizer,
        max_length=cfg.max_seq_length,
        batch_size=cfg.eval_batch_size,
    )
    test_loader = build_loader(
        spec,
        "test",
        tokenizer,
        max_length=cfg.max_seq_length,
        batch_size=cfg.eval_batch_size,
    )
    return dev_loader, test_loader


def _teacher_student_analysis(
    student_model: "PreTrainedModel",
    teacher_model: "PreTrainedModel | None",
    test_loader: "DataLoader",
    device: str,
) -> TeacherStudentAnalysis | None:
    if teacher_model is None:
        return None

    teacher_probs, labels = collect_probabilities(teacher_model, test_loader, device=device)
    student_probs, student_labels = collect_probabilities(student_model, test_loader, device=device)
    if not torch.equal(labels, student_labels):
        raise RuntimeError("Teacher and student evaluation labels did not align")
    return compute_teacher_student_analysis(teacher_probs, student_probs, labels)


def _student_epoch_entry(
    stats: StudentEpochStats,
    dev_result: EvaluationResult,
    epoch: int,
) -> dict:
    return asdict(
        TrainEpochEntry(
            epoch=epoch,
            global_step=stats.global_step,
            epoch_time_seconds=stats.epoch_time_seconds,
            loss_total=stats.loss_total_mean,
            losses=stats.loss_means,
            grad_norm_mean=stats.grad_norm_mean,
            dev={
                "macro_f1": dev_result.macro_f1,
                "micro_f1": dev_result.micro_f1,
                "accuracy": dev_result.accuracy,
                "ECE": dev_result.ECE,
                "NLL": dev_result.NLL,
                "Brier": dev_result.Brier,
            },
        )
    )


def _require_student_artifacts(ckpt_path: Path, metadata_path: Path, cond: ConditionSpec) -> None:
    validate_run_artifacts(
        ckpt_path,
        metadata_path,
        regenerate_hint="Run scripts/02_train_student.py to regenerate training metadata before evaluation.",
        expected_condition=cond.name,
    )
