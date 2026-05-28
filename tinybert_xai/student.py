"""Student training/evaluation pipeline contracts."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from tqdm.auto import tqdm

from tinybert_xai.checkpoints import load_state_dict, results_dir, save_state_dict, student_dir
from tinybert_xai.conditions import ConditionSpec
from tinybert_xai.datasets import build_loader
from tinybert_xai.earlystop import EarlyStopper
from tinybert_xai.eval import EfficiencyMetrics, EvaluationResult, compute_efficiency, evaluate
from tinybert_xai.losses import compute_student_losses
from tinybert_xai.models import load_classifier, load_tokenizer
from tinybert_xai.runlog import (
    RunMetadata,
    TrainEpochEntry,
    collect_hardware,
    dumps_metadata_payload,
    make_run_id,
    write_run_metadata,
)
from tinybert_xai.utils import clone_state_dict_cpu, count_params, move_batch_to_device, training_autocast

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from tinybert_xai.config import Config
    from tinybert_xai.datasets import DatasetSpec


@dataclass(frozen=True)
class StudentData:
    tokenizer: "PreTrainedTokenizerBase"
    train_loader: "DataLoader"
    dev_loader: "DataLoader"
    train_size: int
    dev_size: int


@dataclass(frozen=True)
class StudentModel:
    model: "PreTrainedModel"
    optimizer: torch.optim.Optimizer
    parameter_count: int


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
    best_dev_macro_f1: float
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
    efficiency: EfficiencyMetrics


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
        optimization={
            "optimizer": "AdamW",
            "learning_rate": cfg.learning_rate,
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "scheduler": None,
            "grad_clip": None,
            "precision": cfg.precision,
            "train_batch_size": cfg.train_batch_size,
            "eval_batch_size": cfg.eval_batch_size,
            "num_epochs": cfg.num_epochs,
        },
        checkpoint_selection={
            "monitor": "dev_macro_f1",
            "mode": "max",
            "patience": cfg.patience,
            "best_epoch": None,
            "early_stopped": None,
            "checkpoint": None,
        },
        reproducibility={
            "seed": cfg.seed,
            "deterministic_algorithms": True,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "shuffle_seed_scheme": "seed + epoch",
        },
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


def prepare_student_model(cfg: "Config", spec: "DatasetSpec", device: str) -> StudentModel:
    model = load_classifier(cfg.student_checkpoint, spec.num_labels, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    return StudentModel(model=model, optimizer=optimizer, parameter_count=count_params(model))


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

        print(
            f"  epoch {epoch}  "
            f"train_loss={epoch_stats.loss_total_mean:.4f}  "
            f"dev_macro_f1={dev_result.macro_f1:.4f}  "
            f"dev_acc={dev_result.accuracy:.4f}  "
            f"({epoch_stats.epoch_time_seconds:.1f}s)"
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
        best_dev_macro_f1=stopper.best_value,
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
    teacher_model: "PreTrainedModel | None",
    device: str,
    seed: int,
    epoch: int,
    global_step: int,
    precision: str,
) -> StudentEpochStats:
    epoch_start = time.perf_counter()
    model.train()
    if teacher_model is not None:
        teacher_model.eval()

    generator = getattr(loader, "generator", None)
    if generator is not None:
        generator.manual_seed(seed + epoch)

    loss_total_sum = 0.0
    loss_sums: dict[str, float] = {}
    grad_norm_sum = 0.0
    n_batches = 0

    pbar = tqdm(loader, total=len(loader), desc=f"epoch {epoch}", unit="batch")
    for batch in pbar:
        batch = move_batch_to_device(batch, device)
        loss, losses = _student_batch_losses(
            model,
            batch,
            cond,
            teacher_model=teacher_model,
            device=device,
            precision=precision,
        )

        if not torch.isfinite(loss):
            pbar.write(
                f"  [WARN] non-finite loss at epoch {epoch} "
                f"step {global_step}: {loss.item():.6f} - skipping batch"
            )
            optimizer.zero_grad()
            continue

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))
        optimizer.step()
        optimizer.zero_grad()

        loss_total_sum += loss.item()
        for name, value in losses.items():
            loss_sums[name] = loss_sums.get(name, 0.0) + value
        grad_norm_sum += grad_norm.item()
        n_batches += 1
        global_step += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return StudentEpochStats(
        loss_total_mean=loss_total_sum / max(n_batches, 1),
        loss_means={name: value / max(n_batches, 1) for name, value in loss_sums.items()},
        grad_norm_mean=grad_norm_sum / max(n_batches, 1),
        global_step=global_step,
        epoch_time_seconds=time.perf_counter() - epoch_start,
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
        "best_dev_macro_f1": result.best_dev_macro_f1,
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
) -> StudentEvaluationResult:
    ckpt_path = student_dir(spec.name, cond.name) / "best.pt"
    metadata_path = results_dir("student", spec.name, cond.name) / "run_metadata.json"
    _require_student_artifacts(ckpt_path, metadata_path, cond)

    tokenizer = load_tokenizer(cfg.tokenizer_checkpoint)
    model = load_classifier(cfg.student_checkpoint, spec.num_labels, device)
    load_state_dict(model, ckpt_path, device)

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

    dev_result = evaluate(model, dev_loader, device=device, num_classes=spec.num_labels)
    test_result = evaluate(model, test_loader, device=device, num_classes=spec.num_labels)
    efficiency = compute_efficiency(
        model,
        tokenizer,
        device=device,
        max_length=cfg.max_seq_length,
        batch_size=cfg.eval_batch_size,
    )

    return StudentEvaluationResult(
        metadata_path=metadata_path,
        dev_size=len(dev_loader.dataset),
        test_size=len(test_loader.dataset),
        dev_result=dev_result,
        test_result=test_result,
        test_metrics=_student_test_metrics(test_result),
        efficiency=efficiency,
    )


def save_student_evaluation_result(result: StudentEvaluationResult) -> None:
    with open(result.metadata_path) as f:
        metadata = json.load(f)

    metadata["dataset"]["splits"]["test"] = result.test_size
    metadata["metrics"] = {
        "dev": asdict(result.dev_result),
        "test": result.test_metrics,
    }
    metadata["efficiency"] = asdict(result.efficiency)

    with open(result.metadata_path, "w") as f:
        f.write(dumps_metadata_payload(metadata))
        f.write("\n")


def _student_batch_losses(
    model: "PreTrainedModel",
    batch: dict[str, torch.Tensor],
    cond: ConditionSpec,
    *,
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
    return compute_student_losses(student_out, teacher_out, cond)


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


def _student_test_metrics(test_result: EvaluationResult) -> dict:
    return asdict(test_result)


def _require_student_artifacts(ckpt_path: Path, metadata_path: Path, cond: ConditionSpec) -> None:
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Run scripts/02_train_student.py first."
        )
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"run_metadata.json not found: {metadata_path}\n"
            "Run scripts/02_train_student.py first."
        )
    with open(metadata_path) as f:
        metadata = json.load(f)
    if metadata.get("schema_version") != "2":
        raise RuntimeError(
            f"run_metadata.json is not schema v2: {metadata_path}\n"
            "Run scripts/02_train_student.py to regenerate training metadata before evaluation."
        )
    if metadata.get("run", {}).get("condition") != cond.name:
        raise RuntimeError(
            f"run_metadata.json condition does not match {cond.name!r}: {metadata_path}\n"
            "Run scripts/02_train_student.py for the requested condition."
        )
