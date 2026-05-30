"""Teacher training/evaluation pipeline contracts.

The public functions in this module are medium-level contracts used by the
scripts. Private helpers carry the lower-level tensor and metadata mechanics.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from tqdm.auto import tqdm
from transformers import set_seed as hf_set_seed

from tinybert_xai.checkpoints import load_state_dict, results_dir, save_state_dict, teacher_dir, validate_run_artifacts
from tinybert_xai.datasets import build_loader
from tinybert_xai.earlystop import EarlyStopper
from tinybert_xai.eval import EvaluationResult, evaluate
from tinybert_xai.models import load_classifier, load_tokenizer
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
class TeacherData:
    tokenizer: PreTrainedTokenizerBase
    train_loader: DataLoader
    dev_loader: DataLoader
    train_size: int
    dev_size: int


@dataclass(frozen=True)
class TeacherModel:
    model: PreTrainedModel
    optimizer: torch.optim.Optimizer
    parameter_count: int


@dataclass(frozen=True)
class TeacherEpochStats:
    loss_total_mean: float
    loss_ce_mean: float
    grad_norm_mean: float
    global_step: int
    epoch_time_seconds: float


@dataclass
class TeacherTrainingResult:
    best_state: dict[str, torch.Tensor]
    best_epoch: int
    early_stopped: bool
    history: list[dict]
    train_time_seconds: float
    checkpoint_dir: Path


@dataclass(frozen=True)
class TeacherEvaluationResult:
    metadata_path: Path
    dev_size: int
    test_size: int
    dev_result: EvaluationResult
    test_result: EvaluationResult
    test_metrics: dict


def configure_reproducibility(seed: int) -> None:
    # Required for deterministic matmul on CUDA >= 10.2; cuBLAS reads it on
    # first call, so set before any model forward.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    hf_set_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(cfg: "Config") -> str:
    return cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")


def start_teacher_metadata(cfg: "Config", spec: "DatasetSpec", device: str) -> RunMetadata:
    hardware = collect_hardware(device)
    return RunMetadata(
        schema_version="2",
        run={
            "run_id": make_run_id("teacher", spec.name),
            "stage": "teacher",
            "condition": None,
        },
        dataset={
            "name": spec.hf_path,
            "config": spec.hf_config,
            "num_labels": spec.num_labels,
            "label_names": spec.label_names,
            "input_type": spec.input_type,
            "split_scheme": spec.split_scheme,
            "splits": {},
            "max_seq_length": cfg.max_seq_length,
            "truncation": True,
            "padding": "max_length",
        },
        model={
            "checkpoint": cfg.teacher_checkpoint,
            "tokenizer": cfg.tokenizer_checkpoint,
        },
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


def load_teacher_data(cfg: "Config", spec: "DatasetSpec") -> TeacherData:
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
    return TeacherData(
        tokenizer=tokenizer,
        train_loader=train_loader,
        dev_loader=dev_loader,
        train_size=len(train_loader.dataset),
        dev_size=len(dev_loader.dataset),
    )


def prepare_teacher_model(cfg: "Config", spec: "DatasetSpec", device: str) -> TeacherModel:
    model = load_classifier(cfg.teacher_checkpoint, spec.num_labels, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    return TeacherModel(model=model, optimizer=optimizer, parameter_count=count_params(model))


def load_trained_teacher(cfg: "Config", spec: "DatasetSpec", device: str) -> "PreTrainedModel":
    """Load the fine-tuned teacher checkpoint for a dataset, ready for inference."""
    model = load_classifier(cfg.teacher_checkpoint, spec.num_labels, device)
    load_state_dict(model, teacher_dir(spec.name) / "best.pt", device)
    model.eval()
    return model


def fine_tune_teacher(
    cfg: "Config",
    spec: "DatasetSpec",
    data: TeacherData,
    teacher: TeacherModel,
    *,
    device: str,
) -> TeacherTrainingResult:
    stopper = EarlyStopper(patience=cfg.patience, mode="max")
    history: list[dict] = []
    best_state: dict[str, torch.Tensor] | None = None
    early_stopped = False
    global_step = 0

    ckpt_dir = teacher_dir(spec.name)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    total_train_start = time.perf_counter()

    for epoch in range(cfg.num_epochs):
        epoch_stats = train_teacher_epoch(
            teacher.model,
            data.train_loader,
            teacher.optimizer,
            device=device,
            seed=cfg.seed,
            epoch=epoch,
            global_step=global_step,
            precision=cfg.precision,
        )
        global_step = epoch_stats.global_step

        dev_result = evaluate(
            teacher.model,
            data.dev_loader,
            device=device,
            num_classes=spec.num_labels,
        )
        history.append(_teacher_epoch_entry(epoch_stats, dev_result, epoch))

        log_epoch(
            epoch,
            epoch_stats.loss_total_mean,
            dev_result.macro_f1,
            dev_result.accuracy,
            epoch_stats.epoch_time_seconds,
        )

        save_state_dict(teacher.model, ckpt_dir / f"epoch_{epoch}.pt")

        is_best, should_stop = stopper.update(dev_result.macro_f1, epoch)
        if is_best:
            best_state = clone_state_dict_cpu(teacher.model)
        if should_stop:
            early_stopped = True
            print(f"  Early stop triggered after epoch {epoch} (no improvement for {cfg.patience} epochs)")
            break

    if best_state is None:
        raise RuntimeError("No valid epoch completed - check for NaN losses")

    return TeacherTrainingResult(
        best_state=best_state,
        best_epoch=stopper.best_step,
        early_stopped=early_stopped,
        history=history,
        train_time_seconds=time.perf_counter() - total_train_start,
        checkpoint_dir=ckpt_dir,
    )


def train_teacher_epoch(
    model: "PreTrainedModel",
    loader: "DataLoader",
    optimizer: torch.optim.Optimizer,
    *,
    device: str,
    seed: int,
    epoch: int,
    global_step: int,
    precision: str,
) -> TeacherEpochStats:
    model.train()

    result = run_training_epoch(
        loader,
        optimizer,
        batch_loss_fn=lambda batch: _teacher_batch_loss_with_components(
            model,
            batch,
            device=device,
            precision=precision,
        ),
        parameters=list(model.parameters()),
        device=device,
        seed=seed,
        epoch=epoch,
        global_step=global_step,
        progress_factory=tqdm,
    )

    return TeacherEpochStats(
        loss_total_mean=result.stats.loss.total,
        loss_ce_mean=result.stats.loss.ce,
        grad_norm_mean=result.stats.grad_norm_mean,
        global_step=result.global_step,
        epoch_time_seconds=result.epoch_time_seconds,
    )


def save_teacher_training_result(
    meta: RunMetadata,
    result: TeacherTrainingResult,
    spec: "DatasetSpec",
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

    metadata_path = results_dir("teacher", spec.name) / "run_metadata.json"
    write_run_metadata(meta, metadata_path)
    return best_ckpt_path, metadata_path


def evaluate_saved_teacher(
    cfg: "Config",
    spec: "DatasetSpec",
    *,
    device: str,
) -> TeacherEvaluationResult:
    ckpt_path = teacher_dir(spec.name) / "best.pt"
    metadata_path = results_dir("teacher", spec.name) / "run_metadata.json"
    _require_teacher_artifacts(ckpt_path, metadata_path)

    tokenizer = load_tokenizer(cfg.tokenizer_checkpoint)
    model = load_classifier(cfg.teacher_checkpoint, spec.num_labels, device)
    load_state_dict(model, ckpt_path, device)

    dev_loader, test_loader = _build_eval_loaders(cfg, spec, tokenizer)

    dev_result = evaluate(model, dev_loader, device=device, num_classes=spec.num_labels)
    test_result = evaluate(model, test_loader, device=device, num_classes=spec.num_labels)

    return TeacherEvaluationResult(
        metadata_path=metadata_path,
        dev_size=len(dev_loader.dataset),
        test_size=len(test_loader.dataset),
        dev_result=dev_result,
        test_result=test_result,
        test_metrics=asdict(test_result),
    )


def save_teacher_evaluation_result(result: TeacherEvaluationResult) -> None:
    def mutate(metadata: dict) -> None:
        metadata["dataset"]["splits"]["test"] = result.test_size
        metadata["metrics"] = {
            "dev": asdict(result.dev_result),
            "test": result.test_metrics,
        }

    patch_metadata_file(result.metadata_path, mutate)


def _teacher_batch_loss(
    model: "PreTrainedModel",
    batch: dict[str, torch.Tensor],
    *,
    device: str,
    precision: str,
) -> torch.Tensor:
    with training_autocast(device, precision):
        out = model(**batch)
    if out.loss is None:
        raise RuntimeError("Teacher batch must include labels so the model returns CE loss")
    return out.loss


def _teacher_batch_loss_with_components(
    model: "PreTrainedModel",
    batch: dict[str, torch.Tensor],
    *,
    device: str,
    precision: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss = _teacher_batch_loss(model, batch, device=device, precision=precision)
    return loss, {"ce": loss}


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


def _teacher_epoch_entry(
    stats: TeacherEpochStats,
    dev_result: EvaluationResult,
    epoch: int,
) -> dict:
    return asdict(
        TrainEpochEntry(
            epoch=epoch,
            global_step=stats.global_step,
            epoch_time_seconds=stats.epoch_time_seconds,
            loss_total=stats.loss_total_mean,
            losses={"ce": stats.loss_ce_mean},
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


def _require_teacher_artifacts(ckpt_path: Path, metadata_path: Path) -> None:
    validate_run_artifacts(
        ckpt_path,
        metadata_path,
        regenerate_hint="Run scripts/01_train_teacher.py to regenerate training metadata before evaluation.",
    )
