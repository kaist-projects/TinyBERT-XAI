"""Teacher training/evaluation pipeline contracts.

The public functions in this module are medium-level contracts used by the
scripts. Private helpers carry the lower-level tensor and metadata mechanics.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from tqdm.auto import tqdm
from transformers import set_seed as hf_set_seed

from tinybert_xai.checkpoints import load_state_dict, results_dir, save_state_dict, teacher_dir
from tinybert_xai.datasets import build_loader
from tinybert_xai.earlystop import EarlyStopper
from tinybert_xai.eval import EfficiencyMetrics, EvaluationResult, compute_efficiency, evaluate
from tinybert_xai.models import load_classifier, load_tokenizer
from tinybert_xai.runlog import (
    RunMetadata,
    TrainEpochEntry,
    collect_hardware,
    collect_package_versions,
    make_run_id,
    write_run_metadata,
)
from tinybert_xai.utils import clone_state_dict_cpu, move_batch_to_device

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from tinybert_xai.config import Config
    from tinybert_xai.datasets import DatasetSpec


@dataclass(frozen=True)
class TeacherData:
    tokenizer: "PreTrainedTokenizerBase"
    train_loader: "DataLoader"
    dev_loader: "DataLoader"
    train_size: int
    dev_size: int


@dataclass(frozen=True)
class TeacherModel:
    model: "PreTrainedModel"
    optimizer: torch.optim.Optimizer


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
    best_dev_macro_f1: float
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
    efficiency: EfficiencyMetrics


def configure_reproducibility(seed: int) -> None:
    # Required for deterministic matmul on CUDA >= 10.2; cuBLAS reads it on
    # first call, so set before any model forward.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    hf_set_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(cfg: "Config") -> str:
    return cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")


def start_teacher_metadata(cfg: "Config", spec: "DatasetSpec", device: str) -> RunMetadata:
    return RunMetadata(
        run_id=make_run_id("teacher", spec.name),
        stage="teacher",
        dataset=f"{spec.hf_path}:{spec.hf_config}",
        dataset_family="sentiment",
        condition=None,
        seed=cfg.seed,
        config=asdict(cfg),
        package_versions=collect_package_versions(),
        hardware=collect_hardware(device),
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
    return TeacherModel(model=model, optimizer=optimizer)


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
        )
        global_step = epoch_stats.global_step

        dev_result = evaluate(
            teacher.model,
            data.dev_loader,
            device=device,
            num_classes=spec.num_labels,
        )
        history.append(_teacher_epoch_entry(epoch_stats, dev_result, cfg, epoch))

        print(
            f"  epoch {epoch}  "
            f"train_loss={epoch_stats.loss_total_mean:.4f}  "
            f"dev_macro_f1={dev_result.macro_f1:.4f}  "
            f"dev_acc={dev_result.accuracy:.4f}  "
            f"({epoch_stats.epoch_time_seconds:.1f}s)"
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
        best_dev_macro_f1=stopper.best_value,
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
) -> TeacherEpochStats:
    epoch_start = time.perf_counter()
    model.train()

    generator = getattr(loader, "generator", None)
    if generator is not None:
        generator.manual_seed(seed + epoch)

    loss_total_sum = 0.0
    loss_ce_sum = 0.0
    grad_norm_sum = 0.0
    n_batches = 0

    pbar = tqdm(loader, total=len(loader), desc=f"epoch {epoch}", unit="batch")
    for batch in pbar:
        batch = move_batch_to_device(batch, device)
        loss = _teacher_batch_loss(model, batch)

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
        loss_ce_sum += loss.item()
        grad_norm_sum += grad_norm.item()
        n_batches += 1
        global_step += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return TeacherEpochStats(
        loss_total_mean=loss_total_sum / max(n_batches, 1),
        loss_ce_mean=loss_ce_sum / max(n_batches, 1),
        grad_norm_mean=grad_norm_sum / max(n_batches, 1),
        global_step=global_step,
        epoch_time_seconds=time.perf_counter() - epoch_start,
    )


def save_teacher_training_result(
    meta: RunMetadata,
    result: TeacherTrainingResult,
    spec: "DatasetSpec",
) -> tuple[Path, Path]:
    best_ckpt_path = result.checkpoint_dir / "best.pt"
    best_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result.best_state, best_ckpt_path)

    meta.training = {
        "epochs_completed": len(result.history),
        "best_epoch": result.best_epoch,
        "best_dev_macro_f1": result.best_dev_macro_f1,
        "train_time_seconds": result.train_time_seconds,
        "early_stopped": result.early_stopped,
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

    return TeacherEvaluationResult(
        metadata_path=metadata_path,
        dev_size=len(dev_loader.dataset),
        test_size=len(test_loader.dataset),
        dev_result=dev_result,
        test_result=test_result,
        test_metrics=_teacher_test_metrics(test_result),
        efficiency=efficiency,
    )


def save_teacher_evaluation_result(result: TeacherEvaluationResult) -> None:
    with open(result.metadata_path) as f:
        metadata = json.load(f)

    metadata["splits"]["test"] = result.test_size
    metadata["dev_metrics"] = asdict(result.dev_result)
    metadata["test_metrics"] = result.test_metrics
    metadata["efficiency"] = asdict(result.efficiency)

    with open(result.metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def _teacher_batch_loss(
    model: "PreTrainedModel",
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    out = model(**batch)
    if out.loss is None:
        raise RuntimeError("Teacher batch must include labels so the model returns CE loss")
    return out.loss


def _teacher_epoch_entry(
    stats: TeacherEpochStats,
    dev_result: EvaluationResult,
    cfg: "Config",
    epoch: int,
) -> dict:
    return asdict(
        TrainEpochEntry(
            epoch=epoch,
            train_loss_total=stats.loss_total_mean,
            train_loss_ce=stats.loss_ce_mean,
            train_raw_loss_ce=stats.loss_ce_mean,
            train_loss_logit=None,
            train_raw_loss_logit=None,
            train_loss_hidden=None,
            train_raw_loss_hidden=None,
            train_loss_attention=None,
            train_raw_loss_attention=None,
            grad_norm_mean=stats.grad_norm_mean,
            learning_rate=cfg.learning_rate,
            global_step=stats.global_step,
            epoch_time_seconds=stats.epoch_time_seconds,
            dev_macro_f1=dev_result.macro_f1,
            dev_micro_f1=dev_result.micro_f1,
            dev_accuracy=dev_result.accuracy,
            dev_ECE=dev_result.ECE,
            dev_NLL=dev_result.NLL,
            dev_Brier=dev_result.Brier,
        )
    )


def _teacher_test_metrics(test_result: EvaluationResult) -> dict:
    test_metrics = asdict(test_result)
    test_metrics["top1_agreement"] = None
    test_metrics["teacher_student_kl"] = None
    test_metrics["teacher_correct_student_wrong"] = None
    test_metrics["teacher_wrong_student_correct"] = None
    test_metrics["error_copying"] = None
    return test_metrics


def _require_teacher_artifacts(ckpt_path: Path, metadata_path: Path) -> None:
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Run scripts/01_train_teacher.py first."
        )
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"run_metadata.json not found: {metadata_path}\n"
            "Run scripts/01_train_teacher.py first."
        )
