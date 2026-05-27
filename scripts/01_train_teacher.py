"""scripts/01_train_teacher.py — Teacher fine-tuning on TweetEval-sentiment.

Iteration 1 deliverable.  Trains bert-base-uncased on TweetEval-sentiment,
applies early stopping on dev macro-F1 (patience=2), saves one checkpoint per
epoch plus best.pt, and writes results/teachers/tweet_eval-sentiment/run_metadata.json.

Usage
-----
    conda activate tinybert-xai
    python scripts/01_train_teacher.py

Output
------
    checkpoints/teachers/tweet_eval-sentiment/
        epoch_0.pt, epoch_1.pt, ..., best.pt
    results/teachers/tweet_eval-sentiment/
        run_metadata.json
"""

from __future__ import annotations

import datetime
import json
import os
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

from tinybert_xai import (
    DATASET_TWEETEVAL_SENTIMENT,
    Config,
    evaluate,
    get_device,
    load_classifier,
    load_split,
    load_tokenizer,
    set_seed,
)
from tinybert_xai.datasets import encode_batch

# ── determinism ──────────────────────────────────────────────────────────────
torch.use_deterministic_algorithms(True, warn_only=True)


# ── helpers ──────────────────────────────────────────────────────────────────

def _package_versions() -> dict:
    import transformers
    import datasets as hf_datasets
    import sklearn
    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": hf_datasets.__version__,
        "sklearn": sklearn.__version__,
        "python": platform.python_version(),
    }


def _gpu_model(device: str) -> str | None:
    if device.startswith("cuda"):
        return torch.cuda.get_device_name(device)
    return None


def _save_checkpoint(state_dict: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, path)


def _results_dir(spec_name: str) -> Path:
    return Path("results") / "teachers" / spec_name


def _checkpoint_dir(spec_name: str) -> Path:
    return Path("checkpoints") / "teachers" / spec_name


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = Config()
    spec = DATASET_TWEETEVAL_SENTIMENT
    spec_name = "tweet_eval-sentiment"

    set_seed(cfg.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = cfg.device or get_device()
    print(f"Device: {device}")

    # ── run metadata skeleton ────────────────────────────────────────────────
    run_id = f"teacher-{spec_name}-{datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}"
    metadata: dict = {
        "run_id": run_id,
        "stage": "teacher",
        "dataset": f"{spec.hf_path}:{spec.hf_config}",
        "dataset_family": "sentiment",
        "condition": None,
        "seed": cfg.seed,
        "config": asdict(cfg),
        "package_versions": _package_versions(),
        "hardware": {
            "gpu_model": _gpu_model(device),
            "gpu_memory_total_mb": (
                torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
                if device.startswith("cuda") else None
            ),
        },
    }

    # ── load data ────────────────────────────────────────────────────────────
    print("Loading datasets …")
    train_ds = load_split(spec, "train")
    dev_ds   = load_split(spec, "validation")

    metadata["splits"] = {
        "train": len(train_ds),
        "validation": len(dev_ds),
    }
    print(f"  train={len(train_ds)}  dev={len(dev_ds)}")

    # ── load model ───────────────────────────────────────────────────────────
    tokenizer = load_tokenizer(cfg.tokenizer_checkpoint)
    model = load_classifier(cfg.teacher_checkpoint, spec.num_labels, device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    # ── training loop ────────────────────────────────────────────────────────
    best_f1:   float = -1.0
    best_state: dict | None = None
    best_epoch: int = -1
    no_improve: int = 0
    history: list[dict] = []
    early_stopped = False

    ckpt_dir = _checkpoint_dir(spec_name)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    total_train_start = time.perf_counter()

    for epoch in range(cfg.num_epochs):
        epoch_start = time.perf_counter()
        model.train()

        shuffled = train_ds.shuffle(seed=cfg.seed + epoch)
        loss_total_sum = 0.0
        loss_ce_sum    = 0.0
        grad_norm_sum  = 0.0
        n_batches      = 0
        global_step    = epoch * (len(shuffled) // cfg.train_batch_size)

        for i in range(0, len(shuffled), cfg.train_batch_size):
            raw = shuffled.select(range(i, min(i + cfg.train_batch_size, len(shuffled))))
            batch = encode_batch(
                tokenizer, raw, max_length=cfg.max_seq_length, device=device
            )

            out = model(**batch)   # AutoModelForSequenceClassification: loss = CE when labels present
            loss = out.loss

            # guard against nan/inf
            if not torch.isfinite(loss):
                print(f"  [WARN] non-finite loss at epoch {epoch} step {global_step}: {loss.item():.6f} — skipping batch")
                optimizer.zero_grad()
                global_step += 1
                continue

            loss.backward()

            # gradient norm (for logging)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))

            optimizer.step()
            optimizer.zero_grad()

            loss_total_sum += loss.item()
            loss_ce_sum    += loss.item()   # teacher: total loss IS CE loss
            grad_norm_sum  += grad_norm.item()
            n_batches      += 1
            global_step    += 1

        epoch_time = time.perf_counter() - epoch_start

        # ── per-epoch averages ────────────────────────────────────────────
        avg_loss_total = loss_total_sum / max(n_batches, 1)
        avg_loss_ce    = loss_ce_sum    / max(n_batches, 1)
        avg_grad_norm  = grad_norm_sum  / max(n_batches, 1)

        # ── dev eval ─────────────────────────────────────────────────────
        dev_metrics = evaluate(
            model, dev_ds, tokenizer,
            max_length=cfg.max_seq_length,
            device=device,
            batch_size=cfg.eval_batch_size,
            num_classes=spec.num_labels,
        )

        epoch_entry = {
            "epoch": epoch,
            # §6 loss fields (teacher: KD losses are null)
            "train_loss_total":       avg_loss_total,
            "train_loss_ce":          avg_loss_ce,
            "train_raw_loss_ce":      avg_loss_ce,
            "train_loss_logit":       None,
            "train_raw_loss_logit":   None,
            "train_loss_hidden":      None,
            "train_raw_loss_hidden":  None,
            "train_loss_attention":   None,
            "train_raw_loss_attention": None,
            "grad_norm_mean":         avg_grad_norm,
            "learning_rate":          cfg.learning_rate,
            "global_step":            global_step,
            "epoch_time_seconds":     epoch_time,
            # dev metrics (primary only; full metrics in dev_metrics)
            "dev_macro_f1":  dev_metrics["macro_f1"],
            "dev_micro_f1":  dev_metrics["micro_f1"],
            "dev_accuracy":  dev_metrics["accuracy"],
            "dev_ECE":       dev_metrics["ECE"],
            "dev_NLL":       dev_metrics["NLL"],
            "dev_Brier":     dev_metrics["Brier"],
        }
        history.append(epoch_entry)

        print(
            f"  epoch {epoch}  "
            f"train_loss={avg_loss_total:.4f}  "
            f"dev_macro_f1={dev_metrics['macro_f1']:.4f}  "
            f"dev_acc={dev_metrics['accuracy']:.4f}  "
            f"({epoch_time:.1f}s)"
        )

        # ── save per-epoch checkpoint ─────────────────────────────────────
        epoch_ckpt_path = ckpt_dir / f"epoch_{epoch}.pt"
        _save_checkpoint(
            {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            epoch_ckpt_path,
        )

        # ── early stopping ────────────────────────────────────────────────
        if dev_metrics["macro_f1"] > best_f1:
            best_f1    = dev_metrics["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                early_stopped = True
                print(f"  Early stop triggered after epoch {epoch} (no improvement for {cfg.patience} epochs)")
                break

    total_train_time = time.perf_counter() - total_train_start

    # ── save best checkpoint ─────────────────────────────────────────────────
    assert best_state is not None, "No valid epoch completed — check for NaN losses"
    best_ckpt_path = ckpt_dir / "best.pt"
    _save_checkpoint(best_state, best_ckpt_path)
    print(f"\nBest checkpoint: epoch {best_epoch}, dev macro-F1={best_f1:.4f}")
    print(f"Saved to: {best_ckpt_path}")

    # ── finalise metadata (training section) ─────────────────────────────────
    metadata["training"] = {
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_dev_macro_f1": best_f1,
        "train_time_seconds": total_train_time,
        "early_stopped": early_stopped,
        "history": history,
    }
    # dev_metrics / test_metrics will be filled by 01b_eval_teacher.py
    metadata["dev_metrics"] = None
    metadata["test_metrics"] = None
    metadata["efficiency"] = None

    # ── write run_metadata.json ───────────────────────────────────────────────
    results_dir = _results_dir(spec_name)
    results_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = results_dir / "run_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata written to: {metadata_path}")

    print("\n[OK] Teacher training complete.")


if __name__ == "__main__":
    main()
