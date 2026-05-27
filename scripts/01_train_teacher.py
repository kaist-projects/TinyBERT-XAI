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

import time
from dataclasses import asdict

import torch

from tinybert_xai import (
    DATASET_TWEETEVAL_SENTIMENT,
    Config,
    EarlyStopper,
    RunMetadata,
    clone_state_dict_cpu,
    collect_hardware,
    collect_package_versions,
    encode_batch,
    evaluate,
    get_device,
    iter_batches,
    load_classifier,
    load_tokenizer,
    load_split,
    make_run_id,
    results_dir,
    save_state_dict,
    set_seed,
    teacher_dir,
    write_run_metadata,
)


def main() -> None:
    cfg = Config()
    spec = DATASET_TWEETEVAL_SENTIMENT

    set_seed(cfg.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = cfg.device or get_device()
    print(f"Device: {device}")

    # ── run metadata skeleton ────────────────────────────────────────────────
    meta = RunMetadata(
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

    # ── load data ────────────────────────────────────────────────────────────
    print("Loading datasets …")
    train_ds = load_split(spec, "train")
    dev_ds   = load_split(spec, "validation")

    meta.splits = {"train": len(train_ds), "validation": len(dev_ds)}
    print(f"  train={len(train_ds)}  dev={len(dev_ds)}")

    # ── load model ───────────────────────────────────────────────────────────
    tokenizer = load_tokenizer(cfg.tokenizer_checkpoint)
    model = load_classifier(cfg.teacher_checkpoint, spec.num_labels, device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    # ── training loop ────────────────────────────────────────────────────────
    stopper = EarlyStopper(patience=cfg.patience, mode="max")
    best_state: dict | None = None
    history: list[dict] = []
    early_stopped = False
    global_step = 0

    ckpt_dir = teacher_dir(spec.name)
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

        for chunk in iter_batches(shuffled, cfg.train_batch_size):
            batch = encode_batch(
                tokenizer, chunk, max_length=cfg.max_seq_length, device=device
            )

            out = model(**batch)   # AutoModelForSequenceClassification: loss = CE when labels present
            loss = out.loss

            if not torch.isfinite(loss):
                print(f"  [WARN] non-finite loss at epoch {epoch} step {global_step}: {loss.item():.6f} — skipping batch")
                optimizer.zero_grad()
                continue

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))
            optimizer.step()
            optimizer.zero_grad()

            loss_total_sum += loss.item()
            loss_ce_sum    += loss.item()
            grad_norm_sum  += grad_norm.item()
            n_batches      += 1
            global_step    += 1

        epoch_time = time.perf_counter() - epoch_start

        avg_loss_total = loss_total_sum / max(n_batches, 1)
        avg_loss_ce    = loss_ce_sum    / max(n_batches, 1)
        avg_grad_norm  = grad_norm_sum  / max(n_batches, 1)

        # ── dev eval ─────────────────────────────────────────────────────
        dev_result = evaluate(
            model, dev_ds, tokenizer,
            max_length=cfg.max_seq_length,
            device=device,
            batch_size=cfg.eval_batch_size,
            num_classes=spec.num_labels,
        )

        history.append({
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
            "dev_macro_f1":  dev_result.macro_f1,
            "dev_micro_f1":  dev_result.micro_f1,
            "dev_accuracy":  dev_result.accuracy,
            "dev_ECE":       dev_result.ECE,
            "dev_NLL":       dev_result.NLL,
            "dev_Brier":     dev_result.Brier,
        })

        print(
            f"  epoch {epoch}  "
            f"train_loss={avg_loss_total:.4f}  "
            f"dev_macro_f1={dev_result.macro_f1:.4f}  "
            f"dev_acc={dev_result.accuracy:.4f}  "
            f"({epoch_time:.1f}s)"
        )

        save_state_dict(model, ckpt_dir / f"epoch_{epoch}.pt")

        is_best, should_stop = stopper.update(dev_result.macro_f1, epoch)
        if is_best:
            best_state = clone_state_dict_cpu(model)
        if should_stop:
            early_stopped = True
            print(f"  Early stop triggered after epoch {epoch} (no improvement for {cfg.patience} epochs)")
            break

    total_train_time = time.perf_counter() - total_train_start

    # ── save best checkpoint ─────────────────────────────────────────────────
    if best_state is None:
        raise RuntimeError("No valid epoch completed — check for NaN losses")
    best_ckpt_path = ckpt_dir / "best.pt"
    best_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, best_ckpt_path)
    print(f"\nBest checkpoint: epoch {stopper.best_step}, dev macro-F1={stopper.best_value:.4f}")
    print(f"Saved to: {best_ckpt_path}")

    # ── finalise metadata (training section) ─────────────────────────────────
    meta.training = {
        "epochs_completed": len(history),
        "best_epoch": stopper.best_step,
        "best_dev_macro_f1": stopper.best_value,
        "train_time_seconds": total_train_time,
        "early_stopped": early_stopped,
        "history": history,
    }
    # dev_metrics / test_metrics / efficiency filled by 01b_eval_teacher.py

    metadata_path = results_dir("teacher", spec.name) / "run_metadata.json"
    write_run_metadata(meta, metadata_path)
    print(f"Metadata written to: {metadata_path}")

    print("\n[OK] Teacher training complete.")


if __name__ == "__main__":
    main()
