"""TinyBERT-XAI — main entry point.

Run with: python main.py

Shows the current working state of the project.
Each iteration adds to what this script demonstrates.
"""

import torch

from config import Config
from data import NUM_LABELS_TWEETEVAL_SENTIMENT, load_tweeteval_sentiment_batch
from models import load_student_for_classification, load_teacher_for_classification
from utils import get_device, set_seed

# ──────────────────────────────────────────────────────────────────────────────
# ITERATION 0 — Foundation
# What works: load teacher + student on GPU, inspect hidden states + attentions.
# Nothing is trained yet. This demonstrates the shape of the data the KD losses
# in iters 3–5 will consume.
# ──────────────────────────────────────────────────────────────────────────────


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _sep(title: str = "") -> None:
    width = 60
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'─' * pad} {title} {'─' * pad}")
    else:
        print(f"\n{'─' * width}")


def main() -> None:
    set_seed(42)
    device = get_device()
    cfg = Config()

    _sep("Config")
    print(f"  seed             : {cfg.seed}")
    print(f"  device           : {device}")
    print(f"  max_seq_length   : {cfg.max_seq_length}")
    print(f"  teacher          : {cfg.teacher_model_name}")
    print(f"  student          : {cfg.student_model_name}")
    print(f"  pilot dataset    : {cfg.pilot_dataset} [{cfg.pilot_dataset_config}]")

    _sep("Loading models")
    print("  Loading teacher …")
    teacher, tok = load_teacher_for_classification(
        cfg.teacher_model_name,
        num_labels=NUM_LABELS_TWEETEVAL_SENTIMENT,
        device=device,
    )
    print(f"  Teacher loaded   : {_count_params(teacher)/1e6:.1f}M params on {device}")

    print("  Loading student …")
    student, _ = load_student_for_classification(
        cfg.student_model_name,
        num_labels=NUM_LABELS_TWEETEVAL_SENTIMENT,
        device=device,
    )
    print(f"  Student loaded   : {_count_params(student)/1e6:.1f}M params on {device}")

    _sep("Loading pilot batch")
    batch = load_tweeteval_sentiment_batch(
        tok, batch_size=4, max_length=cfg.max_seq_length
    )
    print(f"  Dataset          : {cfg.pilot_dataset} [{cfg.pilot_dataset_config}]")
    print(f"  Batch keys       : {list(batch.keys())}")
    print(f"  input_ids shape  : {tuple(batch['input_ids'].shape)}")
    print(f"  labels           : {batch['labels'].tolist()}  (0=neg, 1=neu, 2=pos)")
    batch = {k: v.to(device) for k, v in batch.items()}

    _sep("Forward pass (no_grad — nothing is trained yet)")
    with torch.no_grad():
        t_out = teacher(**batch)
        s_out = student(**batch)

    print("\n  Teacher outputs:")
    print(f"    logits shape           : {tuple(t_out.logits.shape)}")
    print(f"    hidden_states count    : {len(t_out.hidden_states)}  (embedding + 12 layers)")
    print(f"    hidden_states[0] shape : {tuple(t_out.hidden_states[0].shape)}  (embedding, 768-d)")
    print(f"    hidden_states[-1] shape: {tuple(t_out.hidden_states[-1].shape)}  (last layer, 768-d)")
    print(f"    attentions count       : {len(t_out.attentions)}  (one per transformer layer)")
    print(f"    attentions[0] shape    : {tuple(t_out.attentions[0].shape)}  (batch, heads, seq, seq)")

    print("\n  Student outputs:")
    print(f"    logits shape           : {tuple(s_out.logits.shape)}")
    print(f"    hidden_states count    : {len(s_out.hidden_states)}  (embedding + 4 layers)")
    print(f"    hidden_states[0] shape : {tuple(s_out.hidden_states[0].shape)}  (embedding, 312-d)")
    print(f"    hidden_states[-1] shape: {tuple(s_out.hidden_states[-1].shape)}  (last layer, 312-d)")
    print(f"    attentions count       : {len(s_out.attentions)}  (one per transformer layer)")
    print(f"    attentions[0] shape    : {tuple(s_out.attentions[0].shape)}  (batch, heads, seq, seq)")

    _sep("Why the shapes matter for KD")
    print("  iter 3 — Logit KD  : teacher.logits vs student.logits → KL divergence")
    print("  iter 4 — Hidden KD : teacher.hidden_states[{3,6,9,12}] vs")
    print("                        student.hidden_states[{1,2,3,4}]")
    print("                        (needs 312→768 projection per mapped layer)")
    print("  iter 5 — Attn KD   : teacher.attentions[{2,5,8,11}] vs")
    print("                        student.attentions[{0,1,2,3}]")
    print("                        (post-softmax probabilities, MSE)")

    _sep("VRAM")
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Peak VRAM used     : {peak_gb:.2f} GB  (budget: ~24 GB on RTX 3090)")

    _sep()
    print("  Iteration 0 complete. Next: iter 1 — fine-tune teacher on TweetEval-sentiment.")
    print()


if __name__ == "__main__":
    main()
