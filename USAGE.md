# TinyBERT-XAI — Iteration 0 Usage Guide

## Prerequisites

- miniforge3 / conda installed
- NVIDIA GPU with CUDA 12.4-compatible driver (tested on RTX 3090)
- Internet access for first run (model + dataset downloads)

---

## Setup (one-time, fresh checkout)

```bash
# 1. Create the conda environment (~5–15 min, downloads ~2–3 GB)
conda env create -f environment.yml

# 2. Activate it
conda activate tinybert-xai

# 3. Install the local src/ package in editable mode
pip install -e .
```

From here on, activate with `conda activate tinybert-xai` at the start of every session.

---

## Use-case examples

### 1. Smoke test — verify everything works end-to-end

Loads teacher + student on GPU, runs one forward pass, checks all tensor shapes.

```bash
conda activate tinybert-xai
python scripts/00_smoke_test.py
```

Expected output:
```
[OK] Smoke test passed. Teacher 109.5M params, student 14.4M params. Peak VRAM: 0.59 GB.
```

First run downloads `bert-base-uncased` (~440 MB), `TinyBERT_General_4L_312D` (~55 MB),
and `cardiffnlp/tweet_eval` (~2 MB) into the HuggingFace cache. Subsequent runs take ~10 seconds.

---

### 2. Load the teacher model interactively

```python
conda activate tinybert-xai
python

>>> from config import Config
>>> from models import load_teacher_for_classification
>>> from utils import get_device, set_seed

>>> set_seed(42)
>>> device = get_device()
>>> print(device)          # cuda

>>> cfg = Config()
>>> teacher, tok = load_teacher_for_classification(
...     cfg.teacher_model_name, num_labels=3, device=device
... )
>>> print(teacher.__class__.__name__)          # BertForSequenceClassification
>>> print(sum(p.numel() for p in teacher.parameters()) / 1e6)   # ~109.5
```

---

### 3. Load the student model interactively

```python
>>> from models import load_student_for_classification

>>> student, _ = load_student_for_classification(
...     cfg.student_model_name, num_labels=3, device=device
... )
>>> print(sum(p.numel() for p in student.parameters()) / 1e6)   # ~14.4
```

---

### 4. Inspect hidden states and attention shapes

These are the tensors that knowledge distillation losses in iters 3–5 will consume.

```python
>>> import torch
>>> from data import NUM_LABELS_TWEETEVAL_SENTIMENT, load_tweeteval_sentiment_batch

>>> batch = load_tweeteval_sentiment_batch(tok, batch_size=4, max_length=128)
>>> batch = {k: v.to(device) for k, v in batch.items()}

>>> with torch.no_grad():
...     t_out = teacher(**batch)
...     s_out = student(**batch)

# Logits
>>> t_out.logits.shape        # torch.Size([4, 3])
>>> s_out.logits.shape        # torch.Size([4, 3])

# Hidden states: embedding layer + one per transformer layer
>>> len(t_out.hidden_states)  # 13  (1 embedding + 12 layers)
>>> len(s_out.hidden_states)  # 5   (1 embedding + 4 layers)
>>> t_out.hidden_states[0].shape   # torch.Size([4, 128, 768])  — embedding
>>> s_out.hidden_states[0].shape   # torch.Size([4, 128, 312])  — embedding

# Attention probabilities: one per transformer layer
>>> len(t_out.attentions)     # 12
>>> len(s_out.attentions)     # 4
>>> t_out.attentions[0].shape # torch.Size([4, 12, 128, 128])  — [batch, heads, seq, seq]
>>> s_out.attentions[0].shape # torch.Size([4, 12, 128, 128])
```

---

### 5. Fetch a real batch from TweetEval-sentiment

```python
>>> from data import load_tweeteval_sentiment_batch

>>> batch = load_tweeteval_sentiment_batch(tok, batch_size=4, max_length=128, split="train")
>>> list(batch.keys())        # ['input_ids', 'attention_mask', 'token_type_ids', 'labels']
>>> batch["input_ids"].shape  # torch.Size([4, 128])
>>> batch["labels"]           # tensor([...])  — 0=negative, 1=neutral, 2=positive
```

---

### 6. Inspect the global Config

```python
>>> from config import Config
>>> cfg = Config()
>>> cfg.seed                  # 42
>>> cfg.max_seq_length        # 128
>>> cfg.teacher_model_name    # 'bert-base-uncased'
>>> cfg.student_model_name    # 'huawei-noah/TinyBERT_General_4L_312D'
>>> cfg.pilot_dataset         # 'cardiffnlp/tweet_eval'
```

---

## What's next

Iteration 0 is a smoke test only — no training, no losses, no evaluation.

| Iteration | What it adds |
|---|---|
| 1 | Fine-tune teacher on TweetEval-sentiment; add `eval.py` with macro-F1 |
| 2 | Train student with CE loss only (`ce_only` baseline); add config YAML system |
| 3 | Add logit KD loss (`L_logit = T² · KL(teacher ‖ student)`) |
| 4 | Add hidden KD loss with 4 trainable projection layers (312→768) |
| 5 | Add attention KD loss (post-softmax); complete all 8 conditions |
| 6 | Full factorial sweep on TweetEval-sentiment; analyze main effects |
| 7 | Scale to remaining 8 datasets |
| 8 | Cross-dataset analysis + all artifacts |
| 9 | Presentation prep |
