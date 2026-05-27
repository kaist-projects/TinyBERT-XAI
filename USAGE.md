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

# 3. Install the local package in editable mode
pip install -e .
```

From here on, activate with `conda activate tinybert-xai` at the start of every session.

---

## Smoke test

Loads teacher + student on GPU, runs one forward pass, checks all tensor shapes.

```bash
conda activate tinybert-xai
python scripts/00_smoke_test.py
```

Expected output:
```
[OK] Smoke test passed.
  teacher params    : 109.5M
  student params    : 14.4M
  logits            : teacher (4, 3)  student (4, 3)
  hidden_states     : teacher ×13 (768-d)  student ×5 (312-d)
  attentions        : teacher ×12  student ×4
  peak VRAM         : 0.59 GB
```

First run downloads `bert-base-uncased` (~440 MB), `TinyBERT_General_4L_312D` (~55 MB),
and `cardiffnlp/tweet_eval` (~2 MB) into the HuggingFace cache. Subsequent runs take ~10 seconds.

---

## Interactive use

The entire public API is one import:

```python
from tinybert_xai import KDPair
```

### Load the pair for a dataset

```python
pair = KDPair.for_dataset("tweet_eval/sentiment")

print(pair.num_labels)    # 3
print(pair.label_names)   # ['negative', 'neutral', 'positive']
print(pair.device)        # 'cuda'
```

### Get a batch

```python
batch = pair.sample_batch(n=4)
# batch is a BatchEncoding already on device
# keys: input_ids, attention_mask, token_type_ids, labels
print(batch["input_ids"].shape)   # torch.Size([4, 128])
print(batch["labels"])            # tensor([...])  — 0=neg, 1=neu, 2=pos
```

### Run a forward pass

```python
out = pair.forward(batch)

# Logits
print(out.teacher.logits.shape)   # torch.Size([4, 3])
print(out.student.logits.shape)   # torch.Size([4, 3])

# Hidden states: embedding + one per transformer layer
print(len(out.teacher.hidden_states))          # 13  (1 + 12 layers)
print(len(out.student.hidden_states))          # 5   (1 + 4 layers)
print(out.teacher.hidden_states[0].shape)      # torch.Size([4, 128, 768])
print(out.student.hidden_states[0].shape)      # torch.Size([4, 128, 312])

# Attention probabilities: one per transformer layer
print(len(out.teacher.attentions))             # 12
print(len(out.student.attentions))             # 4
print(out.teacher.attentions[0].shape)         # torch.Size([4, 12, 128, 128])

# Pretty-print summary
print(out.summary())

# Assert all shapes are consistent (used by smoke test)
out.assert_shapes_consistent()
```

### Override defaults (advanced)

```python
pair = KDPair.for_dataset(
    "tweet_eval/sentiment",
    device="cpu",     # force CPU
    seed=0,           # override seed (design doc mandates 42 for experiments)
)
```

---

## What's next

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
