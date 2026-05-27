# Iteration 0 — Foundation & Smoke Test (Design)

**Project:** TinyBERT-XAI — Multi-level KD factorial ablation
**Iteration:** 0 / 9 (Day 1 of 14)
**Status:** Design — pending implementation plan
**Date:** 2026-05-27

## Purpose

A working conda environment plus a minimal `src/` skeleton that proves the three load-bearing pieces — teacher (`bert-base-uncased`), student (`huawei-noah/TinyBERT_General_4L_312D`), and pilot dataset (`cardiffnlp/tweet_eval` config `sentiment`) — can co-exist on the user's RTX 3090 with the HuggingFace API we'll rely on for iterations 1–9.

This iteration does NOT train anything, does NOT compute any KD loss, and does NOT touch the 8-condition factorial. Its only job is to remove environment risk before iter 1.

## Scope (from `docs/notes/03-roadmap.md` Iteration 0)

**Deliverables**
- `environment.yml` — conda env, name `tinybert-xai`
- `pyproject.toml` — makes `src/` modules installable via `pip install -e .`
- `src/` with `config.py`, `models.py`, `data.py`, `utils.py`
- `scripts/00_smoke_test.py`
- Empty placeholder dirs: `configs/`, `tests/` (lazy-created: `checkpoints/`, `results/`)

**Concept learned (per roadmap)**
- BERT-family input shape (`input_ids`, `attention_mask`, `token_type_ids`)
- HF `output_hidden_states=True` / `output_attentions=True` — *the* mechanism the whole project depends on
- Teacher (12 layers, 768d) vs student (4 layers, 312d) shape mismatch — *why* projections are needed in iter 4

## Architectural decisions for iter 0

These were settled during brainstorming. They are settled — implementation should follow them, not re-litigate them.

| Decision | Choice | Rationale |
|---|---|---|
| Package layout | Flat `src/` as source root, modules become top-level (`import models`) | User preference; avoids nested `src/tinybert_xai/` while staying installable |
| Install model | `pip install -e .` once after env create | No per-script `sys.path` boilerplate; modules importable from any cwd |
| Env tool | `conda` (already uses the `libmamba` solver by default since conda 23.10) | Conventional and familiar; mamba available via miniforge3 if you switch later |
| Conda channels | `conda-forge` only | Avoids mixed-channel solver conflicts |
| ML stack delivery | pip inside `environment.yml` | HF libs ship fastest on PyPI; PyTorch pip wheels bundle CUDA runtime |
| Python version | `3.12.*` | 3.13 still has spotty HF/PyTorch wheel coverage as of early 2026 |
| Numpy | `<2.0` | Some downstream metric backends still trip on numpy 2 |
| HF library upper bounds | Pin minor (`<4.50`, `<4.0`, etc.) | Prevents silent attention-output dtype/shape drift mid-project |
| Smoke-test batch size | 4 (roadmap literal said 1) | Same DoD spirit; exercises batching from day one |
| Determinism flag | `set_seed(42)` only; **no** `torch.use_deterministic_algorithms(True)` yet | That flag's overhead/constraints belong in iter 1 when we actually train |

## File layout (end-state of iter 0)

```
TinyBERT-XAI/
├── environment.yml              # conda env: python + base scientific (mostly pip-driven)
├── pyproject.toml               # py_modules = [config, models, data, utils]; package-dir = {"": "src"}
├── src/
│   ├── config.py                # Config dataclass
│   ├── models.py                # load_teacher_for_classification, load_student_for_classification
│   ├── data.py                  # load_tweeteval_sentiment_batch, NUM_LABELS_TWEETEVAL_SENTIMENT
│   └── utils.py                 # set_seed, get_device
├── scripts/
│   └── 00_smoke_test.py         # end-to-end smoke test
├── configs/                     # empty in iter 0; populated from iter 2 onward
├── checkpoints/                 # gitignored; lazy-created by iter 1
├── results/                     # gitignored; lazy-created by iter 2
└── tests/                       # empty in iter 0; first test arrives in iter 3 with losses.py
```

`.gitignore` already covers `checkpoints/`, `results/`, `data/`, conda envs, HF cache, wandb, etc. — no changes needed there.

## `environment.yml` — pinned versions

```yaml
name: tinybert-xai
channels:
  - conda-forge
dependencies:
  - python=3.12.*
  - pip
  - pip:
      # PyTorch + CUDA 12.4 runtime (bundled)
      - --extra-index-url https://download.pytorch.org/whl/cu124
      - torch==2.5.*
      # HuggingFace stack (upper-bounded to avoid silent attention-API drift)
      - "transformers>=4.46,<4.50"
      - "datasets>=3.0,<4.0"
      - "tokenizers>=0.20,<0.22"
      - "accelerate>=1.0,<2.0"
      - "evaluate>=0.4,<0.5"
      # Scientific (numpy<2 for HF compatibility headroom)
      - "numpy<2.0"
      - scipy
      - scikit-learn
      - pandas
      # Viz
      - matplotlib
      - seaborn
      # Util
      - pyyaml
      - tqdm
      - rich
      # Optional experiment tracking (not required for any DoD)
      - wandb
      # Dev
      - pytest
      - ruff
```

Exact patch versions are whatever the solver picks at env-create time (the minor pins above are the contract).

## `pyproject.toml` — sketch

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "tinybert-xai"
version = "0.0.1"
requires-python = ">=3.12,<3.13"

[tool.setuptools]
package-dir = {"" = "src"}
py-modules = ["config", "models", "data", "utils"]
```

`py-modules` (not `packages`) because `src/` holds loose top-level modules, not a package directory. As new modules land in later iters (`losses.py`, `trainer.py`, etc.), they get appended here.

## Module contents

### `src/config.py`
```python
@dataclass(frozen=True)
class Config:
    seed: int = 42
    device: str = "cuda"
    max_seq_length: int = 128
    teacher_model_name: str = "bert-base-uncased"
    student_model_name: str = "huawei-noah/TinyBERT_General_4L_312D"
    pilot_dataset: str = "cardiffnlp/tweet_eval"
    pilot_dataset_config: str = "sentiment"
```
Frozen to prevent accidental mutation. Per-condition / per-dataset configs come in iters 2 and 7 and will be separate dataclasses layered on top of this.

### `src/models.py`
Two loaders with the same signature shape:
- `load_teacher_for_classification(model_name: str, num_labels: int, device: str) -> tuple[BertForSequenceClassification, BertTokenizerFast]`
- `load_student_for_classification(model_name: str, num_labels: int, device: str) -> tuple[BertForSequenceClassification, BertTokenizerFast]`

Both:
- Build via `AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, output_hidden_states=True, output_attentions=True)`
- Load tokenizer via `AutoTokenizer.from_pretrained('bert-base-uncased')` (design-doc spec: same tokenizer for both)
- Move model to `device`, set `.eval()` (iter 1 explicitly switches student to `.train()`)
- Return `(model, tokenizer)`

### `src/data.py`
```python
NUM_LABELS_TWEETEVAL_SENTIMENT = 3

def load_tweeteval_sentiment_batch(
    tokenizer, batch_size: int = 4, max_length: int = 128, split: str = "train"
) -> dict[str, torch.Tensor]:
    """Return one batch as {input_ids, attention_mask, token_type_ids, labels} on CPU."""
```

The full 9-dataset adapter contract from `docs/notes/02-project-synthesis.md` §4 lands in iter 7 — iter 0 deliberately ignores it.

### `src/utils.py`
- `set_seed(seed: int)` — sets `random.seed`, `numpy.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`
- `get_device() -> str` — `"cuda" if torch.cuda.is_available() else "cpu"`

## `scripts/00_smoke_test.py` — behavior

```
1. set_seed(42); device = get_device(); assert device == "cuda"
2. cfg = Config()
3. teacher, tok = load_teacher_for_classification(cfg.teacher_model_name, num_labels=3, device)
4. student, _   = load_student_for_classification(cfg.student_model_name, num_labels=3, device)
5. batch = load_tweeteval_sentiment_batch(tok, batch_size=4, max_length=cfg.max_seq_length)
6. batch = {k: v.to(device) for k, v in batch.items()}
7. with torch.no_grad():
       t_out = teacher(**batch)  # output_hidden_states/attentions already on via config
       s_out = student(**batch)
8. Assert + print:
    - teacher.logits.shape == [4, 3]
    - student.logits.shape == [4, 3]
    - len(t_out.hidden_states) == 13;  each shape == [4, 128, 768]
    - len(s_out.hidden_states) == 5;   each shape == [4, 128, 312]
    - len(t_out.attentions)    == 12;  each shape == [4, 12, 128, 128]
    - len(s_out.attentions)    == 4;   each shape == [4, 12, 128, 128]
    - teacher param count ≈ 110M; student ≈ 14.5M
9. print f"Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB"
```

The script uses plain `assert` statements with informative messages, then prints a one-line success banner. No pytest dependency for the smoke test — it's a script, not a test.

## Definition of Done

1. `conda env create -f environment.yml` succeeds on a fresh checkout.
2. `conda activate tinybert-xai && pip install -e .` succeeds.
3. `python -c "import torch; assert torch.cuda.is_available() and torch.cuda.get_device_name(0).startswith('NVIDIA GeForce RTX 3090')"` passes.
4. `python -c "import transformers, datasets; print(transformers.__version__, datasets.__version__)"` prints versions matching the env contract.
5. `python scripts/00_smoke_test.py` exits 0 with all assertions passed and a peak-VRAM line printed.
6. No `CUDA OOM`. No import errors. No tokenizer warnings about missing `pad_token` (default `[PAD]` is fine for bert-base-uncased).
7. Iter 0 lands as **one** git commit (per roadmap §"Cross-cutting concerns").

## Non-goals (deferred to later iterations)

| Item | Lands in |
|---|---|
| Training loop, optimizer, AMP | Iter 1 (teacher fine-tune) |
| 9-dataset adapter contract | Iter 7 |
| Per-condition Config dataclass + YAML loader | Iter 2 |
| `losses.py` with CE / logit / hidden / attention KD | Iters 2, 3, 4, 5 |
| `eval.py` with macro/micro F1, calibration, etc. | Iter 1 |
| `run_metadata.json` writer | Iter 1 |
| Hidden projections (4× `nn.Linear(312, 768)`) | Iter 4 |
| `torch.use_deterministic_algorithms(True)` | Iter 1 |
| `torch.cuda.amp` / `autocast(bfloat16)` | Iter 1 |
| Any test (unit or otherwise) | Iter 3 (first loss function) |
| W&B integration | Optional; if used, iter 1 |

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| PyTorch cu124 wheel doesn't load on driver 580.x | Wheel is forward-compatible; verified the driver supports CUDA ≥12.4 |
| `cardiffnlp/tweet_eval` not in HF cache and offline at run-time | Smoke test must run online once to seed the cache; document in iter 1 if we move to offline workflows |
| Solver picks an incompatible `tokenizers` patch | Upper bound `<0.22` should prevent it; if it bites, drop to `0.20.*` exact |
| `huawei-noah/TinyBERT_General_4L_312D` returns only 4 hidden states (no embedding layer) | Verify in the smoke test — if it does, document and adjust iter 4's `student_to_teacher_layer` indexing |
| `BertForSequenceClassification` warns about randomly-initialized classifier head | Expected and benign for both models; smoke test ignores the warning |

## Verification commands (copy/pasteable for the user)

```bash
# Create env
conda env create -f environment.yml
conda activate tinybert-xai

# Install local package
pip install -e .

# Run smoke test
python scripts/00_smoke_test.py
```

Expected final stdout line shape:
```
[OK] Smoke test passed. Teacher 109.5M params, student 14.5M params. Peak VRAM: 1.43 GB.
```

(Numbers approximate; the VRAM figure is the first real datapoint for the CLAUDE.md hardware section.)
