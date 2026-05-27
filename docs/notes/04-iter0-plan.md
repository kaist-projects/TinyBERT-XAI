# Iteration 0 — Foundation & Smoke Test (DI/SRP refresh)

## Context

Iteration 0 of the 9-iteration TinyBERT-XAI roadmap establishes a working conda env plus a minimal `src/` skeleton that proves teacher (`bert-base-uncased`), student (`huawei-noah/TinyBERT_General_4L_312D`), and pilot dataset (`cardiffnlp/tweet_eval[sentiment]`) co-exist on the RTX 3090 with the HuggingFace API iters 1–9 will rely on.

The prior iter-0 (in git HEAD, deleted in working tree) collapsed loading, batching, dataset registry, forward, and factory construction into a single `KDPair` facade with a static `for_dataset` factory. That design works but **violates SRP** (one class owns six responsibilities) and **violates DI** (the factory hides construction, so the teacher/student/tokenizer can't be swapped or stubbed without monkey-patching the registry).

This plan rewrites iter-0 with explicit dependency injection: each loader is a free function taking exactly its arguments, the registry is data, and `KDPair` becomes a slim container that only does what its name says — pair a teacher and student and run them forward together.

Confirmed decisions (from clarifying questions):
- **Start fresh** (HEAD's iter-0 stays in history but is superseded).
- **KDPair = slim container** holding `(teacher, student, tokenizer)`, with exactly one method: `forward(batch) → KDOutputs`.
- **Verification = smoke script only**; `tests/` stays empty until iter-3 introduces `losses.py`.
- **Dataset registry lives in its own module** `src/tinybert_xai/datasets.py`.

## Design principles applied

| Principle | How this plan honors it |
|---|---|
| **Single Responsibility** | Each module owns one concern: config = settings, datasets = registry, models = checkpoint loading, data = tokenized batches, utils = generic helpers, kdpair = paired forward. No module knows two things. |
| **Dependency Injection** | All loaders take their dependencies as explicit args (`checkpoint`, `num_labels`, `device`, `tokenizer`, `spec`). No factory hides construction. No module reaches into a registry or global config implicitly. The smoke test (and later, the trainer) is responsible for wiring. |
| **Pure data vs. behavior** | `Config`, `DatasetSpec`, `KDOutputs` are frozen dataclasses. `KDPair` carries three references and one verb. Registries are plain dicts. |
| **Public API at the boundary** | `__init__.py` re-exports the wiring blocks users need: `Config`, `KDPair`, `KDOutputs`, `load_classifier`, `load_tokenizer`, `load_batch`, `get_dataset_spec`, plus the seed/device helpers. Nothing is name-mangled with `_` — DI requires composable, importable building blocks. |

## Module layout

```
TinyBERT-XAI/
├── environment.yml                  # conda env (pinned versions)
├── pyproject.toml                   # src-layout package
├── src/tinybert_xai/
│   ├── __init__.py                  # public re-exports
│   ├── config.py                    # Config dataclass (project-wide settings)
│   ├── datasets.py                  # DatasetSpec + DATASET_REGISTRY + get_dataset_spec
│   ├── models.py                    # load_tokenizer, load_classifier
│   ├── data.py                      # load_batch
│   ├── utils.py                     # set_seed, get_device, count_params
│   └── kdpair.py                    # KDPair + KDOutputs
├── scripts/
│   └── 00_smoke_test.py             # wires it all up, asserts shapes, prints summary
├── configs/                         # empty placeholder (gitkeep) — populated iter 2+
└── tests/                           # empty placeholder (gitkeep) — populated iter 3+
```

## Module contracts

### `src/tinybert_xai/config.py`
```python
@dataclass(frozen=True)
class Config:
    seed: int = 42
    device: str | None = None         # None → auto-detect via get_device()
    max_seq_length: int = 128
    teacher_checkpoint: str = "bert-base-uncased"
    student_checkpoint: str = "huawei-noah/TinyBERT_General_4L_312D"
    tokenizer_checkpoint: str = "bert-base-uncased"   # design-doc: shared
```
Pure data. No methods. Iter-2's per-condition config and iter-7's per-dataset config layer on top, do not replace, this one.

### `src/tinybert_xai/datasets.py`
```python
@dataclass(frozen=True)
class DatasetSpec:
    hf_path: str
    hf_config: str | None
    label_names: tuple[str, ...]
    text_column: str = "text"
    label_column: str = "label"
    default_split: str = "train"

    @property
    def num_labels(self) -> int:
        return len(self.label_names)

DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "tweet_eval/sentiment": DatasetSpec(
        hf_path="cardiffnlp/tweet_eval",
        hf_config="sentiment",
        label_names=("negative", "neutral", "positive"),
    ),
}

def get_dataset_spec(key: str) -> DatasetSpec:
    """Lookup with helpful error listing available keys."""
```
SRP win: registry doesn't know about models, models don't know about registry. Iter-7 grows this file to 9 entries without touching anything else.

### `src/tinybert_xai/models.py`
Two free functions; no class. Both pure: same inputs → same model.
```python
def load_tokenizer(checkpoint: str) -> PreTrainedTokenizerBase: ...

def load_classifier(
    checkpoint: str,
    num_labels: int,
    device: str,
) -> PreTrainedModel:
    """Load AutoModelForSequenceClassification with output_hidden_states=True,
    output_attentions=True, attn_implementation='eager'. Move to device, set eval()."""
```
DI win: callers pass `num_labels` and `device` explicitly. No reaching into config or registry. Testable with any HuggingFace checkpoint and any device string.

### `src/tinybert_xai/data.py`
One free function:
```python
def load_batch(
    spec: DatasetSpec,
    tokenizer: PreTrainedTokenizerBase,
    *,
    batch_size: int,
    max_length: int,
    split: str | None = None,
    device: str | None = None,
) -> BatchEncoding:
    """Return {input_ids, attention_mask, token_type_ids, labels}, padded to max_length,
    optionally moved to device. Uses spec.text_column / label_column / default_split."""
```
DI win: takes spec + tokenizer as args, doesn't construct them. Sentence-pair handling (iter-7) extends this signature with an optional `text_b_column`-aware path — non-breaking.

### `src/tinybert_xai/utils.py`
```python
def set_seed(seed: int) -> None: ...        # random, numpy, torch CPU + CUDA
def get_device() -> str: ...                # "cuda" if available else "cpu"
def count_params(model: torch.nn.Module) -> int: ...
```
Pure helpers. No imports beyond stdlib + torch + numpy. Unchanged from HEAD.

### `src/tinybert_xai/kdpair.py`
The slim KDPair:
```python
@dataclass(frozen=True)
class KDPair:
    teacher: PreTrainedModel
    student: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase

    def forward(self, batch: BatchEncoding, *, train_mode: bool = False) -> "KDOutputs":
        """Run teacher + student on batch. Wraps in torch.no_grad() when not train_mode."""
```
Plus the result type:
```python
@dataclass
class KDOutputs:
    teacher: Any        # ModelOutput with logits, hidden_states, attentions
    student: Any
    num_labels: int
    batch_size: int
    seq_len: int

    def assert_shapes_consistent(self) -> None: ...   # same checks as HEAD
    def summary(self) -> str: ...
```

What KDPair **does not** do (compared to HEAD):
- No `for_dataset` factory. The smoke test wires components explicitly.
- No `sample_batch`. Call `load_batch(...)` directly.
- No knowledge of the dataset registry, the device, or `max_seq_length`.
- No knowledge of param counts — `count_params` is in `utils.py`; smoke test computes them when summarizing.

DI win: `KDPair(fake_teacher, fake_student, fake_tokenizer)` is one line in a test. Forward exercises both models with the same batch — that's the actual invariant the class encodes.

### `src/tinybert_xai/__init__.py`
```python
from tinybert_xai.config import Config
from tinybert_xai.datasets import DatasetSpec, DATASET_REGISTRY, get_dataset_spec
from tinybert_xai.kdpair import KDPair, KDOutputs
from tinybert_xai.models import load_tokenizer, load_classifier
from tinybert_xai.data import load_batch
from tinybert_xai.utils import set_seed, get_device, count_params

__all__ = [
    "Config", "DatasetSpec", "DATASET_REGISTRY", "get_dataset_spec",
    "KDPair", "KDOutputs",
    "load_tokenizer", "load_classifier", "load_batch",
    "set_seed", "get_device", "count_params",
]
```
Public surface = every building block. DI requires composability; nothing is hidden behind underscores.

## Smoke test wiring

`scripts/00_smoke_test.py`:
```python
"""Iteration 0 smoke test — explicit wiring proves DI/SRP layout works on GPU."""
import torch
from tinybert_xai import (
    Config, get_dataset_spec, set_seed, get_device, count_params,
    load_tokenizer, load_classifier, load_batch, KDPair,
)

def main() -> None:
    cfg = Config()
    set_seed(cfg.seed)
    device = cfg.device or get_device()

    spec = get_dataset_spec("tweet_eval/sentiment")
    tokenizer = load_tokenizer(cfg.tokenizer_checkpoint)
    teacher   = load_classifier(cfg.teacher_checkpoint, spec.num_labels, device)
    student   = load_classifier(cfg.student_checkpoint, spec.num_labels, device)
    pair      = KDPair(teacher, student, tokenizer)

    batch = load_batch(
        spec, tokenizer,
        batch_size=4, max_length=cfg.max_seq_length,
        split=spec.default_split, device=device,
    )

    out = pair.forward(batch)
    out.assert_shapes_consistent()

    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
    print(
        f"[OK] Smoke test passed.\n"
        f"{out.summary()}\n"
        f"  teacher params    : {count_params(teacher) / 1e6:.1f}M\n"
        f"  student params    : {count_params(student) / 1e6:.1f}M\n"
        f"  peak VRAM         : {peak_vram_gb:.2f} GB"
    )

if __name__ == "__main__":
    main()
```

This script is the **demonstration of DI**: every dependency is constructed up-front and passed in. Swapping the teacher for a smaller checkpoint, or the dataset for a different one, is a one-line edit at the wiring layer. No module is touched.

## `environment.yml` & `pyproject.toml`

Restore exactly what was in HEAD (already-validated pins). The current working tree has both deleted; the plan re-introduces them unchanged. No reason to re-litigate dependency pins in iter-0.

`pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "tinybert-xai"
version = "0.0.1"
description = "TinyBERT-XAI — multi-level KD factorial ablation across 9 classification datasets."
requires-python = ">=3.12,<3.13"

[tool.setuptools]
package-dir = {"" = "src"}
packages = ["tinybert_xai"]
```

`environment.yml`: conda-forge + python=3.12.* + pip block with `torch==2.5.*` (CUDA 12.4 wheels), `transformers>=4.46,<4.50`, `datasets>=3.0,<4.0`, `tokenizers>=0.20,<0.22`, `accelerate`, `evaluate`, `numpy<2.0`, scipy/sklearn/pandas, matplotlib/seaborn, pyyaml/tqdm/rich, wandb, pytest, ruff. See HEAD for the canonical pin set.

## Definition of Done

1. `conda env create -f environment.yml` succeeds.
2. `pip install -e .` succeeds inside `tinybert-xai` env.
3. `python -c "from tinybert_xai import KDPair; print(KDPair)"` imports cleanly.
4. `python scripts/00_smoke_test.py` exits 0 with:
   - teacher logits `[4, 3]`; student logits `[4, 3]`
   - teacher `hidden_states` length 13 (embed + 12 layers), each `[4, 128, 768]`
   - student `hidden_states` length 5 (embed + 4 layers), each `[4, 128, 312]`
   - teacher `attentions` length 12, each `[4, 12, 128, 128]`
   - student `attentions` length 4, each `[4, 12, 128, 128]`
   - teacher params ≈ 110M; student ≈ 14.5M
   - peak VRAM line printed; no OOM, no import errors.
5. `ruff check src/` clean.

## Verification (how the user runs it)

```bash
# one-time
conda env create -f environment.yml
conda activate tinybert-xai
pip install -e .

# end-to-end check
python scripts/00_smoke_test.py
```

Expected runtime: ~30–60 s on a warm HF cache, mostly model download time on a cold cache.

## What this plan deliberately does NOT include

- No training, no losses, no KD math (iters 1–5).
- No `tests/` content (iter-3).
- No `configs/` YAMLs (iter-2).
- No per-iteration verification scripts beyond `00_smoke_test.py` — those land iteration-by-iteration as each one starts, per the conversation that prompted this re-planning. Iter-0's verifier IS the smoke test.
- No `torch.use_deterministic_algorithms(True)` — that flag's overhead and constraints belong in iter-1 when training actually starts. `set_seed(42)` is enough for iter-0.
- No abstract `Protocol` / `ABC` interfaces for the loaders. Python function-level DI is sufficient; adding `ClassifierLoader: Protocol` would be over-engineering at iter-0 scale. Iter-3+ can introduce protocols if multiple loss-source implementations appear.
