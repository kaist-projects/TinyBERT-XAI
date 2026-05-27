# TinyBERT-XAI — Iteration Results

Each entry documents what *shipped* at the end of an iteration. Per-iteration plan docs in `docs/notes/0X-*-plan.md` record intent at planning time; this file records outcome. They will sometimes diverge — that divergence is useful information.

---

## Iteration 0 — Foundation & smoke test (2026-05-28)

**Goal.** A working conda env + `src/` skeleton that loads `bert-base-uncased`, `huawei-noah/TinyBERT_General_4L_312D`, and TweetEval-sentiment on GPU; runs a teacher+student forward pass; asserts output shapes match design-doc expectations.

**Deliverables.**
- `environment.yml`, `pyproject.toml` (src-layout package)
- `src/tinybert_xai/` — 5 modules:
  - `config.py` — `Config` frozen dataclass (seed, device, checkpoints, max_seq_length)
  - `datasets.py` — `DatasetSpec`, `SentimentLabel` (IntEnum), `DATASET_TWEETEVAL_SENTIMENT` registry constant, `load_split`, `encode_batch`
  - `models.py` — `load_tokenizer`, `load_classifier`
  - `kdpair.py` — `KDPair`, `KDOutputs`
  - `utils.py` — `set_seed`, `get_device`, `count_params`
- `scripts/00_smoke_test.py` — explicit DI wiring + module-level `assert_shapes_consistent`

**Public API (end of iter-0).**
- `Config(seed, device, max_seq_length, teacher_checkpoint, student_checkpoint, tokenizer_checkpoint)` — frozen dataclass, all defaults set
- `DatasetSpec(hf_path, hf_config, num_labels)` — frozen dataclass
- `SentimentLabel` — `IntEnum` with `NEGATIVE=0 / NEUTRAL=1 / POSITIVE=2`
- `DATASET_TWEETEVAL_SENTIMENT` — `DatasetSpec` registry constant
- `load_split(spec, split) -> Dataset` — downloads + returns one HF split
- `encode_batch(tokenizer, ds, *, max_length, device=None) -> BatchEncoding` — tokenizes a HF Dataset slice; adds `labels` tensor
- `load_tokenizer(checkpoint) -> PreTrainedTokenizerBase`
- `load_classifier(checkpoint, num_labels, device) -> PreTrainedModel` — loads `AutoModelForSequenceClassification` with `output_hidden_states=True`, `output_attentions=True`, `attn_implementation="eager"`, moved to device, set to `eval()`
- `KDPair(teacher, student)` — frozen dataclass; `.forward(batch, *, train_mode=False) -> KDOutputs`
- `KDOutputs(teacher, student, num_labels, batch_size, seq_len)` — data carrier; `.summary() -> str`
- `set_seed(seed)`, `get_device() -> str`, `count_params(model) -> int`

**Verification.**
```bash
conda activate tinybert-xai
python scripts/00_smoke_test.py
```
Expected output: `[OK] Smoke test passed.` with logits `(16, 3)`, teacher hidden_states ×13 (768-d), student hidden_states ×5 (312-d), teacher attentions ×12 `(16, 12, 128, 128)`, student attentions ×4 `(16, 12, 128, 128)`, teacher ≈ 110M params, student ≈ 14.5M, peak VRAM line printed.

**Notable departures from `docs/notes/04-iter0-plan.md`.**
- `DatasetLoader` class → `load_split` free function (HF caches Arrow on disk; the class's caching of `splits` between calls had negligible value at one call per split)
- `TweetEvalSentimentBatchEncoder` class → `encode_batch` free function (was a 4-line tokenizer wrapper; no state worth encapsulating)
- `TweetEvalSentimentData` dataclass + `from_row` → dropped entirely (HF Dataset is Arrow-backed; it cannot yield custom class instances; `tokenizer(ds["text"])` and `ds["label"]` are direct and idiomatic)
- `DatasetSpec.data_cls` field → dropped with the above
- `KDOutputs.assert_shapes_consistent()` method → module-level function in smoke script (smoke-test-only logic belongs in the smoke test, not in a production dataclass)
- Smoke-test batch size: 4 → 16 (matches design-doc global batch=16; makes peak VRAM reading meaningful)

**Hands off to iter-1.**
- Reuse as-is: `Config`, `load_tokenizer`, `load_classifier`, `load_split`, `encode_batch`, `KDPair`, `KDOutputs`, `set_seed`, `get_device`, `count_params`
- Iter-1 needs to add: `train_teacher.py`, `eval.py` (macro-F1 / micro-F1 / per-class / confusion-matrix), `metrics.json` per design-doc §6 schema, checkpoint save/load, early-stopping on dev macro-F1
- Open: TweetEval-sentiment has `train` / `validation` / `test` splits — iter-1 plan should confirm "dev" = `validation`

---
