# TinyBERT-XAI — Agile Roadmap

## Context

This is a 2-week mini-project for a CS graduate course at KAIST (Team 18, Spring 2026). The research question is fixed by `docs/source/03-design-doc.pdf`: **run an 8-condition factorial ablation of `{Logit, Hidden, Attention}` knowledge distillation across 9 text-classification datasets, with `bert-base-uncased` as teacher and the public `TinyBERT_General_4L_312D` as student.**

Two non-negotiable constraints shape this plan:

1. **Time:** ~14 days to final presentation. Each iteration must deliver something runnable and verifiable. We prefer "full results on a defensible subset" to "partial results on all 9 datasets."
2. **Learning:** the user wants to understand how KD and fine-tuning work internally as we build. Each iteration explicitly introduces *one* new KD concept, in escalating order of structural complexity (logit → hidden → attention).

**Hardware:** RTX 3090/4090-class (24 GB). Comfortable budget.
**Pilot dataset:** TweetEval-sentiment (`cardiffnlp/tweet_eval`, config `sentiment`).
**Environment:** conda env + `requirements.txt` (pip).
**Layout:** root `tinybert_xai/` package + `configs/` + `scripts/`.

The exploration of `reference/` (see Phase 1 findings) showed it is the original 2019/2020 TinyBERT authors' code, depends on a custom pre-modern-HF `transformer/` module, uses pre-softmax attention, a single shared projection, and a sequential phase split — **all of which disagree with our design doc**. We will **not** port `reference/`. We will **lift loss-computation patterns** from `reference/task_distill.py` (`soft_cross_entropy`, MSE-with-mask) and reimplement on modern HuggingFace `transformers`.

The full project context lives in `docs/notes/02-project-synthesis.md` and `CLAUDE.md`. This file is the execution roadmap.

---

## Architectural decisions (decided once, bind the rest)

| Decision | Choice | Rationale |
|---|---|---|
| Model framework | HuggingFace `transformers` 4.40+ | Modern, has `output_hidden_states` / `output_attentions` built-in. The `reference/transformer/` module is obsolete. |
| Trainer | **Custom PyTorch training loop**, not `HF Trainer` | We need fine-grained control over 4 loss terms with per-condition toggles. Trainer's `compute_loss` override gets ugly for 8 conditions. |
| Repo layout & imports | Root `tinybert_xai/` package; no `src/`, no `pip install -e .`; scripts use a `sys.path` bootstrap; deps via `requirements.txt` (pip into a conda env). | Course-budget pragmatism — avoid packaging overhead; keep imports flat. |
| Config format | `Config` dataclass in `tinybert_xai/config.py` + typed `ConditionSpec` instances in `tinybert_xai/conditions.py` (8 specs, `uses_teacher` property). **No YAML.** | One Python source of truth; sweep driver iterates over `ALL_CONDITIONS`. |
| Logging | Plain JSON per run (`run_metadata.json`) + Weights & Biases optional | Schema v2: nested run/dataset/model/optimization/checkpoint_selection/reproducibility/environment/training/metrics blocks; **active-only loss components** (absent when condition disables a signal; no raw/scaled twins); flat 5-dp rounding with an exact-key whitelist (`learning_rate`, `weight_decay`, `eps`). Intentionally supersedes the noisier design doc §6 field list. `training.best_dev_macro_f1` and the `efficiency` block were dropped (architecture-fixed across conditions; redundant with `metrics.dev.macro_f1`). |
| Seed / reproducibility | `seed=42` everywhere, `torch.use_deterministic_algorithms(True)` where possible | Design doc §9 forbids seed variation. |
| Pipeline layering | Scripts orchestrate; `tinybert_xai/teacher.py` and `tinybert_xai/student.py` own pipeline contracts (data, model prep, epoch loop, fit, eval, save); shared low-level helpers (`evaluate`, `EarlyStopper`, `runlog`, `checkpoints`, `move_batch_to_device`, `clone_state_dict_cpu`) stay separate. | Keeps abstraction levels consistent; student.py mirrors teacher.py without inheriting its loss shape. **Revisit (2026-05-28):** with teacher+student both implemented, the "extract only after two real callers" condition in note 06 is met; `teacher.py`/`student.py` are now ~90% duplicated and `docs/plan/abstraction-level-refactoring.md` proposes a phased merge. **Low urgency** — see the cross-cutting note on bounded duplication. |
| Mixed precision | `torch.autocast("cuda", dtype=torch.bfloat16)`, applied consistently to teacher + all 8 student conditions. | RTX 3090+ has bf16; halves activation memory; matches design doc §5's "consistent across the run" rule. |
| Layer mapping | Hard-coded `[3, 6, 9, 12]` for teacher per student layer `[1..4]` | Design doc §4. |
| Hidden projection | **4 separate** trainable `nn.Linear(312, 768)` — one per mapped layer | Design doc §6. Differs from reference (`reference/transformer/modeling.py:1126` has one shared). |
| Attention KD input | **Post-softmax probabilities** | Design doc §6. Differs from reference (`reference/transformer/modeling.py:420` returns pre-softmax). |
| Training mode | **Joint** — all active losses in one backward pass | Design doc §5. No `--pred_distill` two-phase split. |

These decisions are locked. Per-iteration plans implement them; they are not revisited per iteration.

---

## Agile roadmap — 9 iterations across 14 days

Every iteration has the same shape:
- **Goal** — one sentence
- **Deliverable** — what file/artifact exists at the end
- **Concept learned** — the KD/ML idea this iteration teaches
- **Definition of done** — concrete checks that prove it works

Each iteration gets its own detailed sub-plan **at the time we start it**, not now. This file is the roadmap, not the implementation spec.

---

### Iteration 0 — Foundation & smoke test (Day 1)

**Goal:** A working conda environment that can load all three pieces (teacher, student, dataset) on the GPU.

**Deliverable:**
- `requirements.txt` (pip dependencies for a Python 3.12 conda env)
- `tinybert_xai/` package skeleton: `models.py`, `datasets.py`, `config.py`, `utils.py`
- `scripts/00_smoke_test.py` — loads `bert-base-uncased`, `huawei-noah/TinyBERT_General_4L_312D`, and one batch of TweetEval-sentiment; runs one forward pass through each on GPU; prints shapes.

**Concept learned:**
- BERT-family input shape (`input_ids`, `attention_mask`, `token_type_ids`).
- The HF `output_hidden_states=True` / `output_attentions=True` flags — *the* mechanism this whole project depends on.
- Teacher (12 layers, 768d) vs student (4 layers, 312d) shape mismatch — this is *why* the project needs projections.

**Definition of done:**
- `python scripts/00_smoke_test.py` prints teacher logits shape `[1, 3]`, student logits shape `[1, 3]`, teacher hidden states list of length 13 (embedding + 12 layers), student hidden states list of length 5 (embedding + 4 layers).
- No `CUDA OOM`. No import errors.

---

### Iteration 1 — Teacher fine-tuning on the pilot dataset (Day 2) — **DONE** (commit `a2c300a`, PR #2)

**Goal:** A frozen, dataset-specific teacher checkpoint for TweetEval-sentiment, plus a working evaluation pipeline.

**Deliverable:**
- `scripts/01_train_teacher.py` — high-level teacher training orchestration.
- `scripts/01b_eval_teacher.py` — high-level saved-teacher evaluation orchestration.
- `tinybert_xai/teacher.py` — teacher pipeline contracts for data, model prep, epoch training, fine-tuning, evaluation, and artifact saving.
- `tinybert_xai/eval/` with `macro_f1`, `micro_f1`, `accuracy`, `per_class_f1`, `confusion_matrix` plus calibration (`ECE`, `NLL`, `Brier`) — the design doc §7 primary + secondary metrics. (Calibration shipped here in iter-1, not iter-8; iter-3 later tightened NLL to `sklearn.log_loss` and fixed ECE final-bin handling.)
- `checkpoints/teachers/tweet_eval-sentiment/` containing the frozen teacher.
- `results/teachers/tweet_eval-sentiment/run_metadata.json`.

**Concept learned:**
- **Plain task fine-tuning** — what BERT does without any distillation. This is the *target* the student will try to approximate.
- Macro-F1 vs micro-F1 — and why class imbalance makes macro the design doc's primary.
- Early stopping mechanics on a dev set.

**Definition of done:**
- Teacher reaches sensible macro-F1 on TweetEval-sentiment dev (the published BERT-base baseline is ~0.66 macro-F1; we should match within a couple points).
- `run_metadata.json` validates against schema v2 and records optimizer, checkpoint selection, reproducibility, environment, split sizes, and metrics.
- Teacher checkpoint loadable in `eval()` mode in a separate process.

---

### Iteration 2 — `ce_only` student baseline (Day 3) — **DONE** (commit `7d816a6`; schema cleanup `29ecc5f`)

**Goal:** Train the student with **CE loss only** — no distillation. This is condition #1 of the 8. Also builds the student pipeline that all 8 conditions will reuse — iter-3→5 only add KD loss terms, not new loops.

**Deliverable:**
- `scripts/02_train_student.py` — orchestration (same shape as `01_train_teacher.py`).
- `scripts/02b_eval_student.py` — test-set evaluation + `run_metadata.json` patching.
- `tinybert_xai/student.py` — student pipeline contracts mirroring `teacher.py` (data, model prep, KD-aware epoch loop, fit, eval, save).
- `tinybert_xai/conditions.py` — `ConditionSpec` dataclass + 8 instances (`CE_ONLY`, `KD_LOGIT`, … `KD_FULL`) + `uses_teacher` property.
- `tinybert_xai/losses.py` — `compute_student_losses(student_out, teacher_out, cond)`: the KD-ready seam, CE-only body. Iter-3→5 add terms without changing the call shape.
- `results/students/tweet_eval-sentiment/ce_only/run_metadata.json` (schema v2).

**Concept learned:**
- The TinyBERT checkpoint *already* has general-distilled knowledge baked in. Even with no task-specific KD, it's not a random init. This iteration measures *that* floor.
- Why we need the public checkpoint vs random init: design doc §9 forbids random-init baseline because the GD checkpoint **is** the baseline.
- Dev-set early-stop loop pattern that will be reused for all 8 conditions.

**Definition of done:**
- `ce_only` finishes 3 epochs (or early-stops) on TweetEval-sentiment.
- Macro-F1 ≥ random-class-prior baseline (a 3-class task: random = 0.33).
- Writes schema-v2 `run_metadata.json` with `run.condition = "ce_only"`, `model.student_checkpoint` set, active-only `losses: {ce}` per epoch.

---

### Iteration 3 — Add `L_logit` (Day 4–5) — **DONE** (PR #4, commit `87b71e2`)

**Goal:** Implement Logit KD and run conditions `kd_logit` (#2).

**Deliverable:**
- `tinybert_xai/losses.py` gains `logit_kd_loss(student_logits, teacher_logits, T=1.0)` returning `T² · KL(softmax(teacher/T) || softmax(student/T))` with teacher logits `.detach()`-ed inside the function. Extend `compute_student_losses`: `if cond.logit: losses["logit"] = logit_kd_loss(...)`.
- `scripts/02_train_student.py` gains the teacher-loading branch: when `cond.uses_teacher`, load `cfg.teacher_checkpoint`'s saved `best.pt` into a frozen `eval()` classifier and pass it to `fine_tune_student(..., teacher_model=...)`. (The `student.py` signature already accepts it.)
- **Teacher-student analysis block** (deferred from iter-2 per `docs/plan/iteration-2-ce-only-student.md` §Decisions): adds `metrics.test.teacher_student_analysis = {top1_agreement, teacher_student_kl, teacher_correct_student_wrong, teacher_wrong_student_correct, error_copying}`. Lands here so all 7 KD conditions inherit it.
- Condition is `KD_LOGIT` from `tinybert_xai/conditions.py` — no YAML.
- `results/students/tweet_eval-sentiment/kd_logit/run_metadata.json`.

**Concept learned:**
- **Hinton-style soft-label distillation**, the classic KD signal. *Why* `T²` is there (gradient scale preservation when T≠1). KL direction matters: `KL(teacher || student)` penalizes the student for placing low probability where the teacher places high.
- "Dark knowledge" — the relative ranking among non-top classes that hard labels throw away.
- Why teacher logits must be `.detach()`-ed.

**Definition of done:**
- `kd_logit` macro-F1 ≥ `ce_only` macro-F1 (almost always; if not, debug — teacher forward is probably broken).
- Per-epoch active-only `losses` dict contains `{ce, logit}` (no `logit` key when condition disables it).
- Sanity check: `top1_agreement` between teacher and student is higher under `kd_logit` than `ce_only`.

**Observed (TweetEval-sentiment):** test macro-F1 **0.663**, top1_agreement **0.80**, teacher_student_kl 0.149, error_copying **0.90** (90% of shared errors are the identical wrong class), teacher_correct_student_wrong 1226 vs teacher_wrong_student_correct 954. **Open verification gap:** the `kd_logit ≥ ce_only` and `top1_agreement(kd_logit) > top1_agreement(ce_only)` comparisons require the planned `02b` `ce_only` backfill (results are gitignored and no `ce_only` `teacher_student_analysis` is committed) — confirm that backfill ran before relying on the comparison.

---

### Iteration 4 — Add `L_hidden` (Day 6–7) — **DONE** (commit `b194a33`, PR pending)

**Goal:** Implement Hidden KD and run conditions `kd_hidden` (#3) and `kd_logit_hidden` (#5).

**Deliverable:**
- `tinybert_xai/models.py` (or a new `tinybert_xai/projections.py` if the projection module grows) gains a `HiddenProjection` module: 4 independent `nn.Linear(312, 768)` modules, indexed by student layer.
- **Wire projections into the optimizer** (the iter-2 review surfaced this): `prepare_student_model` currently builds `AdamW(model.parameters(), ...)`. The projection layers are not part of `model`, so their params won't be optimized as written. `StudentModel` gains a `projections: nn.Module | None` field; `prepare_student_model` passes `list(model.parameters()) + list(projections.parameters())` to `AdamW`.
- `tinybert_xai/losses.py` gains `hidden_kd_loss(h_student_list, h_teacher_list, projections, attention_mask)` — MSE averaged over valid tokens × hidden dim, per mapped layer, then averaged across layers.
- Layer mapping utility `student_to_teacher_layer: {1:3, 2:6, 3:9, 4:12}`.
- Conditions are `KD_HIDDEN`, `KD_LOGIT_HIDDEN` from `conditions.py` — no YAML.
- Results dirs for both conditions.
- Note: `load_classifier` already passes `output_hidden_states=True` (iter-0 carryover) — no model-loading change needed.
- **Decision (resolved):** 4 projected pairs only (student layers 1–4 → teacher 3,6,9,12); the embedding-output hidden state is **not** a 5th pair — consistent with the design doc's "4 projections" and locked `g(m)=3m`. Implemented as `HiddenProjection` in `tinybert_xai/projections.py` (created, not folded into `models.py`).

**Concept learned:**
- **Feature-level distillation** — beyond outputs, match internal representations.
- **Why projection is needed:** dimensional mismatch (312 vs 768). The projection is *learned* during student training; it's not a fixed reduction.
- **Layer mapping** — student layer 1 doesn't learn from teacher layer 1; it learns from a *deeper* teacher layer. Why uniform `g(m) = 3m` works.
- **Token masking in losses** — padding tokens leak garbage into MSE if not masked.

**Definition of done:**
- Each of the 4 projection matrices is in the optimizer's parameter list (verify via `optimizer.param_groups`).
- `loss_hidden` logged per batch.
- `layer_cosine_similarity` (post-projection student vs teacher) is closer to 1 in `kd_hidden` than in `ce_only`.
- Sanity check: total params trained ≈ student params + 4 × 312 × 768 ≈ 14.5M + 0.96M.

**Observed (TweetEval-sentiment, test):** `kd_hidden` macro-F1 **0.647** (acc 0.647, ECE 0.090, top1_agreement 0.78, error_copying 0.89; epoch losses `{ce 0.57, hidden 0.26}`); `kd_logit_hidden` macro-F1 **0.652** (acc 0.654, ECE 0.054, top1_agreement 0.79, error_copying 0.90; losses `{ce 0.55, logit 0.19, hidden 0.25}`). Projections are in the optimizer (`tests/test_student.py` asserts `projection_parameter_count = 961536 = 4×(312×768+768)` in `optimizer.param_groups`); `projection_parameter_count` is recorded in `run_metadata.json`. Note: neither beat `kd_logit` alone (0.663) on this pilot — the ablation reads this at iter-6/8, not now. The `layer_cosine_similarity` check is deferred to iter-8 (per-layer similarity analysis); not computed here.

---

### Iteration 5 — Add `L_attention` (Day 8)

**Goal:** Implement Attention KD and bring all 4 remaining conditions online: `kd_attn` (#4), `kd_logit_attn` (#6), `kd_hidden_attn` (#7), `kd_full` (#8).

**Deliverable:**
- `tinybert_xai/losses.py` gains `attention_kd_loss(a_student_list, a_teacher_list, attention_mask)` — uses **post-softmax** attention probabilities (design doc says so; departs from reference); MSE averaged over valid token-pairs × heads × mapped layers.
- All 4 new conditions wired via `KD_ATTN`, `KD_LOGIT_ATTN`, `KD_HIDDEN_ATTN`, `KD_FULL` in `conditions.py` (already defined since iter-2).
- Note: `load_classifier` already passes `output_attentions=True` + `attn_implementation="eager"` (iter-0 carryover) — no loader change needed.

**Concept learned:**
- **Structural / relational distillation** — match *which tokens attend to which*. Attention heads encode syntactic and coreference patterns (Clark et al. 2019).
- Why both models must have the same number of heads (12 here) — per-head MSE assumes alignment. (The design doc has a fallback "average over heads if mismatch" — we don't need it but the code handles it.)
- The **pre-softmax vs post-softmax** debate — what each signal means, why our design doc chose probabilities. Real consequence: gradient flows differently through `softmax` than through raw logits.

**Definition of done:**
- All 8 conditions defined and each one runs at least one forward+backward step on a tiny dev sample without OOM or NaN.
- `loss_attention` logged per batch.

**Observed (TweetEval-sentiment, test):** the four attention-bearing conditions completed:
`kd_attn` macro-F1 **0.661**, `kd_logit_attn` **0.643**, `kd_hidden_attn`
**0.648**, and `kd_full` **0.655**. Final-epoch `loss_attention` stayed near
**0.0045** across attention conditions, far below CE (~0.55), logit (~0.18), and
hidden (~0.25). This supports the iter-6 finding that post-softmax attention KD
was near-inert in the pilot.

---

### Iteration 6 — Full factorial sweep on the pilot dataset (Day 9)

**Goal:** Run all 8 conditions to completion on TweetEval-sentiment. Validate the analysis pipeline. Decision point before scaling.

**Deliverable:**
- 8 completed student runs for TweetEval-sentiment.
- `scripts/06_analyze_factorial.py` — computes factorial main effects (e.g., main effect of Logit = average across the 4 conditions where Logit=ON minus the 4 where Logit=OFF) and pairwise interactions.
- A bar chart (8 bars, one per condition) and a "main effects + interactions" table.
- A **go/no-go decision** documented: do results look sane enough to scale to the other 8 datasets?

**Concept learned:**
- **Factorial design analysis** — how to read main effects vs. interactions. Why a 2³ factorial is so much more informative than the proposal's 5-condition additive ladder.
- **Marginal contribution** — the difference each signal adds, *averaged over the contexts it appears in*. Not just `kd_full - kd_logit_attn`.

**Definition of done:**
- Plot exists. Numbers exist.
- Main effects table has signs and rough magnitudes for `Logit`, `Hidden`, `Attention`, and the three pairwise interactions.
- We can articulate, in one sentence: *"On TweetEval-sentiment, signal X contributes most; the Y×Z interaction is/isn't significant."*

**Observed (TweetEval-sentiment):** `scripts/06_analyze_factorial.py
tweet_eval-sentiment` passes the pipeline-validity gate: all 8 student
conditions are present and valid, all completed 3 epochs, required metrics and
active losses are finite, KD teacher-student agreement is above random, metric
ranges are valid, and the script writes all expected tables/figures. Verdict:
**GO to iter-7** on pipeline health.

Do not over-read the effect magnitudes from this single seed: student macro-F1
spread is only **0.0198** (`kd_logit` best at **0.6631**, `kd_logit_attn` lowest
at **0.6433**), which is within expected single-seed noise. Factorial estimates
on test macro-F1 are informational only: Logit **-0.00041**, Hidden **-0.00596**,
Attention **-0.00366**, Logit×Hidden **+0.00639**, Logit×Attention **-0.00469**,
Hidden×Attention **+0.00539**, and Logit×Hidden×Attention **+0.00608**.
Attention KD should be fixed before the 9-dataset scale-up because its
post-softmax loss magnitude is near-inert (**mean 0.00453**).

---

### Iteration 7 — Scale to remaining 8 datasets (Day 10–12)

**Goal:** Run teacher fine-tune + 8 student conditions on each of the other 8 datasets. Triage by difficulty.

**Sub-iteration order** (lowest-risk first; descope from the back of this list if time runs out):
1. **IMDB** (sentiment, easy HF load, binary) — Day 10 morning.
2. **Davidson** (hate speech, single text) — Day 10 afternoon.
3. **DynaHate** (hate speech, possibly multi-class) — Day 11 morning.
4. **ANLI** (NLI, sentence pair — *first sentence-pair dataset*, validate `token_type_ids`) — Day 11 afternoon.
5. **HatEval** (hate speech, HF-gated — needs access) — Day 11 evening.
6. **FEVER** (NLI, large + claim/evidence) — Day 12 morning.
7. **Aepli/VarDial** (dialects, custom adapter) — Day 12 afternoon.
8. **Multi-VALUE** (dialects, possibly 50-class — *highest risk*) — Day 12 evening; descope if blocked.

**Deliverable:**
- New `DatasetSpec` entries (and tokenizer-encoding logic if needed) in `tinybert_xai/datasets.py` per new dataset, conforming to the dataset adapter contract (`docs/notes/02-project-synthesis.md` §4). Promote to `tinybert_xai/data/adapters/` if any single adapter outgrows the file.
- 8 × 9 = 72 student `run_metadata.json` files (or fewer if descoped).
- 8 new teacher checkpoints.

**Concept learned:**
- **Dataset adapter discipline** — uniform `{train, dev, test}` interface; explicit label-id mapping; sentence-pair vs single-text handling.
- **Cross-task variance** — first observation of whether per-signal contributions are stable across task families.
- **Practical KD operations** — running the same pipeline at scale, debugging the rare run that diverges.

**Definition of done:**
- All adapters expose the contract.
- All runs that completed have valid `run_metadata.json`.
- Failed/skipped runs are explicitly documented with a reason.

---

### Iteration 8 — Cross-dataset analysis + artifacts (Day 13)

**Goal:** Compute every analysis the design doc §7 lists, across all completed runs. Generate the assets the final presentation will use.

**Deliverable:**
- `results/analysis/`:
  - Per-condition confusion matrices.
  - Calibration: aggregate the already-per-run `ECE`, `NLL`, `Brier` (computed since iter-1) into per-(dataset, condition) tables.
  - Teacher-student analysis: `top1_agreement`, `error_copying`, `teacher_correct_student_wrong`, `teacher_wrong_student_correct`.
  - Layer similarity: `layer_cosine_similarity`, `layer_kl_divergence` per mapped pair, per condition.
  - Attention heatmaps for representative examples (CE-only vs KD; teacher vs student; correct vs wrong).
  - Efficiency *(once, teacher-vs-student, not per condition)*: a single latency/size figure for the writeup. The student architecture is fixed across all 8 conditions, so per-condition efficiency adds no signal to the ablation.
- A **cross-task heatmap**: rows = 9 datasets, columns = 8 conditions, cell = macro-F1 (or Δ from `ce_only`).
- A short written interpretation (3–5 paragraphs) of which signals matter where.

**Concept learned:**
- **Calibration vs accuracy** — a student can match teacher accuracy and still be badly calibrated; ECE catches that.
- **Error-copying** — high error-copying means the student inherited teacher mistakes; low means independent errors. KD often *increases* error-copying.
- **Hidden similarity ≠ output similarity** — hidden KD makes layers look like the teacher, doesn't guarantee final-layer agreement.

**Definition of done:**
- Every artifact the design doc §7 mandates exists.
- The cross-task heatmap renders.
- We can answer RQ1 and RQ2 with a sentence each, citing specific numbers.

---

### Iteration 9 — Presentation prep (Day 14)

**Goal:** Synthesize findings into a final presentation.

**Deliverable:**
- Slide deck content (markdown or PowerPoint outline; the actual deck can be assembled in any tool).
- Narrative flow:
  1. RQ + positioning (1 slide).
  2. Method: fix the framework, ablate the signal (1 slide).
  3. Experimental matrix: 8 × 9 (1 slide).
  4. Headline result: cross-task heatmap (1–2 slides).
  5. Per-family deep-dive: hate, NLI, sentiment, dialects (4 slides).
  6. Factorial main effects + interactions (1 slide).
  7. Calibration / error-copying (1 slide).
  8. Layer-similarity / attention heatmaps (1 slide).
  9. Limitations + future work (1 slide).

**Concept learned:**
- How to compress a factorial result into a defensible 10-slide narrative.

**Definition of done:**
- A peer who hasn't seen this project can read the deck and explain what was done, what was found, and what the limitations are.

---

## Cross-cutting concerns (applied throughout, not a separate step)

- **Run metadata.** Every run writes schema-v2 `run_metadata.json` (`schema_version: "2"`). It records run identity, dataset and label semantics, model/tokenizer checkpoints, optimizer settings, checkpoint-selection policy, reproducibility settings, environment, active-only losses, `grad_norm`, `global_step`, train/eval times, and metrics. Raw loss twins are intentionally omitted because loss weights are fixed at 1.0. `training.best_dev_macro_f1` and the `efficiency` block were dropped (redundant with `metrics.dev.macro_f1`; architecture-fixed across conditions).
- **Checkpoint naming.** `{stage}/{dataset}/{condition}/{epoch}.pt`. Stage ∈ {teacher, student}.
- **Determinism.** `seed=42` everywhere. `torch.use_deterministic_algorithms(True)`. CUBLAS workspace env var if needed.
- **Loss safety.** `nan` / `inf` checks in the train loop; fail fast.
- **Project state isolation.** `.serena/` and `wandb/` (if used) gitignored. Checkpoints out of git.
- **Tests** — minimal: one unit test per loss function (CE, logit-KD, hidden-KD, attn-KD) verifies the formula on a tiny tensor. Not full coverage; just guardrails.
- **Pipeline duplication is bounded, not growing.** `teacher.py`/`student.py` share ~90% structure, but adding datasets (iter-7) and conditions (iter-4–6) *reuses* these pipeline functions rather than copying them — the duplication count stays at 2 regardless of experiment scale. So the abstraction-level refactor (`docs/plan/abstraction-level-refactoring.md`) is **code-health only, with zero experiment-velocity benefit**. Correct sequencing: do it after iter-6 if time allows, or skip under time pressure — never ahead of the experiment matrix.

---

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Multi-VALUE / dialects datasets blocked or weirdly formatted | Medium | Order them last in Iteration 7; descope cleanly if blocked. |
| HatEval HF gating not approved in time | Low–Medium | Apply for access on Day 1. |
| FEVER too large for the 3-epoch budget at full size | Medium | Subsample to ~50K examples *if* needed; document the subsampling. |
| Attention KD post-softmax produces vanishing gradients | Low | If we see this, document and try pre-softmax variant as an explicit comparison (already an interesting subresult). |
| Sentence-pair token_type_ids handling differs across HF model versions | Medium | Test on ANLI early (Day 11) before FEVER. |
| Student loss explodes when all 4 losses are summed (kd_full) | Low | Magnitudes of the 4 losses are not naturally comparable; monitor each active loss component (active-only `losses` dict) and `loss_total`, and document. Per design doc, weights are all 1.0 — no rescaling. |
| Compute schedule slips | Medium | Iteration 7 has an explicit descope order. Day 13 buffer absorbs slips. |

---

## How we'll work each step

The user will run iterations one at a time. **Before each iteration begins,** Claude will:

1. Write a per-iteration sub-plan in plan-mode (research → questions → plan → execute).
2. Confirm the sub-plan with the user.
3. Implement, with frequent checkpoints to verify each piece.

The user has stated they want to **learn KD internals through this project**. So per-iteration sub-plans will include short conceptual explanations and "things to look at while running" — not just code.

---

## Final-state verification

The project is done when:
1. Every dataset that was attempted has 8 condition `run_metadata.json` files (or a documented reason for omission).
2. The cross-task heatmap renders with all attempted (dataset, condition) cells populated.
3. Main effects and interactions are computed per dataset and aggregated across families.
4. The presentation deck content covers RQ1 + RQ2 with cited numbers.
5. `git log` shows incremental commits per iteration so the trajectory is reviewable.

---

## Current status

**Iter 0–6 done.** Iter-0 foundation (smoke test, package skeleton) shipped; iter-1 fine-tuned teacher on TweetEval-sentiment and locked schema v2; iter-2 added the student pipeline (`student.py`, `conditions.py`, `losses.py` seam) and ran `ce_only`. A schema cleanup pass (`29ecc5f`) dropped the `efficiency` block and `training.best_dev_macro_f1`. The `KDPair` iter-0 facade was removed once the student pipeline made it redundant (`48d00c5`). Iter-3 (PR #4, `87b71e2`) added `logit_kd_loss` + the `teacher_student_analysis` block; `kd_logit` reached test macro-F1 **0.663** (≈ teacher level; teacher dev ≈ 0.720) with **error_copying 0.90** and top1_agreement 0.80. It also tightened metrics: NLL now uses `sklearn.log_loss`, and the ECE final-bin handling was fixed. Iter-4 (commit `b194a33`, PR pending) added Hidden KD: `tinybert_xai/projections.py` (`HiddenProjection`, 4× `nn.Linear(312→768)`) wired into the student optimizer, `hidden_kd_loss` (masked MSE per mapped layer), and ran `kd_hidden` (test macro-F1 0.647) + `kd_logit_hidden` (0.652). Resolved decision: 4 projected pairs, no embedding pair.

Iter-5 added Attention KD and completed the remaining four conditions (`kd_attn`,
`kd_logit_attn`, `kd_hidden_attn`, `kd_full`). Iter-6 added the reusable
analysis package (`tinybert_xai/analysis/`), `scripts/06_analyze_factorial.py`,
code-generated `results/analysis/student_ablation_table.md`,
`results/analysis/main_effects_table.md`, and four figures (PNG+SVG) under
`results/analysis/figures/`. The iter-6 gate is **GO to iter-7** because the
pipeline is healthy; the observed ±0.0198 student spread is treated as
single-seed noise, and the per-signal effects are informational only. Caveat:
post-softmax attention KD is near-inert (`loss_attention` mean **0.00453**) and
should be fixed before scaling.

A follow-up abstraction-level review found `teacher.py`/`student.py` are ~90% duplicated and produced `docs/plan/abstraction-level-refactoring.md` (a phased merge plan, **not yet executed** — see the "Pipeline layering" decision note and the cross-cutting note on bounded duplication).

**Next action: Iteration 7 sub-plan.** Scale to the remaining datasets, but fix
the attention-loss signal first or explicitly document that iter-7 continues with
the known near-inert post-softmax attention term.
