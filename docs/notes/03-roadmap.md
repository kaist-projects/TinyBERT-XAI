# TinyBERT-XAI — Agile Roadmap

## Context

This is a 2-week mini-project for a CS graduate course at KAIST (Team 18, Spring 2026). The research question is fixed by `docs/source/03-design-doc.pdf`: **run an 8-condition factorial ablation of `{Logit, Hidden, Attention}` knowledge distillation across 9 text-classification datasets, with `bert-base-uncased` as teacher and the public `TinyBERT_General_4L_312D` as student.**

Two non-negotiable constraints shape this plan:

1. **Time:** ~14 days to final presentation. Each iteration must deliver something runnable and verifiable. We prefer "full results on a defensible subset" to "partial results on all 9 datasets."
2. **Learning:** the user wants to understand how KD and fine-tuning work internally as we build. Each iteration explicitly introduces *one* new KD concept, in escalating order of structural complexity (logit → hidden → attention).

**Hardware:** RTX 3090/4090-class (24 GB). Comfortable budget.
**Pilot dataset:** TweetEval-sentiment (`cardiffnlp/tweet_eval`, config `sentiment`).
**Environment:** conda/mamba.
**Layout:** `src/` + `configs/` + `scripts/`.

The exploration of `reference/` (see Phase 1 findings) showed it is the original 2019/2020 TinyBERT authors' code, depends on a custom pre-modern-HF `transformer/` module, uses pre-softmax attention, a single shared projection, and a sequential phase split — **all of which disagree with our design doc**. We will **not** port `reference/`. We will **lift loss-computation patterns** from `reference/task_distill.py` (`soft_cross_entropy`, MSE-with-mask) and reimplement on modern HuggingFace `transformers`.

The full project context lives in `docs/notes/02-project-synthesis.md` and `CLAUDE.md`. This file is the execution roadmap.

---

## Architectural decisions (decided once, bind the rest)

| Decision | Choice | Rationale |
|---|---|---|
| Model framework | HuggingFace `transformers` 4.40+ | Modern, has `output_hidden_states` / `output_attentions` built-in. The `reference/transformer/` module is obsolete. |
| Trainer | **Custom PyTorch training loop**, not `HF Trainer` | We need fine-grained control over 4 loss terms with per-condition toggles. Trainer's `compute_loss` override gets ugly for 8 conditions. |
| Config format | YAML per dataset + YAML per condition; `dataclass` schema in `src/config.py` | One config file per run; sweep driver iterates. |
| Logging | Plain JSON per run (`run_metadata.json`) + Weights & Biases optional | Design doc §6 dictates JSON fields; W&B nice-to-have. |
| Seed / reproducibility | `seed=42` everywhere, `torch.use_deterministic_algorithms(True)` where possible | Design doc §9 forbids seed variation. |
| Mixed precision | `torch.cuda.amp.autocast(bfloat16)` | RTX 3090+ has bf16. Halves activation memory. |
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
- `environment.yml` (conda)
- `src/` skeleton: `models.py`, `data.py`, `config.py`, `utils.py`
- `scripts/00_smoke_test.py` — loads `bert-base-uncased`, `huawei-noah/TinyBERT_General_4L_312D`, and one batch of TweetEval-sentiment; runs one forward pass through each on GPU; prints shapes.

**Concept learned:**
- BERT-family input shape (`input_ids`, `attention_mask`, `token_type_ids`).
- The HF `output_hidden_states=True` / `output_attentions=True` flags — *the* mechanism this whole project depends on.
- Teacher (12 layers, 768d) vs student (4 layers, 312d) shape mismatch — this is *why* the project needs projections.

**Definition of done:**
- `python scripts/00_smoke_test.py` prints teacher logits shape `[1, 3]`, student logits shape `[1, 3]`, teacher hidden states list of length 13 (embedding + 12 layers), student hidden states list of length 5 (embedding + 4 layers).
- No `CUDA OOM`. No import errors.

---

### Iteration 1 — Teacher fine-tuning on the pilot dataset (Day 2)

**Goal:** A frozen, dataset-specific teacher checkpoint for TweetEval-sentiment, plus a working evaluation pipeline.

**Deliverable:**
- `src/train_teacher.py` (or `scripts/01_train_teacher.py`).
- `src/eval.py` with `macro_f1`, `micro_f1`, `accuracy`, `per_class_f1`, `confusion_matrix` — the design doc §7 primary + secondary metrics.
- `checkpoints/teachers/tweet_eval-sentiment/` containing the frozen teacher.
- `results/teachers/tweet_eval-sentiment/metrics.json`.

**Concept learned:**
- **Plain task fine-tuning** — what BERT does without any distillation. This is the *target* the student will try to approximate.
- Macro-F1 vs micro-F1 — and why class imbalance makes macro the design doc's primary.
- Early stopping mechanics on a dev set.

**Definition of done:**
- Teacher reaches sensible macro-F1 on TweetEval-sentiment dev (the published BERT-base baseline is ~0.66 macro-F1; we should match within a couple points).
- `metrics.json` validates against the design doc §6 schema.
- Teacher checkpoint loadable in `eval()` mode in a separate process.

---

### Iteration 2 — `ce_only` student baseline (Day 3)

**Goal:** Train the student with **CE loss only** — no distillation. This is condition #1 of the 8.

**Deliverable:**
- `src/train_student.py` — accepts a config that toggles `{logit, hidden, attention}` independently.
- `src/losses.py` — initially just CE.
- First condition config: `configs/conditions/ce_only.yaml`.
- `results/runs/tweet_eval-sentiment/ce_only/metrics.json`.

**Concept learned:**
- The TinyBERT checkpoint *already* has general-distilled knowledge baked in. Even with no task-specific KD, it's not a random init. This iteration measures *that* floor.
- Why we need the public checkpoint vs random init: design doc §9 forbids random-init baseline because the GD checkpoint **is** the baseline.
- Dev-set early-stop loop pattern that will be reused for all 8 conditions.

**Definition of done:**
- `ce_only` finishes 3 epochs (or early-stops) on TweetEval-sentiment.
- Macro-F1 ≥ random-class-prior baseline (a 3-class task: random = 0.33).
- Metrics file exists, has the same shape as the teacher's.

---

### Iteration 3 — Add `L_logit` (Day 4–5)

**Goal:** Implement Logit KD and run conditions `kd_logit` (#2).

**Deliverable:**
- Teacher forward pass during student training: load frozen teacher, `eval()`, `no_grad()`.
- `src/losses.py` gains `logit_kd_loss(student_logits, teacher_logits, T=1.0)` returning `T² · KL(teacher || student)`.
- `configs/conditions/kd_logit.yaml`.
- A run script that toggles logit KD on or off based on config.
- `results/runs/tweet_eval-sentiment/kd_logit/metrics.json`.

**Concept learned:**
- **Hinton-style soft-label distillation**, the classic KD signal. *Why* `T²` is there (gradient scale preservation when T≠1). KL direction matters: `KL(teacher || student)` penalizes the student for placing low probability where the teacher places high.
- "Dark knowledge" — the relative ranking among non-top classes that hard labels throw away.
- Why teacher logits must be `.detach()`-ed.

**Definition of done:**
- `kd_logit` macro-F1 ≥ `ce_only` macro-F1 (almost always; if not, debug — teacher forward is probably broken).
- Per-batch `loss_logit` field appears in `run_metadata.json`; equals 0.0 when condition disables it, positive when enabled.
- Sanity check: `top1_agreement` between teacher and student is higher under `kd_logit` than `ce_only`.

---

### Iteration 4 — Add `L_hidden` (Day 6–7)

**Goal:** Implement Hidden KD and run conditions `kd_hidden` (#3) and `kd_logit_hidden` (#5).

**Deliverable:**
- `src/models.py` gains a `HiddenProjection` module: 4 independent `nn.Linear(312, 768)` modules, indexed by student layer.
- `src/losses.py` gains `hidden_kd_loss(h_student_list, h_teacher_list, projections, attention_mask)` — MSE averaged over valid tokens × hidden dim, per mapped layer, then averaged across layers.
- Layer mapping utility `student_to_teacher_layer: {1:3, 2:6, 3:9, 4:12}`.
- `configs/conditions/kd_hidden.yaml`, `configs/conditions/kd_logit_hidden.yaml`.
- Results dirs for both conditions.

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

---

### Iteration 5 — Add `L_attention` (Day 8)

**Goal:** Implement Attention KD and bring all 4 remaining conditions online: `kd_attn` (#4), `kd_logit_attn` (#6), `kd_hidden_attn` (#7), `kd_full` (#8).

**Deliverable:**
- `src/losses.py` gains `attention_kd_loss(a_student_list, a_teacher_list, attention_mask)` — uses **post-softmax** attention probabilities (design doc says so; departs from reference); MSE averaged over valid token-pairs × heads × mapped layers.
- Configs for all 4 new conditions.
- All 8 condition configs exist; the training loop reads them and toggles correctly.

**Concept learned:**
- **Structural / relational distillation** — match *which tokens attend to which*. Attention heads encode syntactic and coreference patterns (Clark et al. 2019).
- Why both models must have the same number of heads (12 here) — per-head MSE assumes alignment. (The design doc has a fallback "average over heads if mismatch" — we don't need it but the code handles it.)
- The **pre-softmax vs post-softmax** debate — what each signal means, why our design doc chose probabilities. Real consequence: gradient flows differently through `softmax` than through raw logits.

**Definition of done:**
- All 8 conditions defined and each one runs at least one forward+backward step on a tiny dev sample without OOM or NaN.
- `loss_attention` logged per batch.

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
- `src/data/adapters/<dataset>.py` for each new dataset, all conforming to the dataset adapter contract (`docs/notes/02-project-synthesis.md` §4).
- 8 × 9 = 72 student `metrics.json` files (or fewer if descoped).
- 8 new teacher checkpoints.

**Concept learned:**
- **Dataset adapter discipline** — uniform `{train, dev, test}` interface; explicit label-id mapping; sentence-pair vs single-text handling.
- **Cross-task variance** — first observation of whether per-signal contributions are stable across task families.
- **Practical KD operations** — running the same pipeline at scale, debugging the rare run that diverges.

**Definition of done:**
- All adapters expose the contract.
- All runs that completed have valid `metrics.json`.
- Failed/skipped runs are explicitly documented with a reason.

---

### Iteration 8 — Cross-dataset analysis + artifacts (Day 13)

**Goal:** Compute every analysis the design doc §7 lists, across all completed runs. Generate the assets the final presentation will use.

**Deliverable:**
- `results/analysis/`:
  - Per-condition confusion matrices.
  - Calibration: `ECE`, `NLL`, `Brier` per (dataset, condition).
  - Teacher-student analysis: `top1_agreement`, `error_copying`, `teacher_correct_student_wrong`, `teacher_wrong_student_correct`.
  - Layer similarity: `layer_cosine_similarity`, `layer_kl_divergence` per mapped pair, per condition.
  - Attention heatmaps for representative examples (CE-only vs KD; teacher vs student; correct vs wrong).
  - Efficiency: latency p50/p95, throughput, model size, parameter count.
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

- **Run metadata.** Every run writes `run_metadata.json` with the fields from design doc §6. Includes: config hash, dataset hash, split-index hash, package versions, GPU model, random seed, early-stop info, all losses (component + raw), `grad_norm`, `learning_rate`, `epoch`, `global_step`, train/eval times.
- **Checkpoint naming.** `{stage}/{dataset}/{condition}/{epoch}.pt`. Stage ∈ {teacher, student}.
- **Determinism.** `seed=42` everywhere. `torch.use_deterministic_algorithms(True)`. CUBLAS workspace env var if needed.
- **Loss safety.** `nan` / `inf` checks in the train loop; fail fast.
- **Project state isolation.** `.serena/` and `wandb/` (if used) gitignored. Checkpoints out of git.
- **Tests** — minimal: one unit test per loss function (CE, logit-KD, hidden-KD, attn-KD) verifies the formula on a tiny tensor. Not full coverage; just guardrails.

---

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Multi-VALUE / dialects datasets blocked or weirdly formatted | Medium | Order them last in Iteration 7; descope cleanly if blocked. |
| HatEval HF gating not approved in time | Low–Medium | Apply for access on Day 1. |
| FEVER too large for the 3-epoch budget at full size | Medium | Subsample to ~50K examples *if* needed; document the subsampling. |
| Attention KD post-softmax produces vanishing gradients | Low | If we see this, document and try pre-softmax variant as an explicit comparison (already an interesting subresult). |
| Sentence-pair token_type_ids handling differs across HF model versions | Medium | Test on ANLI early (Day 11) before FEVER. |
| Student loss explodes when all 4 losses are summed (kd_full) | Low | Magnitudes of the 4 losses are not naturally comparable; monitor `raw_loss_*` and document. Per design doc, weights are all 1.0 — no rescaling. |
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
1. Every dataset that was attempted has 8 condition `metrics.json` files (or a documented reason for omission).
2. The cross-task heatmap renders with all attempted (dataset, condition) cells populated.
3. Main effects and interactions are computed per dataset and aggregated across families.
4. The presentation deck content covers RQ1 + RQ2 with cited numbers.
5. `git log` shows incremental commits per iteration so the trajectory is reviewable.

---

## Suggested first move

If the user approves this roadmap, the **next action** is to enter Plan Mode again for **Iteration 0** (foundation + smoke test) and produce a concrete per-iteration sub-plan. That sub-plan will include:
- exact `environment.yml` contents
- `src/` skeleton structure
- the smoke-test script
- the conda + GPU verification commands

Nothing in the present roadmap document binds Iteration 0's specific implementation choices; those get decided at the start of Iteration 0.
