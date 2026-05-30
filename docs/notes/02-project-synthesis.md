# TinyBERT-XAI — Project Synthesis

A consolidated reference for the mini-project, drawn from:
- `../source/01-tinybert-paper.pdf` (Jiao et al., EMNLP 2020) — already digested in `01-tinybert-implementation-notes.md`.
- `../source/02-project-proposal.pdf` — Team 18 / Spring 2026 proposal slides.
- `../source/03-design-doc.pdf` — the implementation contract (Korean).

**When proposal and design doc disagree, the design doc wins.** The proposal is the pitch; the design doc is the spec the implementation team committed to.

---

## 1. What this project is (and is not)

### Title
*Adapting Multi-Level Knowledge Distillation for NLP Text Classification Tasks — Per-level ablation of logit / hidden-state / attention distillation for a broad spectrum of NLP classification tasks.*

### One-sentence framing
> Fix TinyBERT's unified multi-level KD framework, then run a **per-signal factorial ablation** to measure — **per task family** — which distillation signal (logit, hidden, attention) actually carries the gains for text classification.

### Research questions
- **RQ1 (primary):** How much does each multi-level KD component — logit, hidden-state, attention — differentially contribute to text-classification performance?
- **RQ2 (secondary):** Does this pattern vary across task families — hate-speech, NLI, sentiment, dialects?

### Positioning vs. prior work
| Paper | Claim | Gap we address |
|---|---|---|
| **TinyBERT** (Jiao 2020) | Proposes multi-level KD, trained jointly | Never quantifies separate per-level contribution |
| **Beyond Logits** (Gong, ACL 2025) | "Logits alone are not enough" | Never isolates *which* non-logit signal carries the improvement |
| **WID** (Wu, NAACL 2024) | Explicit alignment losses unnecessary | But what value do hidden/attention add when kept? |
| **How to Distill Your BERT** (Wang, ACL 2023) | Attention transfer best on GLUE (binary / 3-way) | Does the ranking hold across diverse classification tasks? |
| **Targeted Distill** (Zhang, EMNLP 2025) | Sentiment-specialized KD | LLM teachers + generative students; encoder-classifier × multi-level × diverse classification tasks still missing |

**Common thread:** No prior work runs a controlled per-level ablation on encoder-based text classification across multiple task families. *That intersection is ours.*

### What this project explicitly does NOT do (§9 of design doc)
- No exact TinyBERT reproduction.
- No **sequential** task-specific KD (no separate intermediate-then-prediction phases — we do **joint** training).
- No General Distillation. We start from the *public* `TinyBERT_General_4L_312D` checkpoint and only do task-specific signal ablation.
- No random-init baseline.
- No data augmentation (Algorithm 1 from TinyBERT is **not** used here).
- No hyperparameter search.
- No seed variation (seed=42, fixed).
- No condition-specific training rules or early-stopping settings.
- No tuning on the test set.

---

## 2. Models

| Role | Model | Layers | Hidden | Params | How |
|---|---|---|---|---|---|
| Teacher | `bert-base-uncased` | 12 | 768 | ~110M | Fine-tuned **per dataset**, then frozen |
| Student | `TinyBERT_General_4L_312D` (public checkpoint) | 4 | 312 | ~14.5M | Same checkpoint init across **all** conditions |

**Layer mapping (student → teacher), fixed:**
```
1 → 3
2 → 6
3 → 9
4 → 12
```
(Uniform `g(m) = 3m`, matching TinyBERT paper.)

**Tokenizer:** `bert-base-uncased` for both teacher and student.

**Sentence-pair tasks:** `text_a`, `text_b`, `token_type_ids` policy must be recorded in metadata. Tokenizer and truncation must be identical for teacher and student.

The public TinyBERT checkpoint is treated as *already general-distilled* — so this experiment isolates only the **task-specific** signal effect.

---

## 3. Experiment matrix — the factorial design

**This is the most important design doc update vs. the proposal.**

- **Proposal slides:** 5 conditions (additive — logit always included once we leave `student_only`).
- **Design doc §2:** **8 conditions — full `Logit × Hidden × Attention` 2³ crossed factorial**. CE is always ON.

| # | Condition | Binary (L,H,A) | CE | Logit KD | Hidden KD | Attention KD |
|---|---|---|---|---|---|---|
| 1 | `ce_only` | 000 | ON | OFF | OFF | OFF |
| 2 | `kd_logit` | 100 | ON | ON | OFF | OFF |
| 3 | `kd_hidden` | 010 | ON | OFF | ON | OFF |
| 4 | `kd_attn` | 001 | ON | OFF | OFF | ON |
| 5 | `kd_logit_hidden` | 110 | ON | ON | ON | OFF |
| 6 | `kd_logit_attn` | 101 | ON | ON | OFF | ON |
| 7 | `kd_hidden_attn` | 011 | ON | OFF | ON | ON |
| 8 | `kd_full` | 111 | ON | ON | ON | ON |

A full factorial lets us compute **main effects** and **interaction effects** for each signal — strictly more informative than the proposal's 5-condition additive ladder.

### Contribution readouts (proposal §4.3, adapted to 8 conditions)
- **Main effects** can be averaged over the half of conditions where each signal is ON vs. OFF.
- **Pairwise interactions** computed via standard factorial analysis (e.g., `kd_full - (kd_logit_hidden + kd_logit_attn + kd_hidden_attn - kd_logit - kd_hidden - kd_attn + ce_only)` style ANOVA-on-metric).
- Per-condition residual against `ce_only` gives the absolute lift.

---

## 4. Datasets — 9 datasets across 4 task families

| Family | Dataset | Source | Input | Labels |
|---|---|---|---|---|
| Hate speech | **DynaHate** (Vidgen et al., ACL 2021) | GitHub | single text | binary or multi-class (adapter-defined) |
| Hate speech | **Davidson hate/offensive** (Davidson, ICWSM 2017) | GitHub | single text | hate / offensive / neither |
| Hate speech | **HatEval** (Basile, NAACL 2019) | HF gated | social media text | English HS-focused |
| NLI | **ANLI** (Nie, ACL 2020) | `facebook/anli` | sentence pair | entail / neutral / contradict |
| NLI | **FEVER** (Thorne, NAACL 2018) | `fever` or custom adapter | claim + evidence | supports / refutes / not enough info |
| Sentiment | **IMDB** (Maas, ACL 2011) | `stanfordnlp/imdb` or `imdb` | long single text | positive / negative |
| Sentiment | **TweetEval sentiment** (Barbieri, EMNLP 2020) | `cardiffnlp/tweet_eval`, config `sentiment` | tweet text | negative / neutral / positive |
| Dialects | **Aepli / VarDial 2023** (Aepli, EACL 2023) | VarDial shared task / SID4LR | single text or intent utterance | dialect / intent labels |
| Dialects | **Multi-VALUE** (Ziems, ACL 2023) | GitHub / `value-nlp` toolkit / SALT-NLP HF variants | transformed item | dialect / variant labels (50-label multiclass per slides) |

### Dataset adapter contract (every adapter exposes the same fields)
- `dataset_name`
- `task_family`
- `input_type`: `single_text` or `sentence_pair`
- `label_names` and label-id mapping
- `train` / `dev` / `test` splits. If no official split or no dev set, seed=42 stratified split, with split-index hash stored.
- `num_labels`, split sizes, class distribution
- raw file hash + split index hash for custom/local datasets

---

## 5. Loss specification (design doc §6)

**Total loss (every batch, sum only the active terms):**
```
L_total = 1.0 * L_CE
        + 1.0 * I_logit     * L_logit
        + 1.0 * I_hidden    * L_hidden
        + 1.0 * I_attention * L_attention
```
where `I_*` are per-condition on/off indicators (0 or 1). All non-zero weights are exactly **1.0** — no weight tuning.

### 5.1 CE (always on)
```
L_CE = CrossEntropy(y_gold, logits_student)
```
- No weighted CE, no label smoothing.
- Label mapping must be identical across teacher, student, and all runs.

### 5.2 Logit KD
```
p_t = softmax(z_teacher / T)
p_s = softmax(z_student / T)
L_logit = T^2 * KL(p_t || p_s)
T = 1.0
```
- **KL direction:** `KL(teacher || student)`.
- `T^2` scaling preserved (canonical Hinton form).
- `T = 1.0` fixed.
- Teacher logits `.detach()`.

### 5.3 Hidden KD
Teacher hidden 768, student hidden 312 → use a **trainable projection 312 → 768 per mapped student layer**.

```
L_hidden = mean_over_mapped_layers(
    mean_over_valid_tokens_and_hidden_dim(
        || P_l(h_student_l) - h_teacher_m ||_2^2
    )
)
```
- Projection is a trainable linear layer **per mapped student layer** (4 projections, one per layer pair).
- "Valid token" = `attention_mask == 1`.
- Average denominator: `valid_token_count × hidden_dim`.
- Teacher hidden `.detach()`.
- Tokenizer and truncation must be identical for teacher and student.

### 5.4 Attention KD
```
L_attention = mean_over_mapped_layers(
    mean_over_valid_token_pairs_and_heads(
        || A_student_l - A_teacher_m ||_2^2
    )
)
```
- Uses **attention probabilities** (post-softmax), per design doc. **NB:** this differs from the original TinyBERT paper, which used unnormalized logits. Design doc explicitly says "attention probability를 사용한다".
- Valid token pair = both query and key are non-padding.
- Average denominator: `valid_token_pair_count × head_count`.
- If teacher and student head counts differ, average each side's attention over the head dimension first, then MSE. (Both are 12 heads here, so this is a no-op for us — but the implementation must handle it.)
- Teacher attention `.detach()`.
- Attention heatmaps are stored as **exploratory artifact**, *not* used for causal explanation.

> **⚠️ Discrepancy with TinyBERT paper:** The original paper uses pre-softmax attention *logits* for `L_attn`; this project uses *probabilities*. Follow the design doc.

---

## 6. Training recipe (design doc §5)

### Joint training, single pass
**This project's task-specific training is NOT sequential.** No "intermediate-first, then prediction" phase split. All active losses go into one batch and one backward pass.

```
Initialization:
  Load TinyBERT_General_4L_312D public checkpoint into student.

Joint training:
  For each batch:
    L_total = sum of active losses (per condition)
    backprop; optimizer step.
```

Example total losses for some conditions:
- `kd_full`        = CE + LogitKD + HiddenKD + AttentionKD
- `kd_hidden_attn` = CE + HiddenKD + AttentionKD
- `kd_logit`       = CE + LogitKD

### Single hyperparameter set (shared by all datasets × conditions)
| Item | Value |
|---|---|
| seed | 42 |
| optimizer | AdamW |
| learning rate | 2e-5 |
| batch size | 16 (use gradient accumulation if memory tight) |
| max epochs | 3 |
| max sequence length | **128** for every dataset/condition |
| early-stop monitor | `dev_macro_f1` |
| early-stop mode | maximize |
| early-stop patience | 2 |
| mixed precision | hardware-dependent, but applied uniformly across runs |

### Early stopping
- Teacher and student may both use early stopping.
- Same rule (monitor/patience/mode) applies to all 8 conditions for the student. Different rules per condition are **forbidden**.
- Final test evaluation: choose dev-best checkpoint, evaluate test once. **No repeated test-set tuning.**
- Metadata to log: `run_metadata.json` records early-stop use, monitor, patience, best step/epoch.

### Teacher training
- Each dataset gets its own fine-tuned `bert-base-uncased`. Frozen after training.
- During student training, teacher is in `eval()` mode with `no_grad()`.

### Checkpointing
- Save **all epoch checkpoints** for each run, under a per-run path.

---

## 7. Evaluation (design doc §7)

### Primary + secondary metrics
| Metric | Role |
|---|---|
| `macro_f1` | **Primary** (and the dev early-stop monitor) |
| `micro_f1` | Secondary, instance-level F1 |
| `accuracy` | Secondary |
| `per_class_f1` | Per-class lift/regression |
| `confusion_matrix` | Class-pair error movement |

### Calibration
- `ECE` (Expected Calibration Error)
- `NLL`
- `Brier`

(All "lower is better".)

### Teacher–student analysis
| Metric | Purpose |
|---|---|
| `top1_agreement` | Teacher/student argmax agreement rate |
| `teacher_student_kl` | KL(p_teacher || p_student) |
| `teacher_correct_student_wrong` | Student missing teacher knowledge |
| `teacher_wrong_student_correct` | Student outperforming teacher |
| `error_copying` | Teacher and student making the same wrong prediction |

### Efficiency
| Metric | Notes |
|---|---|
| `latency_p50_ms`, `latency_p95_ms` | After warmup, fixed batch inference |
| `throughput_samples_per_sec` | Fixed hardware/batch |
| `model_size_mb` | Serialized checkpoint size |
| `parameter_count` | Trainable / total |
| `gpu_memory_mb` | Peak memory |

### Attention heatmaps (artifact)
Saved for representative examples. Required comparisons:
- `ce_only` vs KD conditions
- teacher vs student
- correct vs wrong cases

Each saved entry has: `dataset`, `condition`, `split`, `example_id`, `mapped_layer_pair`, `head_strategy` (`original_heads` or `mean_heads`), `token_list`, `attention_matrix`.

### Layer similarity (per mapped layer pair)
| Metric | Target |
|---|---|
| `layer_cosine_similarity` | Projected student hidden vs. teacher hidden, averaged over valid tokens |
| `layer_kl_divergence` | Attention distribution; `KL(teacher, student)` |

- Hidden similarity compares `P_l(h_student)` and `h_teacher`.
- Token-level cosine averaged over valid tokens.
- For KL on hidden vectors, the normalization method (softmax / other) must be specified in metadata.

### Per-batch loss logging fields
`loss_total`, `loss_ce`, `raw_loss_ce`, `loss_logit`/`raw_loss_logit`, `loss_hidden`/`raw_loss_hidden`, `loss_attention`/`raw_loss_attention` (null when condition disables that term), `grad_norm`, `learning_rate`, `epoch`, `global_step`, `training_time_per_epoch`, `total_training_time`.

---

## 8. Where this differs from the original TinyBERT paper

| Topic | TinyBERT paper | This project |
|---|---|---|
| Stages | Two: General Distillation → Task-specific Distillation | **One:** task-specific only. GD inherited via the public checkpoint. |
| Task-specific training | **Sequential**: intermediate KD → prediction KD | **Joint**: all active losses in one pass |
| `L_attn` input | **Pre-softmax** attention logits | **Post-softmax** attention probabilities |
| Data augmentation | Algorithm 1 (BERT-MLM + GloVe, `N_a=20`, `K=15`) | **None** |
| Hyperparameter search | Per-task search on `lr` ∈ {1e-5, 2e-5, 3e-5}, `batch` ∈ {16, 32} | **None** — single global HP set |
| Random-init baseline | N/A | **Not built** |
| Seeds | Multiple | **Fixed 42** |
| Max seq length | 64 single / 128 pair | **128 everywhere** |
| Epochs (TD-intermediate) | 10–50 | **3 epochs max, joint** |
| Eval scope | GLUE | 9 datasets × 4 task families (hate, NLI, sentiment, dialects) |
| Ablation design | A few specific knock-outs (Table 3) | **Full 2³ factorial on {Logit, Hidden, Attention}**, plus full multi-task readout |

Most of the gap is justified by the project's scope: this is an **ablation study** of signal contributions across task families, not a TinyBERT reimplementation. The public general-distilled checkpoint gives us a strong, comparable starting point without needing to redo GD on Wikipedia.

---

## 9. Run plan and numbers

- Datasets: **9**
- Conditions per dataset: **8** (full factorial)
- Total student runs: **9 × 8 = 72** student-training runs
- Plus **9** teacher fine-tuning runs

Each student run produces: model checkpoint(s), per-class/macro/micro F1 + accuracy, confusion matrix, calibration metrics, teacher-student agreement metrics, efficiency metrics, attention heatmap artifacts for representative examples, layer-similarity metrics.

Per appendix of proposal, four analyses per experiment:
1. **Overall performance** — accuracy / macro-F1 / per-class F1 across all runs.
2. **Confusion matrix** — per-condition; track which class boundaries each KD level resolves.
3. **Marginal contribution** — main effects + interaction (under the 8-condition factorial this is cleaner than the proposal formulas).
4. **Cross-task comparison** — is the per-level ranking consistent across all 9 datasets / 4 families?

---

## 10. Implementation checklist (consolidated from design doc §8)

### Dataset layer
- [ ] 9 dataset configs, all with `train`/`dev`/`test` exposed by the adapter.
- [ ] Label mapping fixed across the whole run.
- [ ] Sentence-pair datasets declare `text_a`, `text_b`, `token_type_ids` policy.
- [ ] Custom/local datasets save raw-file hash + split-index hash.

### Model layer
- [ ] Teacher = `bert-base-uncased`, fine-tuned then frozen per dataset.
- [ ] Student = same `TinyBERT_General_4L_312D` public checkpoint init for every condition.
- [ ] Tokenizer = `bert-base-uncased` for both.
- [ ] Layer mapping hard-coded `1→3, 2→6, 3→9, 4→12`.
- [ ] No random-init baseline built.

### Training layer
- [ ] seed = 42 everywhere.
- [ ] No HP search.
- [ ] No sequential distillation path.
- [ ] Joint training: any combination of active losses goes through one backward pass.
- [ ] Same early-stopping rule for all 8 conditions.
- [ ] Teacher in `eval()` and `no_grad()` during student training.
- [ ] Log both component loss and raw loss.
- [ ] Save every epoch checkpoint per run.

### Loss layer
- [ ] `L_CE` always on.
- [ ] `L_logit` = `T^2 * KL(teacher || student)`, `T = 1.0`.
- [ ] `L_hidden` uses a trainable `312 → 768` projection per mapped student layer.
- [ ] `L_hidden` averages over `valid_token × hidden_dim`.
- [ ] `L_attention` averages over `valid_token_pair × head`.
- [ ] If teacher/student head counts mismatch, use the head-average strategy.

### Evaluation layer
- [ ] Save `macro_f1` as primary.
- [ ] Save `micro_f1`, `accuracy`, `per_class_f1`, `confusion_matrix`.
- [ ] Save `ECE`, `NLL`, `Brier`.
- [ ] Save `top1_agreement`, `teacher_student_kl`, `error_copying`.
- [ ] Measure latency, throughput, model size.
- [ ] Save attention heatmaps as artifacts.
- [ ] Save `layer_cosine_similarity`, `layer_kl_divergence`.
- [ ] Compute factorial main effects and interaction effects.

---

## 11. Open questions to surface during planning

1. **Compute budget.** 72 student runs + 9 teacher runs is substantial. IMDB and ANLI are large; FEVER is *very* large. Do we need to pilot on a subset first?
2. **Attention KD probability vs. logit.** The design doc says post-softmax probabilities; the TinyBERT paper says pre-softmax. Confirm before implementing — the choice can be re-examined if results are weak.
3. **Sentence-pair vs. single-text uniform max_seq=128.** IMDB documents and FEVER claims+evidence often exceed 128 tokens. Confirm the truncation policy is acceptable.
4. **Dialects datasets.** Aepli/VarDial 2023 and Multi-VALUE need custom adapters and are the least-standardized — possible source of implementation risk.
5. **Multi-VALUE 50-label vs. binary.** The proposal slide hints at "50-label multiclass" for Ziems et al., 2023 — confirm the label setup and whether macro-F1 is meaningful with that many classes + class imbalance.
6. **HatEval gating.** HF gated dataset — access needs to be arranged.
7. **Layer-wise projection per mapped layer.** Design doc says "trainable linear layer per mapped student layer" → 4 separate `W_h` matrices, not one shared. The TinyBERT paper used one shared projection. Confirm.
8. **Reporting.** Proposal's 5-condition contribution formulas (e.g., "Hidden = ③ − ②") are not valid for the 8-condition factorial as-is — the analysis plan needs to be reformulated in terms of main/interaction effects.

These are the items I'd want to resolve before locking the implementation plan.
