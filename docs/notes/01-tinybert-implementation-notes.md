# TinyBERT — Implementation Notes

Source: *TinyBERT: Distilling BERT for Natural Language Understanding* (Jiao et al., 2020, arXiv:1909.10351v5).

The goal of these notes is to be a self-contained reference for **implementing TinyBERT from scratch (or from the reference code in `reference/`) on a mini-project**. Background and ablations are kept brief; recipes and equations are explicit.

---

## 1. The big picture

- **Teacher:** a fine-tuned (or pretrained) `BERT_BASE`. `N = 12` layers, `d = 768`, `d_i = 3072`, `h = 12`. ~109M params.
- **Student:** a much smaller Transformer with the same overall architecture but fewer layers and a smaller hidden size.
  - `TinyBERT_4`: `M = 4`, `d' = 312`, `d'_i = 1200`, `h = 12`. ~14.5M params. ~9.4× faster than teacher.
  - `TinyBERT_6`: `M = 6`, `d' = 768`, `d'_i = 3072`, `h = 12`. ~67M params. ~2× faster, on-par with teacher on GLUE.
- **Two distillation stages** (Figure 1 of the paper):
  1. **General Distillation (GD)** — distill on a large unlabeled corpus (English Wikipedia in the paper) using *pretrained* BERT (not fine-tuned) as the teacher. Output: a "general TinyBERT" used as the student initialization for stage 2.
  2. **Task-specific Distillation (TD)** — distill on an *augmented* task-specific dataset using the *fine-tuned* BERT as the teacher.
- **Within each stage**, distillation losses are applied at three "layer types": embedding, every transformer block, and (optionally) the prediction layer.

Both stages are needed. Ablation (Table 2): removing either GD or TD drops avg score from 75.6 → ~68–72 on the dev set.

---

## 2. Transformer distillation losses

Let `M` = number of student transformer layers, `N` = number of teacher transformer layers. Define a **layer mapping** `g : {0..M+1} → {0..N+1}` with conventions:
- `g(0) = 0` (embedding ↔ embedding)
- `g(M+1) = N+1` (prediction ↔ prediction)
- For `TinyBERT_4` ↔ `BERT_BASE`: **uniform strategy** `g(m) = 3·m` (i.e. student layer `m` learns from teacher layer `3m`). The paper compared this to top-only and bottom-only mappings; uniform was best on average (Table 4).

The combined per-layer loss (Eq. 11):

```
L_layer(m) = L_embd                  if m = 0
           = L_hidn + L_attn          if 0 < m ≤ M
           = L_pred                   if m = M+1
```

with the overall objective being a weighted sum over `m` (the paper sets all weights `λ_m = 1`).

### 2.1 Attention-based loss `L_attn` (per transformer block)

```
L_attn = (1/h) * Σ_{i=1..h} MSE(A_i^S, A_i^T)
```

- `A_i ∈ R^{l×l}` is the i-th head's attention matrix at the chosen layer (input length `l`).
- **IMPORTANT:** use the **unnormalized** attention logits (i.e. `QK^T / √d_k`), *not* `softmax(A)`. The paper found this converges faster and performs better.
- Requires that student and teacher have the **same number of heads** `h` (paper keeps `h = 12` for both).
- Motivation: attention heads encode rich linguistic structure (Clark et al. 2019) — syntax, coreference, etc.

### 2.2 Hidden-states loss `L_hidn` (per transformer block)

```
L_hidn = MSE(H^S · W_h, H^T)
```

- `H^S ∈ R^{l×d'}`, `H^T ∈ R^{l×d}` are the outputs of the Transformer layer (post-FFN, pre-next-layer).
- `W_h ∈ R^{d'×d}` is a **learnable** linear projection that maps student's smaller hidden size up to teacher's. This is what enables `d' < d`.
- Initialize `W_h` as a random/identity-like matrix and let it train with the rest.

### 2.3 Embedding-layer loss `L_embd` (m = 0)

```
L_embd = MSE(E^S · W_e, E^T)
```

- `E^S, E^T` are token embedding matrices (same shape conventions as hidden states).
- `W_e ∈ R^{d'×d}` is **a separate learnable projection** (do not share with `W_h`).

### 2.4 Prediction-layer loss `L_pred` (m = M+1)

Soft cross-entropy on logits with temperature `t`:

```
L_pred = CE( softmax(z^T / t), softmax(z^S / t) )
```

- `z^S, z^T` are the student/teacher logits.
- **Paper uses `t = 1`** (worked best in their experiments).
- This is only used at **prediction-layer distillation** — not during intermediate-layer distillation. See §3.

---

## 3. Two-stage training recipe

### 3.1 Stage 1 — General Distillation (GD)

- **Teacher:** pretrained `BERT_BASE` (no fine-tuning).
- **Data:** large unlabeled corpus (English Wikipedia, ~2.5B words in the paper).
- **Losses used:** `L_embd` + `L_hidn` + `L_attn`. **No `L_pred` at this stage** (footnote 2 in the paper — they only want the student to learn *intermediate* representations from pretrained BERT; adding prediction-layer distillation here gave no extra gain).
- **Settings used in paper:**
  - max sequence length: 128
  - epochs: 3 (over the corpus)
  - other hyperparameters: keep the same as BERT pre-training.
- **Output:** "general TinyBERT", which serves as the **initialization** for stage 2.

> *Without good GD initialization, removing intermediate-layer distillation (the `w/o Trm` ablation) collapses MNLI avg from 75.6 → 56.3 (Table 3). The general TinyBERT initialization is doing a lot of work.*

### 3.2 Stage 2 — Task-specific Distillation (TD)

- **Teacher:** `BERT_BASE` **fine-tuned on the target task**.
- **Data:** augmented task-specific dataset (see §4).
- Two sub-steps, in order:
  1. **Intermediate-layer distillation:** loss = `L_embd + L_hidn + L_attn` (same as GD).
     - batch size: 32
     - learning rate: 5e-5
     - epochs: **20** (paper uses 10 for the larger datasets MNLI/QQP/QNLI, and 50 for the small/hard CoLA).
  2. **Prediction-layer distillation:** loss = `L_pred`.
     - epochs: 3
     - batch size: chosen from {16, 32} via dev set
     - learning rate: chosen from {1e-5, 2e-5, 3e-5} via dev set
- **Max sequence length:** 64 for single-sentence tasks, 128 for sentence-pair tasks.
- For STS-B (regression), use the *original* training set (no augmentation).

> *The two sub-steps are sequential — first match the internal representations, then match the output distribution. Don't mix them in the same training pass.*

---

## 4. Data augmentation (task-specific only)

Algorithm 1 from the paper. For each training example `x` (a sequence of words), generate `N_a` augmented copies:

```
for each augmented copy x_m of x:
    for each position i in x:
        # 1. Build a candidate replacement set C for word x[i]
        if x[i] is a single-piece word:
            mask x_m[i] with [MASK]
            C = top-K most probable words from BERT(x_m)[i]
        else:  # multi-piece word
            C = top-K most similar words to x[i] using GloVe cosine similarity

        # 2. Decide whether to replace
        sample p ~ Uniform(0, 1)
        if p ≤ p_t:
            x_m[i] = uniform_random_choice(C)
    append x_m to the augmented dataset
```

**Default hyperparameters used in the paper:** `p_t = 0.4`, `N_a = 20`, `K = 15`.

**Notes for implementation:**
- "Single-piece" vs "multi-piece" is decided by BERT's WordPiece tokenizer: if `tokenize(word)` yields one token, it's single-piece.
- For single-piece words, run a single BERT MLM forward pass per masked position (or batch them).
- GloVe embeddings (Pennington et al. 2014) are needed for the multi-piece path — pick top-K by cosine similarity.
- Ablation (Table 2): removing DA drops avg from 75.6 → 68.4 on dev — augmentation is essential, especially on small tasks like CoLA/MRPC.

---

## 5. Layer mapping function `g(m)`

For `M = 4` student / `N = 12` teacher, three strategies were compared (Table 4):

| Strategy | Definition | MNLI-m | MRPC | CoLA | Avg |
|---|---|---|---|---|---|
| **Uniform** | `g(m) = 3m` | 82.8 | 85.8 | 50.8 | **75.6** |
| Top | `g(m) = m + N - M` (use teacher's top layers) | 81.7 | 83.6 | 35.9 | 70.9 |
| Bottom | `g(m) = m` (use teacher's bottom layers) | 80.6 | 84.6 | 38.5 | 71.3 |

**Default to uniform.** Adaptive per-task mapping was flagged as future work.

For `TinyBERT_6` (`M = 6`), uniform → `g(m) = 2m`.

---

## 6. Model size cheat-sheet

| | layers M/N | hidden d | FFN d_i | heads h | params |
|---|---|---|---|---|---|
| BERT_BASE (teacher) | 12 | 768 | 3072 | 12 | 109M |
| TinyBERT_4 | 4 | 312 | 1200 | 12 | 14.5M |
| TinyBERT_6 | 6 | 768 | 3072 | 12 | 67M |

Note: heads stay at 12 in both student configs — this is required for the `L_attn` formulation (per-head MSE assumes equal head counts).

---

## 7. Key results to sanity-check against

- `TinyBERT_4` vs `BERT_BASE` on GLUE test (Table 1): 77.0 vs 79.5 avg → ~96.8% of teacher, with ~13.3% of params and ~10.6% of inference time.
- `TinyBERT_4` beats 4-layer DistilBERT/BERT-PKD by ≥4.4 avg points with ~28% of their params.
- `TinyBERT_6` (67M) is roughly on-par with `BERT_BASE` on GLUE (79.4 vs 79.5 avg).
- Ablation (Table 3) on dev set:
  - Removing intermediate Transformer-layer distillation (`w/o Trm`) is the most damaging: 75.6 → 56.3 avg.
  - Of the two intermediate losses, **attention distillation matters more** than hidden-state distillation, but both contribute.
  - Removing embedding or prediction distillation: ~1–1.5 point drop.

---

## 8. Implementation checklist for a mini-project

A minimum viable TinyBERT pipeline on a single GLUE task (e.g., SST-2 or MRPC), assuming compute is limited:

1. **Pick a task** — SST-2 or MRPC are good small-scale starting points; MNLI is the most-cited but expensive.
2. **Fine-tune `BERT_BASE` on the task** → this is your teacher. Save logits/intermediate states as needed.
3. **Build student architecture** (`TinyBERT_4` config), with learnable `W_h` and `W_e` projections from `d'=312 → d=768`.
4. **(Optional) General Distillation** — skip if you have no compute; download the released "general TinyBERT" checkpoint to initialize the student instead. Without *some* initialization (general TinyBERT or otherwise), task-specific distillation underperforms badly.
5. **Data augmentation** — run Algorithm 1 (`p_t=0.4, N_a=20, K=15`) on the task training set. Cache the augmented dataset to disk; it's deterministic given a seed and reused across both TD sub-steps.
6. **TD Step 1 — intermediate distillation:**
   - Loss: `Σ_m (L_embd + L_hidn + L_attn)` with `g(m) = 3m`, weights all 1.
   - Use *pre-softmax* attention logits.
   - LR 5e-5, batch 32, ~20 epochs (10 for big sets, 50 for CoLA).
7. **TD Step 2 — prediction distillation:**
   - Loss: `L_pred` with `t = 1`.
   - LR ∈ {1e-5, 2e-5, 3e-5}, batch ∈ {16, 32}, 3 epochs; pick best on dev.
8. **Evaluate** on the task's dev set, then submit to the official test server if applicable.

Recommended sanity checks while building:
- Verify `A^S, A^T` shapes match (`l × l × h` per layer) — wrong attention-mask handling silently breaks `L_attn`.
- Confirm you're using **logits** (not softmax) for `L_attn`.
- Confirm `W_h` and `W_e` are separate, learnable, and included in the optimizer.
- Confirm that during prediction-layer distillation you've turned off the intermediate losses.

---

## 9. References within this repo

- `reference/general_distill.py` — author's GD implementation.
- `reference/task_distill.py` — author's TD implementation (covers both sub-steps).
- `reference/data_augmentation.py` — Algorithm 1 implementation (BERT-MLM + GloVe).
- `reference/pregenerate_training_data.py` — corpus preparation for GD.
- `reference/transformer/` — student/teacher model code.
- `reference/README.md` — original author README; consult for exact CLI flags.
