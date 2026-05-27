# TinyBERT-XAI — Project Context for Claude

## Codebase Navigation

Always use Serena MCP tools first when navigating or searching the codebase (e.g., `mcp__serena__find_symbol`, `mcp__serena__find_declaration`, `mcp__serena__get_symbols_overview`). Only fall back to grep, find, or Bash if Serena doesn't cover the need.

## What this project is

A **CS graduate course mini-project at KAIST**, Team 18, Spring 2026.

**Title:** *Adapting Multi-Level Knowledge Distillation for NLP Text Classification — per-level ablation of logit / hidden / attention distillation across diverse task families.*

**Research question:** Fix TinyBERT's multi-level KD framework, run a per-signal factorial ablation, and measure — per task family (hate speech, NLI, sentiment, dialects) — which distillation signal actually carries the gains.

**The science:** This is an **ablation study**, not a TinyBERT reimplementation. The contribution is the controlled factorial readout across 9 datasets, not a new method.

## Course constraints (read this first)

- **~2 weeks until final presentation.** Time is the dominant constraint.
- **Do not reimplement what already exists.** `reference/` is the original TinyBERT authors' repository (`general_distill.py`, `task_distill.py`, `data_augmentation.py`, `transformer/`). Adapt and refactor it. Do not build a parallel implementation from scratch.
- **Implementation choices favor "good enough + on time" over "elegant + late."** Suggest the leanest path that satisfies the design-doc contract.
- **Compute is not the bottleneck. Wall-clock to write/debug code is.** A 12 GB GPU runs this comfortably; the schedule does not have room for re-runs of failed implementations.

## Document map

`docs/source/` holds user-provided raw artifacts (PDFs). `docs/notes/` holds AI-maintained reference docs (markdown).

| File | Role |
|---|---|
| `docs/source/01-tinybert-paper.pdf` + `docs/notes/01-tinybert-implementation-notes.md` | Source paper and my digested notes. Reference for *what TinyBERT does*. |
| `docs/source/02-project-proposal.pdf` | Pitch slides. Helpful for framing, **outdated on the experiment matrix.** |
| `docs/source/03-design-doc.pdf` | **The binding spec.** Korean. When in doubt, this is the source of truth. |
| `docs/notes/02-project-synthesis.md` | Consolidated reference I wrote that merges all three. **Read this for the full picture.** |
| `docs/notes/03-roadmap.md` | 9-iteration agile roadmap; check iteration status before starting work. |
| `reference/` | Original TinyBERT authors' code. Adapt from this. |

## What's locked by the design doc (do not deviate)

- **Teacher:** `bert-base-uncased`, fine-tuned per dataset then frozen.
- **Student:** public `TinyBERT_General_4L_312D` checkpoint, identical init across all conditions.
- **Layer mapping:** `1→3, 2→6, 3→9, 4→12` (uniform `g(m)=3m`).
- **Loss form:** `L_total = L_CE + I_logit·L_logit + I_hidden·L_hidden + I_attn·L_attn`, all weights 1.0.
- **Logit KD:** `T²·KL(teacher || student)`, `T=1.0`, teacher logits detached.
- **Hidden KD:** trainable `312→768` projection **per mapped student layer** (4 projections). Mean over `valid_tokens × hidden_dim`. Teacher detached.
- **Attention KD:** uses **attention probabilities** (post-softmax). Mean over `valid_token_pairs × heads`. *Differs from the TinyBERT paper, which used pre-softmax logits.*
- **Training:** **joint**, single backward pass. No sequential intermediate→prediction phases.
- **Single global HP set:** seed=42, AdamW, lr=2e-5, batch=16, max_seq=128 everywhere, epochs=3, patience=2, monitor=`dev_macro_f1`.

## The 8-condition factorial (this is the experiment)

| # | Condition | (L,H,A) | CE | Logit | Hidden | Attn |
|---|---|---|---|---|---|---|
| 1 | `ce_only` | 000 | ✓ | | | |
| 2 | `kd_logit` | 100 | ✓ | ✓ | | |
| 3 | `kd_hidden` | 010 | ✓ | | ✓ | |
| 4 | `kd_attn` | 001 | ✓ | | | ✓ |
| 5 | `kd_logit_hidden` | 110 | ✓ | ✓ | ✓ | |
| 6 | `kd_logit_attn` | 101 | ✓ | ✓ | | ✓ |
| 7 | `kd_hidden_attn` | 011 | ✓ | | ✓ | ✓ |
| 8 | `kd_full` | 111 | ✓ | ✓ | ✓ | ✓ |

**Note vs. proposal slides:** the proposal lists only 5 conditions. The design doc supersedes it with the full 2³ factorial above. Use these 8.

## Datasets (9 total, 4 families)

- **Hate speech:** DynaHate, Davidson, HatEval (HF-gated)
- **NLI:** ANLI (`facebook/anli`), FEVER
- **Sentiment:** IMDB, TweetEval-sentiment (`cardiffnlp/tweet_eval`)
- **Dialects:** Aepli/VarDial 2023, Multi-VALUE *(highest implementation risk — custom adapters)*

Total runs: **9 teacher fine-tunes + 9 × 8 = 72 student runs = 81 runs.**

## Forbidden by design doc §9

- TinyBERT exact reproduction
- Sequential task-specific KD
- Random-init baseline
- Data augmentation (Algorithm 1 of the paper)
- Hyperparameter search
- Seed variation (42 only)
- Condition-specific training rules or early-stopping settings
- Test-set tuning

## Working style for this project

1. **Default to adapting `reference/`.** Especially `reference/task_distill.py` (training loop), `reference/transformer/` (model code with hidden-state and attention output hooks already implemented).
2. **Keep the dataset adapter contract uniform** (see synthesis §4). Adding a dataset later is a 30-min job when the contract is clean, a day when it isn't.
3. **Log everything the design doc §6/§7 asks for** from day one, in `run_metadata.json` per run. Adding logging fields after runs are done means re-running.
4. **Pilot on the smallest dataset first** (Davidson or TweetEval-sentiment) before kicking off FEVER/ANLI.
5. **Tell the user explicitly when proposal and design doc disagree.** The design doc wins, but the user should know when we're departing from the pitch.
6. **Do not "improve" the spec.** Cleaner loss weighting, smarter mappings, better schedulers — interesting, out of scope. Two-week budget.

## Open questions pending user resolution

These are flagged in `docs/notes/02-project-synthesis.md` §11 — revisit at planning time before locking implementation:

1. Compute budget / which GPU is actually available.
2. Attention KD: confirm post-softmax (design doc) vs. pre-softmax (paper) — design doc currently wins.
3. Truncation policy for IMDB / FEVER at the hard `max_seq=128` cap.
4. Multi-VALUE label setup (50-class vs. binary).
5. HatEval HF-gated access.
6. Per-layer vs. shared hidden projection (design doc says per-layer; TinyBERT paper used shared).
7. Reporting: proposal's 5-condition contribution formulas need re-derivation for the 8-condition factorial (main effects + interactions).

## Hardware (placeholder — update once confirmed)

Workload requires only **~5 GB peak VRAM** (teacher fine-tune is the heaviest single allocation, with mixed precision). A 12 GB consumer GPU is comfortable; 8 GB is workable with gradient accumulation. 32 GB system RAM + ~50 GB SSD is enough.

> User to confirm actual GPU/VRAM and update this section.
