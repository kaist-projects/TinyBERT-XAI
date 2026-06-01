# TinyBERT-Ablation
![cover](docs/images/02-readme-cover.png)  

TinyBERT-Ablation is a KAIST CS50700 Deep Learning final project that replicates and
extends TinyBERT's task distillation with granular control over knowledge
distillation (KD) loss components.

- **Logit KD**: match teacher output distributions.
- **Hidden KD**: match selected teacher hidden states through learned projections.
- **Attention KD**: match teacher attention probabilities.

Each run writes structured metadata, evaluation metrics, and loss magnitudes so
the ablation can be analyzed reproducibly.

## Table of Contents

- [1. Datasets](#1-datasets)
- [2. Features](#2-features)
- [3. Getting Started](#3-getting-started)
    - [3.1. Environment Setup](#31-environment-setup)
    - [3.2. Teacher Fine-Tuning](#32-teacher-fine-tuning)
    - [3.3. Student Distillation](#33-student-distillation)
        - [3.3.1. Recommended KD loss weights by dataset](#331-recommended-kd-loss-weights-by-dataset)
    - [3.4. Full Factorial Sweep](#34-full-factorial-sweep)
    - [3.5. Analysis](#35-analysis)
    - [3.6. Cross-Dataset Analysis](#36-cross-dataset-analysis)
- [4. Project Structure](#4-project-structure)
- [5. Experimental Conditions](#5-experimental-conditions)
- [6. Notes / Limitations](#6-notes--limitations)
- [7. Acknowledgements](#7-acknowledgements)

## 1. Datasets

Datasets are selected with `--dataset <key>` and share the same teacher/student
pipeline.

| Dataset | `--dataset` | Task | Train / Dev / Test | Notes |
|---|---|---|---|---|
| [TweetEval-sentiment](https://huggingface.co/datasets/cardiffnlp/tweet_eval) | `tweet_eval-sentiment` | sentiment | 45,615 / 2,000 / 12,284 | official splits |
| [IMDB](https://huggingface.co/datasets/stanfordnlp/imdb) | `imdb` | sentiment | 22,500 / 2,500 / 25,000 | seed-42 dev split |
| [ANLI](https://huggingface.co/datasets/facebook/anli) | `anli` | NLI | 162,865 / 3,200 / 3,200 | sentence-pair input |
| [Davidson](https://huggingface.co/datasets/tdavidson/hate_speech_offensive) | `davidson` | hate speech | 19,825 / 2,479 / 2,479 | seed-42 80/10/10 split |
| [DynaHate](https://github.com/bvidgen/Dynamically-Generated-Hate-Speech-Dataset) | `dynahate` | hate speech | 32,924 / 4,100 / 4,120 | local CSV |
| [HatEval](https://huggingface.co/datasets/valeriobasile/HatEval) | `hateval` | hate speech | 13,500 / 1,500 / 4,570 | HF-gated |
| [FEVER](https://huggingface.co/datasets/pietrolesci/nli_fever) | `fever` | NLI | 45,000 / 19,998 / 5,000 | 50K train cap |
| [VarDial](https://huggingface.co/datasets/statworx/swiss-dialects) | `vardial` | dialect ID | 3,793 / 475 / 475 | seed-42 80/10/10 split |

Local datasets are gitignored: save DynaHate to
`data/dynahate/dynahate_v0.2.3.csv`. HatEval requires accepting the Hugging Face
terms and logging in before use.

## 2. Features

- End-to-end BERT teacher and TinyBERT student pipelines.
- Granular CE, Logit, Hidden, and Attention KD loss control.
- Full `2^3` factorial ablation over KD signal combinations.
- Evaluation for accuracy, calibration, and teacher-student agreement.
- Reusable analysis, markdown reports, and PNG visualizations.

## 3. Getting Started

Run commands from the repository root.

### 3.1. Environment Setup

```bash
conda create -n tinybert-xai python=3.12
conda activate tinybert-xai
pip install -r requirements.txt
```

### 3.2. Teacher Fine-Tuning

Train the BERT teacher (`--dataset` selects a registered dataset; default
`tweet_eval-sentiment`, also `imdb`, `anli`):

```bash
python scripts/01_train_teacher.py --dataset tweet_eval-sentiment
```

Training automatically evaluates on dev/test when it finishes. Expected
artifacts:

- `results/checkpoints/tweet_eval-sentiment/teacher/best.pt`
- `results/metadata/tweet_eval-sentiment/teacher/run_metadata.json`

### 3.3. Student Distillation

Describe a student run in a YAML config and pass it with `--config`. `configs/default.yaml` is the standard setup.

```bash
python scripts/02_train_student.py --config configs/kd_full.yaml
```

To define a condition, set which teacher signals the student learns from under
`run.conditions` (`logit`/`hidden`/`attention`); all off is the `ce_only`
baseline.

Expected artifacts:

- `results/checkpoints/<dataset>/student/<condition>/best.pt`
- `results/metadata/<dataset>/student/<condition>/run_metadata.json`

#### 3.3.1. Recommended KD loss weights by dataset

By default every loss term has weight `1.0`. The catch is that the three KD
signals live on very different scales: in our runs the attention loss is tiny
(around `0.003`) while the CE, logit, and hidden losses are roughly `0.2`–`0.5`.
With equal weights the larger terms dominate and the attention signal barely
influences training.

**The rule (plain English).** Make each signal count about the same as the
plain training loss (CE). For each term we take how big the CE loss usually is
and divide by how big that term usually is, using the `kd_full` runs as the
reference. CE stays at `1.0`; a term that is, say, 100x smaller than CE gets a
weight near `100`. This only rebalances how loud each signal is — it does not
change which signals are turned on.

Starting-point weights derived this way, per dataset:

| Dataset | `ce` | `logit` | `hidden` | `attn` |
|---|:---:|:---:|:---:|:---:|
| `tweet_eval-sentiment` | 1.0 | 3.0 | 2.2 | 125 |
| `imdb` | 1.0 | 1.8 | 0.9 | 240 |
| `anli` | 1.0 | 1.3 | 2.1 | 400 |
| `davidson` | 1.0 | 4.0 | 0.8 | 53 |
| `dynahate` | 1.0 | 2.3 | 1.5 | 110 |
| `hateval` | 1.0 | 2.4 | 1.1 | 100 |
| `fever` | 1.0 | 1.8 | 1.5 | 300 |
| `vardial` | 1.0 | 1.6 | 2.3 | 140 |

Set them in the run's YAML, for example for `dynahate`:

```yaml
distillation:
  loss_weights:
    ce: 1.0
    logit: 2.3
    hidden: 1.5
    attn: 110
```

**Caveats — read before using these.**

- These are a **heuristic starting point**, not tuned optimums. They balance how
  much each loss contributes; they do **not** guarantee better accuracy.
- They come from **single-seed** runs, so each magnitude is one noisy sample.
  The numbers are rounded and approximate; treat them as a first guess.
- The attention weights (roughly 100–400) are large, because the attention loss
  is so small. Confirm they actually help with a quick sweep on the **dev** set
  before trusting them, and never tune on the test set.
- The main 81-run factorial study deliberately keeps **all weights at `1.0`**
  (see §6). These dataset-specific weights are for separate, exploratory runs,
  not for the locked experiment.

### 3.4. Full Factorial Sweep

Run the teacher fine-tune (once) plus all 8 student conditions for one dataset
with a single command. The sweep invokes the per-run scripts as subprocesses for
clean per-run GPU memory isolation and is resumable. Artifacts that already
exist are skipped unless `--force`:

```bash
python scripts/07_run_dataset.py --dataset imdb
```

### 3.5. Analysis

Run the factorial analysis for a dataset (defaults to the pilot):

```bash
python scripts/06_analyze_factorial.py tweet_eval-sentiment
```

Artifacts are written per dataset under `results/analysis/<dataset>/`, so running
the analysis for another dataset never overwrites an existing report:

- `results/analysis/<dataset>/REPORT.md`
- `results/analysis/<dataset>/figures`

### 3.6. Cross-Dataset Analysis

Once several datasets have completed sweeps, roll them up into the cross-task
presentation assets. This runs in two stages, written under
`results/analysis/cross_dataset/`.

**Stage 1, metadata only (no GPU).** Reads every
`results/metadata/<dataset>/student/<condition>/run_metadata.json` that is present:

```bash
python scripts/08_cross_dataset_analysis.py
```

- `figures/cross_task_macro_f1.png`, `figures/cross_task_delta.png`: condition heatmaps
- `figures/confusion`: per-condition confusion matrices.
- `tables/*.csv`: per-dataset factorial-effect tables.
- `TABLES.md`: a quick-read index of the matrices.

**Stage 2, representation + XAI artifacts (GPU, reloads checkpoints).** Runs
forward passes on a fixed test sample for every dataset that has both a teacher
and student checkpoint:

```bash
python scripts/08b_representation_analysis.py   # N=256 test sample
```

- `representation/layer_cka.csv`: linear CKA per mapped pair
- `representation/attention_kl.csv`: KL(teacher ‖ student) of attention maps
- `representation/attention/*.png`: teacher-vs-student attention heatmaps

The written interpretation (RQ1/RQ2 answers) lives in
`results/analysis/cross_dataset/CROSS_DATASET.md`.

## 4. Project Structure

```text
src/
  analysis/       Factorial loaders, effect math, tables, plots,
                  cross-dataset roll-ups, and representation (CKA/attention) analysis
  eval/           Metrics and teacher-student analysis
  data/           Dataset registry + adapters (tweet_eval, imdb, anli)
  modeling/       Model/tokenizer loading and the hidden-state projection module
  distill/        The 8 ablation conditions and the KD losses (logit/hidden/attn)
  pipeline/       Teacher + student training/evaluation pipelines, epoch loop, early stop
  storage/        Checkpoint I/O and run-metadata logging
  config.py       Default experiment configuration
  utils.py        Cross-cutting helpers (device, param counts, autocast)

scripts/
  01_train_teacher.py            Teacher fine-tune + auto dev/test eval
  02_train_student.py            Student distillation + auto dev/test eval
  06_analyze_factorial.py        Per-dataset factorial report
  07_run_dataset.py              Teacher + all 8 conditions for one dataset
  08_cross_dataset_analysis.py   Cross-task heatmaps + tables (metadata only)
  08b_representation_analysis.py CKA, attention KL/heatmaps, efficiency (GPU)
  _dataset_cli.py       Shared --dataset flag glue
  _student_cli.py       Shared signal-flag glue

tests/            Unit tests for losses, metrics, run logs, and factorial math
docs/             Notes, plans, and source project documents
reference/        Original TinyBERT reference code used for comparison only
results/          Run metadata and analysis outputs
```

## 5. Experimental Conditions

| Condition | Logit KD | Hidden KD | Attention KD |
|---|:---:|:---:|:---:|
| `ce_only` |  |  |  |
| `kd_logit` | Y |  |  |
| `kd_hidden` |  | Y |  |
| `kd_attn` |  |  | Y |
| `kd_logit_hidden` | Y | Y |  |
| `kd_logit_attn` | Y |  | Y |
| `kd_hidden_attn` |  | Y | Y |
| `kd_full` | Y | Y | Y |

## 6. Notes / Limitations

- The current pilot is single-seed. The observed `0.0198` student spread should
  not be treated as a statistically resolved per-signal effect.
- Post-softmax Attention KD appears near-inert in the pilot: the final attention
  loss magnitude averages about `0.00453`, much smaller than CE, logit, and
  hidden losses. This should be fixed or explicitly documented before larger
  multi-dataset runs.
- `reference/` contains the original TinyBERT authors' older codebase for
  comparison. The active implementation is the modern HuggingFace/PyTorch code
  under `src/`.

## 7. Acknowledgements

This project builds on the TinyBERT distillation idea and uses HuggingFace
Transformers/Datasets for the modern implementation. It was developed as a final
project for KAIST CS50700 Deep Learning.
