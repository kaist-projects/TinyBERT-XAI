# TinyBERT-XAI

TinyBERT-XAI is a KAIST CS50700 Deep Learning final project that replicates and
extends TinyBERT-style task distillation with granular control over knowledge
distillation (KD) loss components.

The project fine-tunes a `bert-base-uncased` teacher, distills into
`huawei-noah/TinyBERT_General_4L_312D` students, and compares CE-only training
against Logit KD, Hidden KD, Attention KD, and all factorial combinations of
those three KD signals. The current completed pilot uses TweetEval sentiment
classification (`cardiffnlp/tweet_eval`, config `sentiment`).

## Overview

The main experimental question is how individual KD signals contribute to a
TinyBERT student's downstream behavior when they can be enabled or disabled
independently. Instead of only comparing an additive ladder of methods, this
repository defines a full `2^3` factorial ablation over:

- Logit KD: match teacher output distributions.
- Hidden KD: match selected teacher hidden states through learned projections.
- Attention KD: match teacher attention probabilities.

Each run writes structured metadata, evaluation metrics, and loss magnitudes so
the ablation can be analyzed reproducibly.

## Features

- End-to-end BERT teacher and TinyBERT student pipelines.
- Granular CE, Logit, Hidden, and Attention KD loss control.
- Full `2^3` factorial ablation over KD signal combinations.
- Evaluation for accuracy, calibration, and teacher-student agreement.
- Reusable analysis, markdown tables, and PNG/SVG visualizations.

## Getting Started

Run commands from the repository root.

### Environment Setup

```bash
conda create -n tinybert-xai python=3.12
conda activate tinybert-xai
pip install -r requirements.txt
```

`requirements.txt` installs PyTorch with CUDA 12.4 wheels, HuggingFace
Transformers/Datasets, the scientific Python stack, plotting libraries, and dev
tools. The default configuration uses bf16 autocast when CUDA is available.

Optional smoke test:

```bash
python scripts/00_smoke_test.py
```

### Teacher Fine-Tuning

Train the BERT teacher:

```bash
python scripts/01_train_teacher.py
```

Evaluate the saved teacher on dev/test and patch its metadata:

```bash
python scripts/01b_eval_teacher.py
```

Expected artifacts:

- `checkpoints/teachers/tweet_eval-sentiment/best.pt`
- `results/teachers/tweet_eval-sentiment/run_metadata.json`

### Student Distillation

Train one student condition:

```bash
python scripts/02_train_student.py kd_logit
```

Evaluate the saved student and patch its metadata:

```bash
python scripts/02b_eval_student.py kd_logit
```

Replace `kd_logit` with any condition listed in the experimental conditions
table below. KD conditions require the teacher checkpoint produced by the
teacher fine-tuning step.

Expected artifacts:

- `checkpoints/students/tweet_eval-sentiment/<condition>/best.pt`
- `results/students/tweet_eval-sentiment/<condition>/run_metadata.json`

### Analysis

Run the factorial analysis for the pilot dataset:

```bash
python scripts/06_analyze_factorial.py tweet_eval-sentiment
```

Generated artifacts:

- `results/analysis/student_ablation_table.md`
- `results/analysis/main_effects_table.md`
- `results/analysis/figures/condition_bars.{png,svg}`
- `results/analysis/figures/main_effects.{png,svg}`
- `results/analysis/figures/loss_magnitudes.{png,svg}`
- `results/analysis/figures/calibration.{png,svg}`

The script also prints a pipeline-validity checklist and a GO/NO-GO verdict for
scaling beyond the pilot dataset.

## Project Structure

```text
tinybert_xai/
  analysis/       Factorial loaders, effect math, tables, and plots
  eval/           Metrics and teacher-student analysis
  conditions.py   The 8 student ablation conditions
  config.py       Default experiment configuration
  losses.py       CE, logit KD, hidden KD, and attention KD losses
  projections.py  Hidden-state projection module
  teacher.py      Teacher training/evaluation pipeline
  student.py      Student training/evaluation pipeline

scripts/
  00_smoke_test.py
  01_train_teacher.py
  01b_eval_teacher.py
  02_train_student.py
  02b_eval_student.py
  06_analyze_factorial.py

tests/            Unit tests for losses, metrics, run logs, and factorial math
docs/             Notes, plans, and source project documents
reference/        Original TinyBERT reference code used for comparison only
results/          Run metadata and analysis outputs
```

## Experimental Conditions

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

## Results

Current TweetEval-sentiment pilot results:

- Teacher test macro-F1: `0.6870`.
- Best student: `kd_logit`, test macro-F1 `0.6631`.
- CE-only student: test macro-F1 `0.6592`.
- Student macro-F1 spread across all 8 conditions: `0.0198`.

The generated tables and plots are in `results/analysis/`. The pilot passes the
pipeline-validity gate, so the analysis verdict is GO for scaling the experiment
framework beyond the pilot dataset.

## Notes / Limitations

- The current pilot is single-seed. The observed `0.0198` student spread should
  not be treated as a statistically resolved per-signal effect.
- Post-softmax Attention KD appears near-inert in the pilot: the final attention
  loss magnitude averages about `0.00453`, much smaller than CE, logit, and
  hidden losses. This should be fixed or explicitly documented before larger
  multi-dataset runs.
- `reference/` contains the original TinyBERT authors' older codebase for
  comparison. The active implementation is the modern HuggingFace/PyTorch code
  under `tinybert_xai/`.
- Checkpoints are intentionally not tracked in git. Recreate them with the
  training scripts.

## Testing

```bash
ruff check
pytest tests/
```

`ruff` excludes `reference/` because that directory contains legacy upstream
code that is not part of the active implementation.

## Acknowledgements

This project builds on the TinyBERT distillation idea and uses HuggingFace
Transformers/Datasets for the modern implementation. It was developed as a final
project for KAIST CS50700 Deep Learning.
