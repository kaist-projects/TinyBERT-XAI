# TinyBERT-XAI

TinyBERT-XAI is a KAIST CS50700 Deep Learning final project that replicates and
extends TinyBERT-style task distillation with granular control over knowledge
distillation (KD) loss components.

The project fine-tunes a `bert-base-uncased` teacher, distills into
`huawei-noah/TinyBERT_General_4L_312D` students, and compares CE-only training
against Logit KD, Hidden KD, Attention KD, and all factorial combinations of
those three KD signals. Datasets are registered in a small registry and selected
with `--dataset`; the completed pilot is TweetEval sentiment
(`cardiffnlp/tweet_eval`, config `sentiment`), with IMDB and ANLI (sentence-pair)
also wired up.

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

## Datasets

The design targets 9 datasets across 4 task families. Each is selected on the
CLI by its registry key via `--dataset` (e.g. `python scripts/01_train_teacher.py
--dataset imdb`). The status column reflects what is wired up today; ✅ are
runnable now, ⬜ are planned (their `--dataset` keys are proposed, not yet
registered), and 🔒 needs gated access.

| Dataset (family) | `--dataset` | Source | Input | Status |
|---|---|---|---|---|
| TweetEval-sentiment (sentiment) | `tweet_eval-sentiment` | [cardiffnlp/tweet_eval](https://huggingface.co/datasets/cardiffnlp/tweet_eval) | single | ✅ |
| IMDB (sentiment) | `imdb` | [stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb) | single | ✅ |
| ANLI (NLI) | `anli` | [facebook/anli](https://huggingface.co/datasets/facebook/anli) | pair | ✅ |
| Davidson (hate speech) | `davidson` | [tdavidson/hate_speech_offensive](https://huggingface.co/datasets/tdavidson/hate_speech_offensive) | single | ✅ |
| DynaHate (hate speech) | `dynahate` | [bvidgen/Dynamically-Generated-Hate-Speech-Dataset](https://github.com/bvidgen/Dynamically-Generated-Hate-Speech-Dataset) | single | ✅ (manual CSV) |
| HatEval (hate speech) | `hateval` | [valeriobasile/HatEval](https://huggingface.co/datasets/valeriobasile/HatEval) | single | ✅ 🔒 (gated) |
| FEVER (NLI) | `fever` | [pietrolesci/nli_fever](https://huggingface.co/datasets/pietrolesci/nli_fever) | pair | ✅ |
| Aepli/VarDial-2023 (dialects) | `vardial` | [VarDial 2023](https://sites.google.com/view/vardial-2023) | single | ⬜ planned |
| Multi-VALUE (dialects) | `multivalue` | [SALT-NLP/multi-value](https://github.com/SALT-NLP/multi-value) | single | ⬜ planned |

Datasets with no official validation/test split are partitioned with a seed-42
stratified split (IMDB: dev only; Davidson: 80/10/10). FEVER's official test split
ships unlabeled, so its `dev` split is used for validation and a seed-42 stratified
test set is carved from train; train is also subsampled to 50K (seed-42, stratified)
to keep the 3-epoch budget comparable across datasets. HatEval is HF-gated — accept
the dataset terms and `huggingface-cli login` before running it; its config slug,
text column, and split names are best-effort assumptions to confirm on first load.
DynaHate is distributed as a GitHub CSV rather than on the Hub — download v0.2.3 and
save it as `data/dynahate/dynahate_v0.2.3.csv` (gitignored) before running it; its
official `split` column is used as-is.

## Features

- End-to-end BERT teacher and TinyBERT student pipelines.
- Granular CE, Logit, Hidden, and Attention KD loss control.
- Full `2^3` factorial ablation over KD signal combinations.
- Evaluation for accuracy, calibration, and teacher-student agreement.
- Reusable analysis, markdown reports, and PNG visualizations.

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

Train the BERT teacher (`--dataset` selects a registered dataset; default
`tweet_eval-sentiment`, also `imdb`, `anli`):

```bash
python scripts/01_train_teacher.py --dataset tweet_eval-sentiment
```

Evaluate the saved teacher on dev/test and patch its metadata (or pass `--eval`
to `01_train_teacher.py` to chain evaluation onto training in one pass):

```bash
python scripts/01b_eval_teacher.py
```

Expected artifacts:

- `checkpoints/teachers/tweet_eval-sentiment/best.pt`
- `results/teachers/tweet_eval-sentiment/run_metadata.json`

### Student Distillation

Train one student condition by toggling distillation signals with flags
(`--logit`, `--hidden`, `--attention`); no flags means the `ce_only` baseline.
`--dataset` selects the dataset (same choices as the teacher):

```bash
python scripts/02_train_student.py --dataset tweet_eval-sentiment --logit
```

Pass `--eval` to chain evaluation onto training in one pass, patching the run's
metadata with dev/test metrics (equivalent to running `02b_eval_student.py`
afterwards):

```bash
python scripts/02_train_student.py --logit --eval
```

Or evaluate a saved student separately (same flags select the run):

```bash
python scripts/02b_eval_student.py --logit
```

Combine flags for any of the 8 conditions in the experimental conditions table
below — e.g. `--logit --attention` is `kd_logit_attn`, `--logit --hidden
--attention` is `kd_full`. KD conditions require the teacher checkpoint produced
by the teacher fine-tuning step.

Expected artifacts:

- `checkpoints/students/<dataset>/<condition>/best.pt`
- `results/students/<dataset>/<condition>/run_metadata.json`

### Full Factorial Sweep

Run the teacher fine-tune (once) plus all 8 student conditions for one dataset
with a single command. The sweep invokes the per-run scripts as subprocesses for
clean per-run GPU memory isolation and is resumable — artifacts that already
exist are skipped unless `--force`:

```bash
python scripts/07_run_dataset.py --dataset imdb
python scripts/07_run_dataset.py --dataset anli --skip-teacher
```

### Analysis

Run the factorial analysis for a dataset (defaults to the pilot):

```bash
python scripts/06_analyze_factorial.py tweet_eval-sentiment
```

Artifacts are written per dataset under `results/analysis/<dataset>/`, so running
the analysis for another dataset never overwrites an existing report:

- `results/analysis/<dataset>/REPORT.md`
- `results/analysis/<dataset>/figures/condition_bars.png`
- `results/analysis/<dataset>/figures/main_effects.png`
- `results/analysis/<dataset>/figures/loss_magnitudes.png`
- `results/analysis/<dataset>/figures/calibration.png`

The script also prints a pipeline-validity checklist and a GO/NO-GO verdict for
scaling beyond the pilot dataset.

## Project Structure

```text
tinybert_xai/
  analysis/       Factorial loaders, effect math, tables, and plots
  eval/           Metrics and teacher-student analysis
  conditions.py   The 8 student ablation conditions
  datasets.py     Dataset registry + adapters (tweet_eval, imdb, anli)
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
  07_run_dataset.py     Teacher + all 8 conditions for one dataset
  _dataset_cli.py       Shared --dataset flag glue
  _student_cli.py       Shared signal-flag glue

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

The generated report and plots are in `results/analysis/tweet_eval-sentiment/`. The pilot passes the
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
