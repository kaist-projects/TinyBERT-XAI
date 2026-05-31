# TinyBERT-XAI
![cover](docs/images/02-readme-cover.png)  

TinyBERT-XAI is a KAIST CS50700 Deep Learning final project that replicates and
extends TinyBERT-style task distillation with granular control over knowledge
distillation (KD) loss components.

- **Logit KD**: match teacher output distributions.
- **Hidden KD**: match selected teacher hidden states through learned projections.
- **Attention KD**: match teacher attention probabilities.

Each run writes structured metadata, evaluation metrics, and loss magnitudes so
the ablation can be analyzed reproducibly.

## Datasets

Datasets are selected with `--dataset <key>` and share the same teacher/student
pipeline.

| Dataset | `--dataset` | Task | Notes |
|---|---|---|---|
| [TweetEval-sentiment](https://huggingface.co/datasets/cardiffnlp/tweet_eval) | `tweet_eval-sentiment` | sentiment | official splits |
| [IMDB](https://huggingface.co/datasets/stanfordnlp/imdb) | `imdb` | sentiment | seed-42 dev split |
| [ANLI](https://huggingface.co/datasets/facebook/anli) | `anli` | NLI | sentence-pair input |
| [Davidson](https://huggingface.co/datasets/tdavidson/hate_speech_offensive) | `davidson` | hate speech | seed-42 80/10/10 split |
| [DynaHate](https://github.com/bvidgen/Dynamically-Generated-Hate-Speech-Dataset) | `dynahate` | hate speech | local CSV |
| [HatEval](https://huggingface.co/datasets/valeriobasile/HatEval) | `hateval` | hate speech | HF-gated |
| [FEVER](https://huggingface.co/datasets/pietrolesci/nli_fever) | `fever` | NLI | 50K train cap |
| [VarDial](https://huggingface.co/datasets/statworx/swiss-dialects) | `vardial` | dialect ID | seed-42 80/10/10 split |
| Multi-VALUE | `multivalue` | dialect ID | generated local CSV |

Local datasets are gitignored: save DynaHate to
`data/dynahate/dynahate_v0.2.3.csv`, and build Multi-VALUE with
`scripts/build_multivalue.py`. HatEval requires accepting the Hugging Face terms
and logging in before use.

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


### Teacher Fine-Tuning

Train the BERT teacher (`--dataset` selects a registered dataset; default
`tweet_eval-sentiment`, also `imdb`, `anli`):

```bash
python scripts/01_train_teacher.py --dataset tweet_eval-sentiment
```

Evaluate the saved teacher on dev/test and patch its metadata (or pass `--eval`
to `01_train_teacher.py` to chain evaluation onto training in one pass):

```bash
python scripts/01b_eval_teacher.py --dataset tweet_eval-sentiment
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
below. For example, `--logit --attention` is `kd_logit_attn`, `--logit --hidden
--attention` is `kd_full`. KD conditions require the teacher checkpoint produced
by the teacher fine-tuning step.

Expected artifacts:

- `checkpoints/students/<dataset>/<condition>/best.pt`
- `results/students/<dataset>/<condition>/run_metadata.json`

### Full Factorial Sweep

Run the teacher fine-tune (once) plus all 8 student conditions for one dataset
with a single command. The sweep invokes the per-run scripts as subprocesses for
clean per-run GPU memory isolation and is resumable. Artifacts that already
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
- `results/analysis/<dataset>/figures`

### Cross-Dataset Analysis

Once several datasets have completed sweeps, roll them up into the cross-task
presentation assets. This runs in two stages, written under
`results/analysis/cross_dataset/`.

**Stage 1, metadata only (no GPU).** Reads every
`results/students/<dataset>/<condition>/run_metadata.json` that is present:

```bash
python scripts/08_cross_dataset_analysis.py
```

- `figures/cross_task_macro_f1.png`, `figures/cross_task_delta.png`: the headline
  dataset × condition heatmaps (absolute macro-F1 and Δ from `ce_only`).
- `figures/confusion/<dataset>__<condition>.png`: per-condition confusion matrices.
- `tables/*.csv`: cross-task matrices + tidy calibration, teacher-student, and
  per-dataset factorial-effect tables.
- `TABLES.md`: a quick-read index of the matrices.

**Stage 2, representation + XAI artifacts (GPU, reloads checkpoints).** Runs
forward passes on a fixed test sample for every dataset that has both a teacher
and student checkpoint:

```bash
python scripts/08b_representation_analysis.py   # N=256 test sample
```

- `representation/layer_cka.csv`: linear CKA per mapped pair. The trained hidden
  projections were never checkpointed
- `representation/attention_kl.csv`: head-averaged KL(teacher ‖ student) of
  attention maps per mapped pair.
- `representation/attention/*.png`: teacher-vs-student attention heatmaps for
  representative examples (`ce_only` and `kd_full`, by correctness category).
- `figures/cka_mean.png`, `figures/efficiency.png`, `representation/efficiency.json`:
  mean-CKA heatmap and the one teacher-vs-student size/latency comparison.

The written interpretation (RQ1/RQ2 answers) lives in
`results/analysis/cross_dataset/CROSS_DATASET.md`.

## Project Structure

```text
tinybert_xai/
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
  00_smoke_test.py
  01_train_teacher.py
  01b_eval_teacher.py
  02_train_student.py
  02b_eval_student.py
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

## Acknowledgements

This project builds on the TinyBERT distillation idea and uses HuggingFace
Transformers/Datasets for the modern implementation. It was developed as a final
project for KAIST CS50700 Deep Learning.
