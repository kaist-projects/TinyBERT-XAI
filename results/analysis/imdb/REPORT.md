# Factorial Analysis Report

Dataset: `imdb`

## Artifact Summary

- Teacher metadata: `results/teachers/imdb/run_metadata.json`
- Student metadata: `results/students/imdb/*/run_metadata.json`
- Report: `results/analysis/imdb/REPORT.md`
- Figures: `figures/`

## Validity Checklist

| Check | Status | Detail |
|---|:---:|---|
| all 8 conditions present and valid | PASS | all 8 condition metadata files are present and valid |
| epochs completed | PASS | all runs completed configured epochs or documented early-stop |
| finite metrics/losses | PASS | all required metrics and active losses are finite |
| teacher forward sane | PASS | top1_agreement is present and above random for every KD condition |
| metric ranges | PASS | F1/accuracy/agreement/ECE values are within [0, 1] |
| artifacts written | PASS | 4 PNG figures and 1 markdown report written |

## Key Results

- Teacher test macro-F1: `0.8885`.
- Best student: `kd_attn` with test macro-F1 `0.8519`.
- CE-only student test macro-F1: `0.8502`.
- Student macro-F1 spread across conditions: `0.0135`.
- Mean final attention-loss magnitude: `0.00112`.

The best student is `kd_attn` (test macro-F1 `0.8519`), but with a single seed the factorial effects
below should be read as pipeline diagnostics and descriptive statistics, not
resolved causal estimates.

## Student Ablation Table

Dataset: `imdb`

Source files:
`results/teachers/imdb/run_metadata.json` and
`results/students/imdb/*/run_metadata.json`

Primary metric: test macro-F1. `Delta` is test macro-F1 relative to `ce_only`.
Rows are ordered by test macro-F1 descending.
Bold marks the best value in each metric column: higher is better for F1,
accuracy, and agreement; lower is better for ECE.

| Condition | Logit | Hidden | Attention | Test Macro-F1 | Delta | Test Acc. | Test ECE | Top-1 Agree |
|---|:---:|:---:|:---:|---:|---:|---:|---:|---:|
| `teacher` | N/A | N/A | N/A | **0.8885** | **+0.0383** | **0.8885** | 0.0433 | N/A |
| `kd_attn` |  |  | Y | 0.8519 | +0.0017 | 0.8522 | 0.0381 | 0.8950 |
| `kd_logit` | Y |  |  | 0.8508 | +0.0006 | 0.8511 | 0.0402 | 0.8973 |
| `kd_logit_hidden` | Y | Y |  | 0.8507 | +0.0005 | 0.8508 | 0.0375 | 0.8965 |
| `kd_logit_attn` | Y |  | Y | 0.8506 | +0.0005 | 0.8509 | 0.0405 | **0.8976** |
| `kd_full` | Y | Y | Y | 0.8505 | +0.0003 | 0.8507 | 0.0383 | 0.8970 |
| `ce_only` |  |  |  | 0.8502 | +0.0000 | 0.8505 | 0.0397 | 0.8931 |
| `kd_hidden_attn` |  | Y | Y | 0.8387 | -0.0115 | 0.8391 | 0.0347 | 0.8820 |
| `kd_hidden` |  | Y |  | 0.8384 | -0.0117 | 0.8388 | **0.0343** | 0.8810 |

Best student test macro-F1 is `kd_attn` at 0.8519, +0.0017 over `ce_only`.
The teacher reference is higher at 0.8885.

## Factorial Effects

Metric: `test_macro_f1`

Positive estimates mean the factor or interaction increases the metric under
standard +/-1 factorial coding. Magnitudes are informational for this
single-seed run.

| Effect | Kind | Estimate | Absolute |
|---|---:|---:|---:|
| `logit` | main | +0.00583 | 0.00583 |
| `hidden` | main | -0.00629 | 0.00629 |
| `attention` | main | +0.00042 | 0.00042 |
| `logit x hidden` | 2-way | +0.00616 | 0.00616 |
| `logit x attention` | 2-way | -0.00058 | 0.00058 |
| `hidden x attention` | 2-way | -0.00037 | 0.00037 |
| `logit x hidden x attention` | 3-way | +0.00036 | 0.00036 |

## Attention-Loss Caveat

Attention KD used post-softmax attention probabilities in this run. Its
final loss magnitude is near-inert compared with CE, logit, and hidden
losses, so the attention factor was only weakly applied. Fix this signal or
explicitly document the caveat before scaling the experiment.

## Figures

### Condition Bars

![Condition Bars](figures/condition_bars.png)

### Main Effects

![Main Effects](figures/main_effects.png)

### Loss Magnitudes

![Loss Magnitudes](figures/loss_magnitudes.png)

### Calibration

![Calibration](figures/calibration.png)
