# Factorial Analysis Report

Dataset: `dynahate`

## Artifact Summary

- Teacher metadata: `results/teachers/dynahate/run_metadata.json`
- Student metadata: `results/students/dynahate/*/run_metadata.json`
- Report: `results/analysis/dynahate/REPORT.md`
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

- Teacher test macro-F1: `0.7779`.
- Best student: `kd_logit_attn` with test macro-F1 `0.7405`.
- CE-only student test macro-F1: `0.7364`.
- Student macro-F1 spread across conditions: `0.0398`.
- Mean final attention-loss magnitude: `0.00368`.

The best student is `kd_logit_attn` (test macro-F1 `0.7405`), but with a single seed the factorial effects
below should be read as pipeline diagnostics and descriptive statistics, not
resolved causal estimates.

## Student Ablation Table

Dataset: `dynahate`

Source files:
`results/teachers/dynahate/run_metadata.json` and
`results/students/dynahate/*/run_metadata.json`

Primary metric: test macro-F1. `Delta` is test macro-F1 relative to `ce_only`.
Rows are ordered by test macro-F1 descending.
Bold marks the best value in each metric column: higher is better for F1,
accuracy, and agreement; lower is better for ECE.

| Condition | Logit | Hidden | Attention | Test Macro-F1 | Delta | Test Acc. | Test ECE | Top-1 Agree |
|---|:---:|:---:|:---:|---:|---:|---:|---:|---:|
| `teacher` | N/A | N/A | N/A | **0.7779** | **+0.0415** | **0.7811** | 0.0667 | N/A |
| `kd_logit_attn` | Y |  | Y | 0.7405 | +0.0041 | 0.7481 | 0.0592 | **0.8437** |
| `kd_logit` | Y |  |  | 0.7377 | +0.0013 | 0.7444 | 0.0589 | 0.8381 |
| `ce_only` |  |  |  | 0.7364 | +0.0000 | 0.7432 | 0.0589 | 0.8306 |
| `kd_attn` |  |  | Y | 0.7361 | -0.0003 | 0.7451 | 0.0618 | 0.8267 |
| `kd_full` | Y | Y | Y | 0.7223 | -0.0141 | 0.7291 | **0.0386** | 0.8160 |
| `kd_hidden_attn` |  | Y | Y | 0.7120 | -0.0244 | 0.7228 | 0.0666 | 0.7990 |
| `kd_logit_hidden` | Y | Y |  | 0.7022 | -0.0342 | 0.7112 | 0.0494 | 0.8097 |
| `kd_hidden` |  | Y |  | 0.7007 | -0.0357 | 0.7112 | 0.0785 | 0.7985 |

Best student test macro-F1 is `kd_logit_attn` at 0.7405, +0.0041 over `ce_only`.
The teacher reference is higher at 0.7779.

## Factorial Effects

Metric: `test_macro_f1`

Positive estimates mean the factor or interaction increases the metric under
standard +/-1 factorial coding. Magnitudes are informational for this
single-seed run.

| Effect | Kind | Estimate | Absolute |
|---|---:|---:|---:|
| `logit` | main | +0.00438 | 0.00438 |
| `hidden` | main | -0.02838 | 0.02838 |
| `attention` | main | +0.00848 | 0.00848 |
| `logit x hidden` | 2-way | +0.00149 | 0.00149 |
| `logit x attention` | 2-way | +0.00299 | 0.00299 |
| `hidden x attention` | 2-way | +0.00724 | 0.00724 |
| `logit x hidden x attention` | 3-way | +0.00142 | 0.00142 |

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
