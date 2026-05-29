# Student Ablation Results

Dataset: `tweet_eval-sentiment`

Source files:
`results/teachers/tweet_eval-sentiment/run_metadata.json` and
`results/students/tweet_eval-sentiment/*/run_metadata.json`

Primary metric: test macro-F1. `Delta` is test macro-F1 relative to `ce_only`.
Rows are ordered by test macro-F1 descending.
Bold marks the best value in each metric column: higher is better for F1,
accuracy, and agreement; lower is better for ECE.

| Condition | Logit | Hidden | Attention | Test Macro-F1 | Delta | Test Acc. | Test ECE | Top-1 Agree |
|---|:---:|:---:|:---:|---:|---:|---:|---:|---:|
| `teacher` | N/A | N/A | N/A | **0.6870** | **+0.0278** | **0.6875** | 0.0919 | N/A |
| `kd_logit` | Y |  |  | 0.6631 | +0.0040 | 0.6653 | **0.0506** | **0.7988** |
| `kd_attn` |  |  | Y | 0.6609 | +0.0017 | 0.6596 | 0.0741 | 0.7898 |
| `ce_only` |  |  |  | 0.6592 | +0.0000 | 0.6576 | 0.0789 | 0.7865 |
| `kd_full` | Y | Y | Y | 0.6552 | -0.0039 | 0.6565 | 0.0508 | 0.7910 |
| `kd_logit_hidden` | Y | Y |  | 0.6521 | -0.0071 | 0.6536 | 0.0539 | 0.7905 |
| `kd_hidden_attn` |  | Y | Y | 0.6478 | -0.0113 | 0.6470 | 0.0929 | 0.7774 |
| `kd_hidden` |  | Y |  | 0.6475 | -0.0117 | 0.6468 | 0.0895 | 0.7768 |
| `kd_logit_attn` | Y |  | Y | 0.6433 | -0.0159 | 0.6419 | 0.1077 | 0.7773 |

Best student test macro-F1 is `kd_logit` at 0.6631, +0.0040 over `ce_only`.
The teacher reference is higher at 0.6870.
