# Factorial Main Effects

Metric: `test_macro_f1`

Positive estimates mean the factor or interaction increases the metric under
standard +/-1 factorial coding. Magnitudes are informational for this
single-seed pilot.

| Effect | Kind | Estimate | Absolute |
|---|---:|---:|---:|
| `logit` | main | -0.00041 | 0.00041 |
| `hidden` | main | -0.00596 | 0.00596 |
| `attention` | main | -0.00366 | 0.00366 |
| `logit x hidden` | 2-way | +0.00639 | 0.00639 |
| `logit x attention` | 2-way | -0.00469 | 0.00469 |
| `hidden x attention` | 2-way | +0.00539 | 0.00539 |
| `logit x hidden x attention` | 3-way | +0.00608 | 0.00608 |
