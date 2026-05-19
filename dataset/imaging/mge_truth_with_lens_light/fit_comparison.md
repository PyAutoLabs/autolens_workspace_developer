# Source-Science Fit Comparison

Dataset: `dataset/imaging/mge_truth_with_lens_light` (zero-point = 25.0)
Posterior draws per fit: 50

## Truth (from tracer)

- image-plane flux:    5910.3687
- source-plane flux:   167.6342
- magnification:       35.2575
- magnitude (zp=25):   19.4391

## sersic__sersic

max log likelihood: 4834.2009

| Quantity | Truth | MLE | MLE / truth | PDF median | PDF ±1σ | within 1σ? | within 3σ? | z-score |
|---|---|---|---|---|---|---|---|---|
| source flux | 167.6 | 202 | 1.205 | 202.4 | +2.252 / -3.674 | **NO** | **NO** | 11.97 |
| image flux | 5910 | 6175 | 1.045 | 6176 | +29.62 / -26.69 | **NO** | **NO** | 9.575 |
| magnification | 35.26 | 30.57 | 0.8672 | 30.53 | +0.4036 / -0.2781 | **NO** | **NO** | -14.17 |
| magnitude | 19.44 | 19.24 | — | 19.23 | +0.01989 / -0.01201 | **NO** | **NO** | -13.09 |


## mge_lens__sersic_source

max log likelihood: 4896.4298

| Quantity | Truth | MLE | MLE / truth | PDF median | PDF ±1σ | within 1σ? | within 3σ? | z-score |
|---|---|---|---|---|---|---|---|---|
| source flux | 167.6 | 256.2 | 1.529 | 256.3 | +6.035 / -5.16 | **NO** | **NO** | 15.7 |
| image flux | 5910 | 6681 | 1.13 | 6679 | +48.38 / -34.11 | **NO** | **NO** | 18.47 |
| magnification | 35.26 | 26.08 | 0.7396 | 26.11 | +0.3666 / -0.441 | **NO** | **NO** | -21.3 |
| magnitude | 19.44 | 18.98 | — | 18.98 | +0.02208 / -0.02527 | **NO** | **NO** | -19.26 |


## mge_lens__mge_source

max log likelihood: 5418.2181

| Quantity | Truth | MLE | MLE / truth | PDF median | PDF ±1σ | within 1σ? | within 3σ? | z-score |
|---|---|---|---|---|---|---|---|---|
| source flux | 167.6 | 1180 | 7.038 | 835.7 | +305.9 / -415.3 | **NO** | **NO** | 2.138 |
| image flux | 5910 | 9402 | 1.591 | 8487 | +963.6 / -1293 | **NO** | **NO** | 2.649 |
| magnification | 35.26 | 7.969 | 0.226 | 10.23 | +7.011 / -2.054 | **NO** | **NO** | -4.733 |
| magnitude | 19.44 | 17.32 | — | 17.69 | +0.746 / -0.3386 | **NO** | **NO** | -3.356 |

