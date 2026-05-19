# Source-Science Fit Comparison

Dataset: `dataset/imaging/simple` (zero-point = 25.0)
Posterior draws per fit: 50

## Truth (from tracer)

- image-plane flux:    5473.9631
- source-plane flux:   156.7253
- magnification:       34.9271
- magnitude (zp=25):   19.5122

## sersic__sersic

max log likelihood: 5381.5253

| Quantity | Truth | MLE | MLE / truth | PDF median | PDF ±1σ | within 1σ? | within 3σ? | z-score |
|---|---|---|---|---|---|---|---|---|
| source flux | 156.7 | 165.9 | 1.059 | 166.7 | +1.78 / -1.863 | **NO** | **NO** | 5.539 |
| image flux | 5474 | 5543 | 1.013 | 5550 | +27.03 / -17.11 | **NO** | **NO** | 3.278 |
| magnification | 34.93 | 33.4 | 0.9564 | 33.34 | +0.2432 / -0.266 | **NO** | **NO** | -5.452 |
| magnitude | 19.51 | 19.45 | — | 19.45 | +0.0122 / -0.01153 | **NO** | **NO** | -5.704 |


## mge_lens__sersic_source

max log likelihood: 5395.2871

| Quantity | Truth | MLE | MLE / truth | PDF median | PDF ±1σ | within 1σ? | within 3σ? | z-score |
|---|---|---|---|---|---|---|---|---|
| source flux | 156.7 | 165.6 | 1.056 | 166 | +3.844 / -3.413 | **NO** | **NO** | 2.604 |
| image flux | 5474 | 5546 | 1.013 | 5539 | +55.37 / -31.23 | **NO** | yes | 1.321 |
| magnification | 34.93 | 33.5 | 0.9591 | 33.38 | +0.5239 / -0.4294 | **NO** | **NO** | -3.432 |
| magnitude | 19.51 | 19.45 | — | 19.45 | +0.02256 / -0.02486 | **NO** | **NO** | -2.694 |


## mge_lens__mge_source

max log likelihood: 5412.8838

| Quantity | Truth | MLE | MLE / truth | PDF median | PDF ±1σ | within 1σ? | within 3σ? | z-score |
|---|---|---|---|---|---|---|---|---|
| source flux | 156.7 | 297.8 | 1.9 | 318.5 | +181.8 / -92.41 | **NO** | **NO** | 1.283 |
| image flux | 5474 | 6071 | 1.109 | 6240 | +613.5 / -362.6 | **NO** | **NO** | 1.69 |
| magnification | 34.93 | 20.38 | 0.5836 | 19.64 | +6.332 / -6.149 | **NO** | **NO** | -2.849 |
| magnitude | 19.51 | 18.82 | — | 18.74 | +0.3719 / -0.4904 | **NO** | **NO** | -2.034 |

