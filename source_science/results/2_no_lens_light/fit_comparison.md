# Source-Science Fit Comparison — No Lens Light

Dataset: `dataset/imaging/no_lens_light` (zero-point = 25.0)
Posterior draws per fit: 50

## Truth (from tracer)

- image-plane flux:    5473.9631
- source-plane flux:   156.7253
- magnification:       34.9271
- magnitude (zp=25):   19.5122

## sersic_source

max log likelihood: 6670.5063

| Quantity | Truth | MLE | MLE / truth | PDF median | PDF ±1σ | within 1σ? | within 3σ? | z-score |
|---|---|---|---|---|---|---|---|---|
| source flux | 156.7 | 163.1 | 1.041 | 162.9 | +1.033 / -1.144 | **NO** | **NO** | 5.162 |
| image flux | 5474 | 5475 | 1 | 5478 | +12.8 / -12.19 | yes | yes | 0.2665 |
| magnification | 34.93 | 33.56 | 0.9608 | 33.67 | +0.1572 / -0.2455 | **NO** | **NO** | -5.809 |
| magnitude | 19.51 | 19.47 | — | 19.47 | +0.007651 / -0.006865 | **NO** | **NO** | -5.271 |


## mge_source

max log likelihood: 6691.1136

| Quantity | Truth | MLE | MLE / truth | PDF median | PDF ±1σ | within 1σ? | within 3σ? | z-score |
|---|---|---|---|---|---|---|---|---|
| source flux | 156.7 | 162 | 1.034 | 162.8 | +1.012 / -1.351 | **NO** | **NO** | 5.069 |
| image flux | 5474 | 5480 | 1.001 | 5478 | +4.614 / -5.776 | yes | yes | 0.6388 |
| magnification | 34.93 | 33.82 | 0.9683 | 33.66 | +0.2854 / -0.2566 | **NO** | **NO** | -5.019 |
| magnitude | 19.51 | 19.48 | — | 19.47 | +0.00905 / -0.006729 | **NO** | **NO** | -5.159 |

