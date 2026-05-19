# Source-Science Fit Comparison — No Lens Light

Dataset: `dataset/imaging/mge_truth_no_lens_light` (zero-point = 25.0)
Posterior draws per fit: 50

## Truth (from tracer)

- image-plane flux:    5910.3687
- source-plane flux:   167.6342
- magnification:       35.2575
- magnitude (zp=25):   19.4391

## sersic_source

max log likelihood: 5929.9261

| Quantity | Truth | MLE | MLE / truth | PDF median | PDF ±1σ | within 1σ? | within 3σ? | z-score |
|---|---|---|---|---|---|---|---|---|
| source flux | 167.6 | 173.2 | 1.033 | 173.7 | +0.952 / -0.8164 | **NO** | **NO** | 6.847 |
| image flux | 5910 | 5791 | 0.9798 | 5802 | +15.85 / -11.4 | **NO** | **NO** | -7.611 |
| magnification | 35.26 | 33.43 | 0.9481 | 33.4 | +0.1643 / -0.1578 | **NO** | **NO** | -11.86 |
| magnitude | 19.44 | 19.4 | — | 19.4 | +0.005116 / -0.005935 | **NO** | **NO** | -6.97 |


## mge_source

max log likelihood: 6603.5477

| Quantity | Truth | MLE | MLE / truth | PDF median | PDF ±1σ | within 1σ? | within 3σ? | z-score |
|---|---|---|---|---|---|---|---|---|
| source flux | 167.6 | 197.4 | 1.178 | 192.4 | +5.342 / -2.465 | **NO** | **NO** | 5.867 |
| image flux | 5910 | 6589 | 1.115 | 6469 | +178.6 / -138.2 | **NO** | **NO** | 3.723 |
| magnification | 35.26 | 33.38 | 0.9467 | 33.52 | +0.353 / -0.2629 | **NO** | **NO** | -5.357 |
| magnitude | 19.44 | 19.26 | — | 19.29 | +0.014 / -0.02974 | **NO** | **NO** | -6.317 |

