# D19 synthetic one-factor ablations

> Correctness ablation only; all current deltas are zero.

## Q2

| variant | factor | recall | Δ recall | SAM calls | Δ calls |
|---|---|---:|---:|---:|---:|
| base | — | 1.000000 | 0.000000 | 5 | 0 |
| retrieval_only | pose_novelty | 1.000000 | 0.000000 | 5 | 0 |
| no_gain_patience | gain_patience | 1.000000 | 0.000000 | 5 | 0 |

## A2

| variant | factor | F1 | Δ F1 | failures |
|---|---|---:|---:|---:|
| base | — | 0.000000 | 0.000000 | 4 |
| without_semantic | semantic | 0.000000 | 0.000000 | 4 |
| without_obb_shape | obb_shape | 0.000000 | 0.000000 | 4 |
| without_quality | quality | 0.000000 | 0.000000 | 4 |
| without_complete_link | complete_link | 0.000000 | 0.000000 | 4 |
