# Q×A development and synthetic tables

## Office-loop complete-cache development replay

> Engineering replay only; no labels or performance claim.

| combination | role | status | frames | observations | SAM calls | objects | stop |
|---|---|---:|---:|---:|---:|---:|---|
| Q0-vggt-slam-upstream-top1__A1-exact-class-spatial-gate | required | PASS | 1 | 4 | 1 | 0 | — |
| Q0-vggt-slam-upstream-top1__A2-evidence-aware-complete-link | required | PASS | 1 | 4 | 1 | 0 | — |
| Q1-fixed-topk-hybrid__A1-exact-class-spatial-gate | required | PASS | 4 | 10 | 4 | 1 | nonredundant_candidates_exhausted |
| Q1-fixed-topk-hybrid__A2-evidence-aware-complete-link | required | PASS | 4 | 10 | 4 | 1 | nonredundant_candidates_exhausted |
| Q2-gain-based-sequential-search__A2-evidence-aware-complete-link | required | PASS | 5 | 14 | 5 | 2 | max_budget_reached |
| Q2-gain-based-sequential-search__A1-exact-class-spatial-gate | diagnostic | PASS | 5 | 14 | 5 | 2 | max_budget_reached |

## Clio apartment complete-cache development replay

> Real GPU engineering replay; unlabelled and not a performance claim.

| combination | role | status | frames | observations | SAM calls | objects | stop |
|---|---|---:|---:|---:|---:|---:|---|
| Q0-vggt-slam-upstream-top1__A1-exact-class-spatial-gate | required | PASS | 1 | 0 | 1 | 0 | — |
| Q0-vggt-slam-upstream-top1__A2-evidence-aware-complete-link | required | PASS | 1 | 0 | 1 | 0 | — |
| Q1-fixed-topk-hybrid__A1-exact-class-spatial-gate | required | PASS | 5 | 1 | 5 | 0 | max_budget_reached |
| Q1-fixed-topk-hybrid__A2-evidence-aware-complete-link | required | PASS | 5 | 1 | 5 | 0 | max_budget_reached |
| Q2-gain-based-sequential-search__A2-evidence-aware-complete-link | required | PASS | 4 | 1 | 4 | 0 | two_consecutive_low_gain |
| Q2-gain-based-sequential-search__A1-exact-class-spatial-gate | diagnostic | PASS | 4 | 1 | 4 | 0 | two_consecutive_low_gain |

## Synthetic correctness

> Correctness fixture only; not held-out performance.

| combination | role | status | frames | observations | pairs | F1 |
|---|---|---:|---:|---:|---:|---:|
| Q0-vggt-slam-upstream-top1__A1-exact-class-spatial-gate | required | PASS | 1 | 1 | 0 | 0.000000 |
| Q0-vggt-slam-upstream-top1__A2-evidence-aware-complete-link | required | PASS | 1 | 1 | 0 | 0.000000 |
| Q1-fixed-topk-hybrid__A1-exact-class-spatial-gate | required | PASS | 5 | 5 | 10 | 0.000000 |
| Q1-fixed-topk-hybrid__A2-evidence-aware-complete-link | required | PASS | 5 | 5 | 10 | 0.000000 |
| Q2-gain-based-sequential-search__A2-evidence-aware-complete-link | required | PASS | 5 | 5 | 10 | 0.000000 |
| Q2-gain-based-sequential-search__A1-exact-class-spatial-gate | diagnostic | PASS | 5 | 5 | 10 | 0.000000 |
