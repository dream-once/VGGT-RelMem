# D14 Fixed Top-K development replay

- Completion: `CPU_COMPLETE / GPU_ACCEPTANCE_PENDING`.
- Policy: `Q1-fixed-topk-hybrid`, budgets `K=1/3/5`, frame gap `2`, camera distance `0.15`, view angle `3°`.
- Selection reads ranked frame metadata only; cached SAM/lifting outcomes are revealed after selection. Evaluator labels are stored separately.
- Real partial cache: selected frame counts `1/3/4`; `K=5` ends with `nonredundant_candidates_exhausted`, not a fabricated fifth outcome. All four selected frames are available.
- Real single-query development recall is `0.667/1.0/1.0`; this is replay evidence, not held-out performance or a new GPU experiment.
- Complete synthetic cache: selected frame counts `1/3/5`, allowing the complete-path validator to run without a GPU.
- `validation.json` must report `PASS`; only JSON/Markdown evidence is retained.
