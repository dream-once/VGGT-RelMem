# D15 gain-based sequential search

- Completion: `CPU_COMPLETE / GPU_ACCEPTANCE_PENDING`.
- Policy: first step retrieval-only; later score is `0.65 × min-max retrieval + 0.35 × pose novelty`.
- Pose novelty is the mean of translation novelty clipped at `0.15 m` and view-angle novelty clipped at `3°`.
- Real partial-cache trace selects `frame_0001/0071/0041`, then selects unmaterialized `frame_0061` and immediately returns `BLOCKED_MISSING_OUTCOME`. It does not skip ahead and contains no performance claim.
- Complete synthetic trace runs five steps and stops at `max_budget_reached`; its Q0/Q1/Q2 file contains engineering counts only, not accuracy or held-out results.
- Observed gain means newly revealed 3D observation IDs, not physical-instance truth or exact frustum coverage.
- `validation.json` must report `PASS`; evidence is JSON/Markdown only.
