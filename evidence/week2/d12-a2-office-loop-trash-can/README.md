# D12 A2 association evidence

This JSON/Markdown-only development bundle replays A2 on the retained ten-observation D8 memory without loading VGGT, PE, or SAM 3.

- `prediction/` is label-free and contains the frozen D8 source, pair features/gates, complete-link merge trace, output ObjectMemory, and CPU manifest.
- `evaluation/` separately contains the copied manual labels, pairwise metrics, failure cases, and CPU manifest.
- `validation/` contains independent deterministic recompute reports for both bundles.

The frozen A2 defaults use semantic/exact-class compatibility, minimum observation quality `0.25`, center distance `0.15` or positive AABB overlap, and weights `0.25/0.25/0.20/0.15/0.15` for semantic/center/overlap/OBB-shape/quality. Clustering is complete-link, so every cross-cluster pair must pass before a merge.

On this small office-loop development fixture, A2 produces the same three clusters, one permanent object, and pairwise F1 `1.0` as frozen A1. This is parity, not an improvement claim or held-out performance. The bridge fix is demonstrated by synthetic tests because the real fixture does not contain an A1 bridge error. No thresholds were tuned after reading these labels.

Status: `CPU_COMPLETE / GPU_ACCEPTANCE_PENDING`. D12 does not require new visual inference; the GPU marker remains pending for the eventual end-to-end rerun only.
