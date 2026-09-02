# Clio Apartment → Cubicle final benchmark

This lightweight bundle retains aggregate metrics, denominators, local full-report SHA-256 values, and claim boundaries only. It deliberately excludes Clio data/task YAML, model outputs, masks, point clouds, videos, and pair-level rows.

Key result on the 18-task Cubicle object-grounding protocol:
The grounding metric checks the predicted center against every official GT OBB for the task with `any(containment)`; the nearest-center GT is diagnostic only. The 2026-09-02 correction changed no final task decision or aggregate value.


- Q0 Top-1 strict center Acc@1: `27.78%`;
- frozen Q1F Top-5 + A2 with deterministic Q0 fallback strict center Acc@1: `38.89%` (`+11.11pp`);
- alignment-RMSE-padded difference: `+16.67pp`;
- final-cluster A2 pairwise F1 is `91.56%`, below A1's `93.47%`; Clio A2 is task-internal geometry+quality complete-link because semantic-embedding coverage is `0/181` (`0/35` on Apartment);
- fixed-confirmatory relation positives require both target and reference GT matches: strict/padded Acc@1 `11.41% / 48.32%`;
- negative rejection / reason-matched rejection / pair-grounded relation rejection: `98.66% / 67.79% / 44.97%`.

The Q1F object-grounding policy was frozen before Cubicle archive content was inspected and its manifest is checked by the validator. The association evaluator and the later relation protocol are reported as fixed-confirmatory, not as an untouched held-out benchmark. A2 is not described as multimodal semantic association. The directional benchmark still uses the uncalibrated engineering threshold `0.60`. FOUND-IT is outside this project's definition, not a pending implementation or comparison.

Validate the public summary without downloading Clio:

```bash
python -m scripts.validate_clio_final_summary \
  evidence/final-clio/benchmark_summary.json
```

`benchmark_summary.json` is automatically derived from six locally retained deterministic reports. `validation_report.json` checks scene roles, all reported deltas, the headline result, source path safety, and preservation of the negative A2 result.
