# Clio Apartment → Cubicle final benchmark

This lightweight bundle retains aggregate metrics, denominators, local full-report SHA-256 values, and claim boundaries only. It deliberately excludes Clio data/task YAML, model outputs, masks, point clouds, videos, and pair-level rows.

Key result on the 18-task Cubicle object-grounding protocol:

- Q0 Top-1 strict center Acc@1: `27.78%`;
- Q1 Top-5 + A2 strict center Acc@1: `38.89%` (`+11.11pp`);
- alignment-RMSE-padded difference: `+16.67pp`;
- A2 pairwise association F1 is `91.56%`, below A1's `93.85%`, so the grounding gain is not attributed to A2 alone;
- fixed-confirmatory relation positives: strict/padded Acc@1 `22.82% / 59.06%`;
- negative rejection / explicit relation-conflict rejection: `98.66% / 67.79%`.

The object-grounding policy was frozen before Cubicle archive content was inspected. The association evaluator and the later relation protocol are reported as fixed-confirmatory, not as an untouched held-out benchmark. The directional benchmark still uses the uncalibrated engineering threshold `0.60`. FOUND-IT is outside this project's definition, not a pending implementation or comparison.

Validate the public summary without downloading Clio:

```bash
python -m scripts.validate_clio_final_summary \
  evidence/final-clio/benchmark_summary.json
```

`benchmark_summary.json` is automatically derived from six locally retained deterministic reports. `validation_report.json` checks scene roles, all reported deltas, the headline result, source path safety, and preservation of the negative A2 result.
