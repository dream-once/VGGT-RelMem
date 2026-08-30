# Clio apartment GPU development acceptance

This lightweight bundle records a real RTX 4090 development replay on the
public Clio `apartment` RGB/task-metadata subset. It is not held-out evidence.
The `cubicle` scene was neither downloaded nor evaluated.

Tracked contents:

- `d16_acquisition_receipt.json`: public-source, split, subset and licence boundary;
- `candidate_cache.json`: 24/24 materialized outcomes, no GT or metrics;
- `d18-qxa/`: label-free Q0/Q1/Q2 × A1/A2 development replay;
- `d19-ablations/`: unlabelled engineering ablations;
- `gpu_acceptance_report.json`: counts, hashes, multiview spans and failures;
- validator reports for D16, D18, D19 and the GPU acceptance bundle.

Observed result: 24 geometry/candidate frames, 3 lifted `pillow` observations in
two query-evidence frames, and zero promoted permanent objects. Q0/Q1/Q2 yield
0/1/1 observations. This is a reproducible failure-oriented engineering result,
not Grounding Acc@1, held-out performance or a superiority claim.

Public validation:

```bash
python -m scripts.validate_clio_gpu_acceptance \
  evidence/week4/clio-apartment-gpu/gpu_acceptance_report.json
python -m scripts.validate_d18
python -m scripts.validate_d19
python -m scripts.validate_d20
python -m scripts.validate_d21
```

Raw Clio data, YAML labels, weights, masks, points, previews and video remain
local and are not redistributed. The upstream code licence is not assumed to
cover dataset redistribution.
