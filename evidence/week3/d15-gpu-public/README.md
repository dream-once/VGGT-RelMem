# Public GPU and D15/D15.5 evidence

This bundle is a portable, lightweight projection of the ignored local
`runs/gpu-acceptance-20260829` and D15.5 visualization products.

It contains:

- a sanitized GPU acceptance report with repository-relative references;
- the complete real D15 trace and its complete candidate cache;
- a persisted D15.5 validator report;
- a bounded overview thumbnail;
- an artifact manifest referencing the retained D15.5 audit and result tables.

The historical schema-0.1 field `observed_gain` is retained for compatibility.
Its canonical meaning is only `new_observation_count`: frame-scoped
observation IDs not previously seen by the trace. It is not object coverage,
spatial coverage, instance recall, or a performance improvement.

The MP4 and PLY remain outside Git. Their byte sizes and SHA-256 hashes are
retained in the visualization manifest and this bundle's artifact manifest.

Regenerate locally in the environment that provides OpenCV:

```bash
conda run --no-capture-output -p /root/autodl-tmp/envs/vggt_geom \
  python -m scripts.export_public_gpu_evidence
```

Validate on CPU:

```bash
python -m scripts.validate_public_gpu_evidence
```

This evidence is an office-loop engineering replay, not a held-out result.
