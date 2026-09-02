# D21 final result card

Status: `GPU_AND_CPU_COMPLETE_WITH_EXTERNAL_PACKAGING_GAPS`.

This lightweight bundle freezes the final source/CPU result boundary. It is
rebuilt from the D16 and D20 canonical JSON inputs; no result number is entered
by hand.

Files:

- `result_card.json`: tracked evidence/config hashes, sample sizes, budgets,
  validation status, scope, and explicit external gaps for all nine retained
  results, including Clio apartment GPU development acceptance;
- `result_card.md`: human-readable table generated from the JSON card;
- `claim_audit.json`: line-level README audit for official/reproduction/
  improvement/navigation/superiority/SOTA terminology;
- `validation.json`: independent rebuild and source-reference validation;
- `docs/PROJECT_PRESENTATION.md`: hash-pinned current resume wording, result
  boundaries, two resume bullets, an approximately 3-minute demo script and 10 interview Q&As.
  The actual narrated recording remains `DEMO_RECORDING_PENDING`; a release tag
  remains pending until the corrected files are explicitly committed.

Rebuild:

```bash
python -m scripts.build_d21_result_card
python -m scripts.validate_d21
```

The evidence contains JSON/Markdown only. It includes a 24/24 Clio candidate
cache plus GPU counts/hashes, but no Clio images, raw task labels, weights,
embeddings, masks, point clouds or videos, and no unsubstantiated held-out or
performance-improvement numbers.
