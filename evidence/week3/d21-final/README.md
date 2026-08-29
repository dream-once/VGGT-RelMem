# D21 final result card

Status: `CPU_COMPLETE`.

This lightweight bundle freezes the final source/CPU result boundary. It is
rebuilt from the D16 and D20 canonical JSON inputs; no result number is entered
by hand.

Files:

- `result_card.json`: tracked evidence/config hashes, sample sizes, budgets,
  validation status, scope, and explicit external gaps for every retained
  result;
- `result_card.md`: human-readable table generated from the JSON card;
- `claim_audit.json`: line-level README audit for official/reproduction/
  improvement/navigation/superiority/SOTA terminology;
- `validation.json`: independent rebuild and source-reference validation.

Rebuild:

```bash
python -m scripts.build_d21_result_card
python -m scripts.validate_d21
```

The evidence contains JSON/Markdown only. It does not contain Clio data,
weights, embeddings, masks, point clouds, images, videos, GT labels, or
unsubstantiated held-out/performance-improvement numbers.
