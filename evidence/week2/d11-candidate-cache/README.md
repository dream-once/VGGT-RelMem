# D11 candidate-cache evidence

This CPU-only development replay freezes the eight-frame D5 candidate universe and reuses retained D6/D7 outcomes without running PE or SAM 3 again.

- `visual_memory_manifest.json`: frame/image hashes, geometry rows, camera pose features, PE provenance, and the explicit `not_retained` embedding state.
- `candidate_cache.json`: ranked candidates plus policy-independent outcomes and costs.
- `source_*.json`: canonical retained D5/D6/D7 inputs used for deterministic replay.
- `run_manifest.json`: D11 CPU completion and GPU-acceptance state.
- `validation.json`: independent replay report.

The real bundle contains eight candidates. Four retained frames have `available` outcomes; the other four are `unmaterialized`, not zero detections. The accepted status is therefore `PASS_WITH_UNMATERIALIZED_OUTCOMES`. A complete synthetic cache is exercised in `tests/test_candidate_cache.py`.

The bundle contains no embeddings, images, masks, point clouds, ground-truth labels, query answers, metrics, or policy traces. It is development evidence, not a held-out performance result. New PE/SAM inference remains `GPU_ACCEPTANCE_PENDING`.
