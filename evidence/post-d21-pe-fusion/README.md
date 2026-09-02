# Post-D21 PE mask-crop representative-center experiment

This is an independent extension after the frozen D21 result. It does not
replace the D21 headline or claim an untouched held-out result.

The frozen method keeps PE Top-5 retrieval, SAM, lifting, A2 clustering, the
highest-confidence A2 object, and Q0 fallback unchanged. For each observation
inside the selected object, it averages two PE-Core-L14-336 query similarities:
a 15%-context crop and the same crop with non-mask pixels replaced by neutral
gray. The final 3D center is the center of the highest-scoring observation.
There are no learned parameters.

| Strict task Acc@1 | Apartment development | Cubicle fixed-confirmatory |
| --- | ---: | ---: |
| Current Q1F | 2/18 (11.11%) | 7/18 (38.89%) |
| Highest-quality observation | 3/18 (16.67%) | 7/18 (38.89%) |
| 3D medoid observation | 2/18 (11.11%) | 8/18 (44.44%) |
| PE semantic representative | 3/18 (16.67%) | 8/18 (44.44%) |
| Raw-observation oracle | 3/18 (16.67%) | 8/18 (44.44%) |

PE produces one strict win and zero regressions versus current Q1F in each
scene. On Apartment it ties the no-model highest-quality baseline, so there is
no development evidence for a PE-specific gain. On Cubicle it adds one strict
win over both current Q1F and the quality baseline while preserving current
RMSE-padded accuracy at 13/18. The rescued tasks are "move pile of clothes"
and "get tape measure".

These are 18-task, one-paired-win observations. No statistical-significance,
learned-ranker, SigLIP2, untouched-held-out, or general-superiority claim is
made.

Local GPU prediction, followed by CPU evaluation:

```bash
export PE_CHECKPOINT=/root/autodl-tmp/cache/huggingface/hub/models--facebook--PE-Core-L14-336/snapshots/bafb0f76541d399057e980a25947f67acec76575/PE-Core-L14-336.pt

python -m scripts.run_clio_pe_semantic_fusion \
  --scene-id apartment \
  --query-manifest configs/clio_apartment_queries.json \
  --run-root runs/clio-apartment-dev-v2-lc \
  --pe-checkpoint "$PE_CHECKPOINT" \
  --output runs/clio-pe-semantic-fusion-v1/apartment/prediction.json
python -m scripts.evaluate_clio_pe_semantic_fusion \
  --prediction runs/clio-pe-semantic-fusion-v1/apartment/prediction.json \
  --grounding-benchmark runs/clio-apartment-dev-v2-lc/grounding_benchmark.json \
  --output runs/clio-pe-semantic-fusion-v1/apartment/evaluation.json

python -m scripts.run_clio_pe_semantic_fusion \
  --scene-id cubicle \
  --query-manifest configs/clio_cubicle_queries.json \
  --run-root runs/clio-cubicle-heldout-v1 \
  --pe-checkpoint "$PE_CHECKPOINT" \
  --output runs/clio-pe-semantic-fusion-v1/cubicle/prediction.json
python -m scripts.evaluate_clio_pe_semantic_fusion \
  --prediction runs/clio-pe-semantic-fusion-v1/cubicle/prediction.json \
  --grounding-benchmark runs/clio-cubicle-heldout-v1/grounding_benchmark.json \
  --output runs/clio-pe-semantic-fusion-v1/cubicle/evaluation.json

python -m scripts.build_clio_pe_semantic_fusion_summary
```

The local predictions retain hashes for every source image and mask but remain
outside Git with the raw Clio artifacts. The tracked summary retains their
SHA-256 values and aggregate results only. A clean clone can validate it with:

```bash
python -m scripts.validate_clio_pe_semantic_fusion_summary
```
