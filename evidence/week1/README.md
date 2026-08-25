# Week 1 D4–D8 JSON evidence

This directory intentionally keeps only small, inspectable JSON/text evidence.
Model weights, source images, geometry archives, masks, point arrays, previews,
and videos are reproducible local artifacts and are excluded from Git.

## Contents

- `d4-single-view/`: controlled B0/B1 result, preprocessing metadata, and run manifest.
- `d5-multiview/`: six PE retrieval results, Top-K selections, and run manifests.
- `d6-multiview/`: six SAM 3/lifting summaries, selections, and run manifests.
- `runs/office-loop-mv-d7-trash-can/`: D7 observation JSON, inventory, and run manifest only.
- `runs/office-loop-mv-d8-trash-can/`: portable D8 ObjectMemory JSON bundle.
- `validation/`: saved independent validator and query-specific reports.

## Verified results

| Stage | Saved result |
| --- | --- |
| D4 | PASS; 6 official-B0 and 4 robust-B1 objects from shared inputs |
| D5 | PASS; six interval-sequence runs used hybrid redundancy suppression |
| D6 | PASS for trash can; 15 SAM instances, 10 lifted observations, 5 rejections |
| D7 | Saved report records 4 frames, 10 observations, 19,062 sampled points and dynamic video validation |
| D8 | PASS; 10 pending observations, 0 objects/decisions, exact JSON round trip |

The D7 mask arrays, sampled point files, previews, and MP4 are deliberately not
published. Consequently this lightweight D7 directory is an audit snapshot,
not a self-contained input for `validate_d7_cache`. The ignored local run can
be regenerated with the command in the project README. D8 remains directly
validatable because its relative source and SHA-256 refer to the retained D7
`observations.json`.

## Query-specific classifications

| Query | Classification |
| --- | --- |
| trash can | TRUE_MULTIVIEW; three passing pairs |
| poster | MULTIFRAME_ONLY |
| blue recycling bin | MULTIFRAME_ONLY |
| printer | MULTIFRAME_ONLY |
| dog | NEGATIVE_CONTROL; zero evidence |
| bed | NEGATIVE_CONTROL; zero evidence |

`dog` and `bed` are absent-object controls, not expected positives.

## Retained artifact hashes

- D7 observations: `1a1f46481b2dd7bfb86f82c09dea1b3fe666823823f637e1e47d53369a5139a1`
- D8 ObjectMemory: `cdf1eccd7dabcae658f927ae122f5dcec77c220b0c4338f220adc88702411879`
- D8 result: `6a83fafd85c646fb67aa449912012bb6f29ad1e3f276c44fe56880800c4a2382`

## Model-free validation

From the repository root:

```bash
python -m scripts.validate_d8_memory \
  evidence/week1/runs/office-loop-mv-d8-trash-can
```

D4-D7 reports preserve measured status and provenance. Full visual or dense
artifacts must be regenerated locally and should not be committed.
