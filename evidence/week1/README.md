# Week 1 D4–D8 evidence

This 12 MiB snapshot publishes lightweight, inspectable evidence from the real
office-loop runs without committing model weights, the source dataset, or the
multi-megabyte D3 geometry NPZ.

## Contents

- `d4-single-view/`: controlled `B0-official` versus
  `B1-robust-single-view` result, preprocessing record, run manifest, and
  preview.
- `d5-multiview/`: six real PE hybrid-query results and manifests; the
  trash-can directory also includes the Top-K preview.
- `d6-multiview/`: six real SAM 3/Robust3DLifter result bundles and per-frame
  previews. Masks and point arrays remain out of this lightweight D6 snapshot.
- `runs/office-loop-mv-d7-trash-can/`: the complete self-contained D7 cache,
  including masks, sampled points, previews, inventory, and the 40-second
  video.
- `runs/office-loop-mv-d8-trash-can/`: the portable D8 ObjectMemory bundle.
  Its D7 source reference is relative, so validation still works after the
  evidence directory is moved.
- `validation/`: saved independent validator reports, including the
  query-specific same-frame-pair gate.

## Verified results

| Stage | Result |
| --- | --- |
| D4 | PASS; 6 official-B0 and 4 robust-B1 objects from shared PE/SAM/VGGT inputs |
| D5 | PASS; all six real interval-sequence runs use hybrid redundancy suppression |
| D6 | PASS for trash can; 15 SAM instances, 10 lifted observations, 5 explicit rejections |
| D7 | PASS; 4 frames, 10 observations, 19,062 finite sampled points, 40.0 s video, motion ratio 0.975 |
| D8 | PASS; 10 pending observations, 0 permanent objects, 0 decisions, exact JSON round trip |

The query-specific gate uses only frames that produced lifted observations and
requires one and the same frame pair to satisfy both `translation >= 0.5`
unscaled reconstruction units and `rotation >= 3 degrees`:

| Query | Classification |
| --- | --- |
| trash can | TRUE_MULTIVIEW; three passing pairs |
| poster | MULTIFRAME_ONLY |
| blue recycling bin | MULTIFRAME_ONLY |
| printer | MULTIFRAME_ONLY |
| dog | NEGATIVE_CONTROL; zero SAM and 3D evidence |
| bed | NEGATIVE_CONTROL; zero SAM and 3D evidence |

Dog and bed are intentional absent-object controls for this office scene. They
test whether the pipeline stays at zero evidence instead of hallucinating an
object; they are not expected positive queries.

## Artifact hashes

- D7 video:
  `aa924ff9913fcca8475b16209a950004f74f20b6774d3903a9a9756eafae0dea`
- D7 observations:
  `1a1f46481b2dd7bfb86f82c09dea1b3fe666823823f637e1e47d53369a5139a1`
- D8 ObjectMemory:
  `cdf1eccd7dabcae658f927ae122f5dcec77c220b0c4338f220adc88702411879`
- D8 result:
  `6a83fafd85c646fb67aa449912012bb6f29ad1e3f276c44fe56880800c4a2382`

## Model-free reproduction

From the repository root, D8 validates with the base environment:

```bash
python -m scripts.validate_d8_memory \
  evidence/week1/runs/office-loop-mv-d8-trash-can
```

D7 video probing additionally needs OpenCV, already present in the project
geometry environment:

```bash
/root/autodl-tmp/envs/vggt_geom/bin/python \
  -m scripts.validate_d7_cache \
  evidence/week1/runs/office-loop-mv-d7-trash-can
```

D4–D6 reports were generated against the complete ignored local run directories
because their lightweight public snapshots intentionally omit large or repeated
mask/point inputs. The saved reports and all result manifests preserve the
measured status and provenance.
