# VGGT-RelMem project memory

This tracked file is the recoverable handoff snapshot for disposable cloud instances. Update it in every user-authorized GitHub publication. Never put credentials, GitHub device codes, tokens, private keys, cookies, authentication files, or private dataset contents here.

## Current snapshot

- Updated: 2026-08-25 publication session
- Public repository: `dream-once/VGGT-RelMem`
- Publication target and pre-publication HEAD: `main` at `7b646deda5e8a01d128cb95f274f97316d01f53c`
- Current milestone: document D1 through D8 are complete; the D4/D6 controlled-baseline correction and a true-multiview evidence gate are also complete.
- Next milestone: D9 same-class cross-frame association with an explainable 3D center-distance/overlap gate and pairwise evaluation.
- Publication state: the D4/D6 correction, true-multiview checks, D8 source, tests, documentation, memory, and baseline are included in the publication commit containing this entry. Environments, weights, datasets, upstream checkouts, and generated run artifacts remain ignored and local-only.

## Document-day progress

| Day | State | Evidence / remaining work |
| --- | --- | --- |
| D1 | Complete | All five upstream sources are pinned; disk/GPU checked; local PE and ModelScope SAM 3 checkpoints are verified. |
| D2 | Complete | `vggt_geom` and Python 3.12 `open_vocab` are isolated; geometry/mask schemas, adapters, and smoke tests pass. |
| D3 | Complete | Two independent 8-frame `office_loop` geometry runs passed validation and produced byte-identical geometry plus run manifests. |
| D4 | Complete | Historical output is relabelled legacy B1; controlled `B0-official` and `B1-robust-single-view` now share the exact PE/SAM/VGGT inputs and differ only in lifting. |
| D5 | Complete | Real PE Top-K retrieval exported deterministic K=1/3/5 results with temporal redundancy suppression and B0-compatible top-1. |
| D6 | Complete | Controlled B2 preprocessing removes post-SAM mask resizing; four selected frames produced 21 SAM instances, 15 valid observations, and six explicit rejections. |
| D7 | Complete | `ObjectObservation` schema 1.0 is frozen; 15 observations from four frames are cached self-contained and reload without any model or GPU; the independent validator accepts the 40-second dynamic evidence video. |
| D8 | Complete | `ObjectMemory 1.0`, `MemoryObject 1.0`, pending observations, explicit evidence, strict serialization, and exact round-trip validation are frozen. |
| D9+ | Pending | Associate the new true-multiview evidence with an explainable spatial gate; do not promote a single pending observation directly to a permanent object. |

## D3 acceptance evidence

- Input: first 8 frames of the official `office_loop` sample; `submap-size=8`, `max-loops=0`, flow filter disabled.
- Both validators reported `PASS` with shape `(8, 294, 518, 3)` and no NaN/Inf.
- Mean valid fraction: `0.7498456911722218`; minimum per-frame valid fraction: `0.712269850024952`.
- First run: `966.735 s` including initial weight download; repeat: `17.101 s`.
- Peak GPU memory in each run: `4008.549 MB`.
- Matching SHA-256 values across both runs:
  - `geometry.npz`: `760bc15dad97482941e7e38940c225a0ba38440bd463944a0c8ea1d9a6c94619`
  - `geometry.manifest.json`: `a5a3d66d3a38af6dd3f1223b0c298e61a7637d6670a2a1b460f45bf11ec52190`
  - `geometry.anchor_poses.json`: `460701d1772392240c905579cd70a0a217a9a4c15a7ada974cda22979b3e828d`

## D4 acceptance evidence

- The separate Python 3.12 environment passes imports for PyTorch `2.10.0+cu128`, torchvision `0.25.0+cu128`, PE, and SAM 3; CUDA detects the RTX 4090.
- Perception Encoder and SAM 3 source commits are fixed below, and `--check-only` resolves all 8 D3 frames with point-map shape `(294, 518, 3)`.
- ModelScope `facebook/sam3` checkpoint: `3,450,062,241` bytes, SHA-256 `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`; strict model loading has no missing or unexpected keys.
- The historical `run_open_vocab_top1` artifacts remain reproducible, but are now explicitly labelled `B1-robust-single-view (legacy)`: they used original-resolution SAM input, resized masks, and `Robust3DLifter` and therefore were not strict official B0.
- `scripts.run_single_view_baselines` executes PE and SAM 3 once, then gives `B0-official` and `B1-robust-single-view` the same Top-1 frame, VGGT-preprocessed image, SAM masks, point map, and transform; lifting is the only controlled variable.
- Query `trash can` selected `frame_0004` at `0.20180504907322777`; SAM 3 produced six masks directly on the `(294, 518)` VGGT grid with no post-inference resize.
- `B0-official` uses finite-only direct mask indexing plus upstream PCA OBB and retained six instances with 5,899 points.
- `B1-robust-single-view` adds confidence, MAD, and minimum-point filtering and retained four instances with 5,379 points; four instances are shared for controlled comparison.
- The controlled run took `19.409 s`, peaked at `5090.538 MB`, and `scripts.validate_single_view_baselines` reported `PASS`.
- The strict validator covers transform/image hashes, mask-grid identity, finite point clouds, and frame/query/class/score consistency.
- `printer` is retained as a negative D4 example: the first eight frames do not show a printer, and all raw SAM candidates stay far below the official `0.50` threshold.

## D5 acceptance evidence

- One PE pass over the eight D3 frames exported deterministic K=1/3/5 selections for query `trash can`.
- Temporal suppression with `min_frame_gap=2` retained 1/3/4 frames for K=1/3/5; K=5 correctly reports exhausted nonredundant candidates instead of silently backfilling duplicates.
- Top-1 is `frame_0004` at score `0.20180504907322777`, exactly matching D4 B0; all K outputs are prefix-consistent.
- Real run time was `9.640 s`, peak GPU memory was `2797.170 MB`, and `scripts.validate_topk_retrieval` reported `PASS`.

## D6 acceptance evidence

- `scripts.run_sam_topk_lifting` consumes the saved D5 selection and does not rerun PE; it loads SAM 3 once and processes all selected frames.
- The corrected B2 runner reproduces VGGT resize/crop/batch-pad for each selected frame, lets SAM masks directly index the same-size point grid, and forbids post-inference mask resizing.
- Per-frame SAM/lifted counts were `frame_0004` 6/4, `frame_0001` 6/4, `frame_0006` 5/4, and `frame_0008` 4/3.
- All four selected frames produced masks and valid world-space observations: 15 of 21 SAM instances passed robust lifting and six were explicitly rejected for insufficient trustworthy points.
- The controlled run took `13.918 s`, peaked at `5088.913 MB`, and the hardened `scripts.validate_d6` reported `PASS`.
- A synthetic regression independently confirms that the radial MAD filter removes a far 3D outlier.

## D7 acceptance evidence

- `ObjectObservation` schema 1.0 freezes the frame/query/score, source mask, sampled world-space points, confidence summary, and world-space OBB fields required by later memory stages.
- The cache copies masks and sampled points into a self-contained directory, records an inventory plus hashes, and reloads independently of D3-D6 artifacts, PE, SAM 3, VGGT, CUDA, or a GPU.
- The real D6 input produced four cached frames and 15 observations containing 20,814 finite sampled points; every stored mask has the VGGT geometry resolution `(294, 518)`.
- The replacement evidence video is a dynamic 40-second, 10 FPS four-stage rendering (source masks, per-frame lifting, accumulated scene, rotating world view), not a four-image slideshow.
- The independent video check measured a motion ratio of `0.9493670886075949`, above its `0.25` threshold. It rejects the retained local slideshow backup, whose motion ratio is about `0.048`.
- Cache construction took `8.484 s`, required no GPU, and the complete local D7 output occupies about `11 MiB`.
- The revalidated MP4 is `5,110,187` bytes with SHA-256 `ada66f1382c8beebb6f2f291677b90d975544a3b6e342233d03fd4756c67fed3` and remains ignored under `runs/*`.

## True-multiview supplemental evidence

- Deterministic `--frame-start/--frame-stride` selection exported frames `0001, 0011, ..., 0071` and recorded the final sources in the geometry manifest.
- The original consecutive eight frames fail the declared gate with maximum translation `0.0669428446` reconstruction units and rotation `0.630177945` degrees.
- The interval sequence passes: maximum translation is `1.1124900418` reconstruction units and maximum rotation is `4.456059477` degrees, above thresholds `0.5` and `3.0`.
- Positive query results `(SAM/lifted/frames)` are: `trash can` 15/10/4, `poster` 8/7/2, `blue recycling bin` 7/5/3, and `printer` 2/2/2.
- `dog` and `bed` both produce zero evidence; their D6 status is `INSUFFICIENT_MULTIFRAME_3D_EVIDENCE`, not a premature D10-style reliable-rejection claim.

## D8 acceptance evidence

- `ObjectMemory 1.0` and `MemoryObject 1.0` have strict field/version checks and explicit per-observation evidence.
- D7 observations enter `pending_observations`; they are not promoted to permanent objects until a later association decision exists.
- The real D8 artifact contains 15 pending observations over four frames, zero permanent objects, and zero association decisions.
- Save/reload produces exactly equal canonical JSON; unknown, missing, or tampered evidence fields fail validation.
- D8 is schema-only and reuses the frozen D7 cache; D9 association will use the new interval-view D6 evidence.

## Publication verification

- `python -m unittest discover -v tests`: all 56 tests passed.
- D4 controlled artifact: `python -m scripts.validate_single_view_baselines runs/office-loop-single-view-trash-can` reported `PASS` with six B0 and four B1 objects from shared inputs.
- D6 controlled artifact: `python -m scripts.validate_d6 runs/office-loop-d6-controlled-trash-can` reported `PASS` for 21 masks, 15 lifts, and six rejections.
- True-multiview geometry: `python -m scripts.validate_multiview_geometry runs/office-loop-multiview-s10/geometry.npz` reported `PASS` for eight views and the declared pose-spread gate.
- D7 real artifact: `python -m scripts.validate_d7_cache runs/office-loop-d7-trash-can` reported `PASS` for four frames, 15 observations, 20,814 points, and motion ratio `0.9493670886075949`.
- D8 real artifact: `python -m scripts.validate_d8_memory runs/office-loop-d8-trash-can` reported `PASS` with exact round-trip equality and no premature permanent objects.
- `python -m compileall` over project modules and `git diff --check` passed.
- The continuity audit found no missing tracked files or upstream pin mismatches; all required local-only assets were present.
- The publication instance has an NVIDIA GeForce RTX 4090 with 24,564 MiB total memory.
- `/root/autodl-tmp` has `24.5 GiB` free, above the `10 GiB` baseline.

## Fixed upstream sources

- VGGT-SLAM 2.0: `35327ac28b7d193df9ccc39ba6346052bb6f1207`
- SALAD: `33ca9c0ca1e10cbb21efc0d6a5fcb6d45688e42d`
- VGGT_SPARK: `6e6e16107b88e8e76c751826af10d4295d87ecd2`
- Perception Encoder: `3e352cca660658d4b5c90f42a7808b11469e4c66`
- SAM 3: `8f0b7f4d4e7eda2ed606ebde6702c93359ad01da`

These checkouts are intentionally ignored under `third_party/VGGT-SLAM`; a clean clone must restore them at the exact commits above.

## Current local-only instance assets

These were present when the baseline was created but are not guaranteed to exist after cloning a new instance:

- Geometry environment: `/root/autodl-tmp/envs/vggt_geom/bin/python`
- VGGT-1B weight: `/root/autodl-tmp/cache/torch/hub/checkpoints/model.pt`, `5,026,874,952` bytes
- Official sample: `data/office_loop`, `473` files
- First and repeat D3 geometry NPZ: `runs/office-loop/geometry.npz` and `runs/office-loop-repeat/geometry.npz`, each `5,780,390` bytes
- Open-vocabulary environment: `/root/autodl-tmp/envs/open_vocab`, about `7.3 GiB`
- PE-Core-L14-336 weight revision `bafb0f76541d399057e980a25947f67acec76575`: `2,684,747,432` bytes under `/root/autodl-tmp/cache/huggingface`
- SAM 3 weight: `/root/autodl-tmp/cache/modelscope/facebook-sam3/sam3.pt`, `3,450,062,241` bytes
- First/repeat real PE results: `runs/office-loop-pe-printer` and `runs/office-loop-pe-printer-repeat`
- Historical legacy-B1 results: `runs/office-loop-b0-trash-can` and `runs/office-loop-b0-trash-can-repeat`
- Passing controlled B0/B1 result: `runs/office-loop-single-view-trash-can/single_view_result.json`
- Passing D5 retrieval result: `runs/office-loop-d5-trash-can/retrieval.json`
- D5 maximum nonredundant selection: `runs/office-loop-d5-trash-can/topk_5.json`
- Historical D6 result retained for D7 reproducibility: `runs/office-loop-d6-trash-can/d6_result.json`
- Passing controlled D6 result: `runs/office-loop-d6-controlled-trash-can/d6_result.json`
- Passing true-multiview geometry: `runs/office-loop-multiview-s10/geometry.npz`, `5,678,598` bytes
- Passing D7 self-contained cache: `runs/office-loop-d7-trash-can/scene_cache.json` and `observations.json`
- Passing D7 dynamic evidence video: `runs/office-loop-d7-trash-can/stage_video.mp4`, `5,110,187` bytes; the full D7 output is about `11 MiB`
- Passing D8 schema/round-trip result: `runs/office-loop-d8-trash-can/object_memory.json`
- Baseline disk requirement: at least `10 GiB` free under `/root/autodl-tmp`

Run `python .agents/skills/vggt-instance-handoff/scripts/audit_instance.py` after every instance migration. Missing ignored assets mean the new server is incomplete, not that Git-tracked source was lost.

## Concrete next task

1. Build D9 inputs from the interval-view D6 query evidence rather than the weak consecutive-frame D7 sample.
2. Implement same-class candidate generation and an explainable 3D center-distance/overlap gate that turns pending observations into association decisions.
3. Add labelled same/different pairs, pairwise precision/recall/F1, failure cases, strict serialization, and an independent D9 validator.

## Publication history

- 2026-08-20: PR #1 merged to private `main` at `5946f3e30bf75c556802085ebeb6374004898a88`; D3 integration and first validated run were published, and the local repeat run subsequently confirmed strict reproducibility.
- 2026-08-22: D4 PE/SAM 3 integration, BF16 inference fix, local checkpoint workflow, two matching real B0 runs, validator evidence, and disposable-instance continuity support were published to private `main` in the commit containing this entry (parent `5946f3e30bf75c556802085ebeb6374004898a88`).
- 2026-08-23: D5 deterministic PE Top-K retrieval and D6 multi-frame SAM 3/robust lifting, real RTX 4090 evidence, validators, 31-test regression, documentation, memory, and baseline were published in the commit containing this entry (parent `d46b3f7d3ba2b91bed86e943273e99be7b2e48ad`).
- 2026-08-24: D7 frozen self-contained observation cache, dynamic four-stage evidence video, independent cache/video validator, 38-test regression, documentation, memory, and baseline were published to public `main` in the commit containing this entry (parent `745d82c1be62a94025b168de4630040edd4d69fe`).
- 2026-08-25: Corrected D4/D6 controlled baselines, true-multiview pose gate and query evidence, D8 frozen object-memory schema, 56-test regression, documentation, memory, and baseline were published to public `main` in the commit containing this entry (parent `7b646deda5e8a01d128cb95f274f97316d01f53c`).

## Scope reminder

The project is currently a semantic-navigation perception front end. It does not yet include path planning, control, or closed-loop navigation. Ground-truth depth, poses, and OBBs belong only in evaluation/oracle paths, not the primary inference input.
