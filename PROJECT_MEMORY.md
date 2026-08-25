# VGGT-RelMem project memory

This tracked file is the recoverable handoff snapshot for disposable cloud instances. Update it in every user-authorized GitHub publication. Never put credentials, GitHub device codes, tokens, private keys, cookies, authentication files, or private dataset contents here.

## Current snapshot

- Updated: 2026-08-25 publication session
- Public repository: `dream-once/VGGT-RelMem`
- Publication target and pre-publication HEAD: `main` at `d0399ffb872c78ec09eae6ab9168f92d47a1fbce`
- Current milestone: D6 → corrected stride-aware D7 → portable D8 is continuous; query-specific same-pair validation is explicit; D4–D8 lightweight evidence is tracked for public inspection.
- Next milestone: run the D3 geometry 0.2 exporter once on a GPU, then start D9 association from the true-multiview trash-can D8 bundle.
- Publication state: the stride fix, D3 schema 0.2 source, query-specific gate, portable D8, 64-test regression, and a 12 MiB D4–D8 evidence snapshot are included in the publication commit containing this entry. Environments, weights, datasets, upstream checkouts, and other generated runs remain local-only.

## Document-day progress

| Day | State | Evidence / remaining work |
| --- | --- | --- |
| D1 | Complete | All five upstream sources are pinned; disk/GPU checked; local PE and ModelScope SAM 3 checkpoints are verified. |
| D2 | Complete | `vggt_geom` and Python 3.12 `open_vocab` are isolated; geometry/mask schemas, adapters, and smoke tests pass. |
| D3 | Complete / 0.2 GPU addendum pending | Two schema-0.1 GPU runs remain byte-identical and valid; schema 0.2 now preserves raw confidence plus valid masks and passes source/unit checks, but needs one real GPU export. |
| D4 | Complete | Historical output is relabelled legacy B1; controlled `B0-official` and `B1-robust-single-view` now share the exact PE/SAM/VGGT inputs and differ only in lifting. |
| D5 | Complete | Temporal and six interval-sequence hybrid GPU runs exported deterministic K=1/3/5 results; all hybrid outputs pass the independent validator. |
| D6 | Complete | Controlled B2 preprocessing removes post-SAM mask resizing; four selected frames produced 21 SAM instances, 15 valid observations, and six explicit rejections. |
| D7 | Complete | Stride inputs resolve from the geometry manifest; the current true-multiview cache has 10 observations over four frames and a validated 40-second dynamic video. |
| D8 | Complete | The true-multiview D7 cache produces 10 pending observations, zero objects/decisions, exact round trip, and a relative source reference that survives bundle moves. |
| D9+ | Pending | Associate the true-multiview trash-can D8 bundle with an explainable spatial gate; do not promote a single pending observation directly to a permanent object. |

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
- Geometry schema 0.2 source now saves `raw_confidence_maps`, boolean `valid_masks`, and a legacy binary alias; 0.1 archives remain loadable.
- Current no-GPU validation confirms both old geometry artifacts still pass with `raw_confidence_available=false`; the GPU addendum must produce schema 0.2 with raw confidence available.

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
- Six 2026-08-24 interval-view GPU runs use `hybrid`, `min_frame_gap=2`, `min_camera_distance=0.15`, and `min_view_angle_deg=3.0`; all six revalidate as `PASS`.

## D6 acceptance evidence

- `scripts.run_sam_topk_lifting` consumes the saved D5 selection and does not rerun PE; it loads SAM 3 once and processes all selected frames.
- The corrected B2 runner reproduces VGGT resize/crop/batch-pad for each selected frame, lets SAM masks directly index the same-size point grid, and forbids post-inference mask resizing.
- Per-frame SAM/lifted counts were `frame_0004` 6/4, `frame_0001` 6/4, `frame_0006` 5/4, and `frame_0008` 4/3.
- All four selected frames produced masks and valid world-space observations: 15 of 21 SAM instances passed robust lifting and six were explicitly rejected for insufficient trustworthy points.
- The controlled run took `13.918 s`, peaked at `5088.913 MB`, and the hardened `scripts.validate_d6` reported `PASS`.
- A synthetic regression independently confirms that the radial MAD filter removes a far 3D outlier.

## D7 acceptance evidence

- `ObjectObservation` schema 1.0 remains frozen and self-contained.
- The stride resolver now reads the D6 run manifest's pinned geometry manifest and verifies image stems; `geometry_index=1` for a stride sequence resolves `frame_0011`, never the continuous-directory decoy `frame_0002`.
- The current true-multiview cache contains 10 observations from `frame_0001/0071/0041/0021`, 19,062 finite sampled points, and `(294, 518)` masks.
- Its 40-second video uses all input frames `0001, 0011, ..., 0071` and the selected Top-K/SAM/3D frames, with independent motion ratio `0.9746835443037974`.
- The MP4 is `6,038,230` bytes with SHA-256 `aa924ff9913fcca8475b16209a950004f74f20b6774d3903a9a9756eafae0dea`.
- The complete 7.9 MiB D7 cache is tracked under `evidence/week1/runs/office-loop-mv-d7-trash-can` and validates without model inference.
- The older 15-observation consecutive-frame cache remains local historical evidence but is no longer the D8/D9 input.

## True-multiview supplemental evidence

- Deterministic `--frame-start/--frame-stride` selection exported frames `0001, 0011, ..., 0071`; the complete sequence has maximum translation `1.1124900418` reconstruction units and rotation `4.456059477` degrees.
- Query-specific validation uses only `frames_with_lifted_observations` and requires the same frame pair to satisfy translation `>=0.5` and rotation `>=3.0`.
- `trash can` is `TRUE_MULTIVIEW`: pairs `0001-0071`, `0071-0041`, and `0071-0021` pass both thresholds.
- `poster`, `blue recycling bin`, and `printer` are honestly classified `MULTIFRAME_ONLY`; none has one observation pair passing both thresholds.
- `dog` and `bed` are intentional absent-object office controls and pass as `NEGATIVE_CONTROL` with zero SAM and zero 3D evidence.
- Their D6 insufficient-evidence state is not claimed as a later calibrated reliable-rejection result.
- Saved per-query reports are tracked under `evidence/week1/validation/query-multiview`.

## D8 acceptance evidence

- `ObjectMemory 1.0` stages the current true-multiview D7 observations as pending evidence and deliberately performs no D9 association.
- The real bundle contains 10 pending observations over four frames, zero permanent objects, and zero association decisions.
- Save/reload produces exactly equal canonical JSON; unknown, missing, or tampered evidence fields fail validation.
- The source cache reference is relative (`../office-loop-mv-d7-trash-can/observations.json`); absolute paths and bundle escapes fail, while moving the whole bundle passes.
- The copied public D8 bundle validates directly and is the deterministic, model-free D9 input.

## Publication verification

- `python -m unittest discover -v tests`: all 64 tests passed.
- Both local schema-0.1 geometry artifacts still validate with finite points and `raw_confidence_available=false`, as expected for legacy files.
- VGGT `--check-only`, project compileall, and the geometry 0.2 fake-solver export all pass without a GPU.
- D4 controlled artifact remains `PASS` with six B0 and four B1 objects from shared inputs.
- All six interval-sequence D5 hybrid directories revalidate as `PASS`.
- The multiview trash-can D6 artifact is `PASS` with 15 masks, 10 lifts, and five rejections.
- Query-specific reports classify trash can `TRUE_MULTIVIEW`, three other positives `MULTIFRAME_ONLY`, and dog/bed `NEGATIVE_CONTROL`.
- The copied public D7 artifact is `PASS` for four frames, 10 observations, 19,062 points, 40 seconds, and motion ratio `0.9746835443037974`.
- The copied public D8 artifact is `PASS` with 10 pending observations, exact round trip, and no premature objects or decisions.
- `python -m compileall` and `git diff --check` passed.
- This publication instance intentionally has no detected GPU; no new model inference is claimed.
- `/root/autodl-tmp` has about `24.5 GiB` free, above the `10 GiB` baseline.

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
- Passing six-query hybrid D5 and D6 matrix: `runs/office-loop-mv-d5-*` and `runs/office-loop-mv-d6-*`
- Passing current D7 cache: `runs/office-loop-mv-d7-trash-can`, about `7.9 MiB`
- Passing current D7 video: `runs/office-loop-mv-d7-trash-can/stage_video.mp4`, `6,038,230` bytes, SHA-256 `aa924ff9913fcca8475b16209a950004f74f20b6774d3903a9a9756eafae0dea`
- Passing portable D8 memory: `runs/office-loop-mv-d8-trash-can/object_memory.json`, SHA-256 `cdf1eccd7dabcae658f927ae122f5dcec77c220b0c4338f220adc88702411879`
- Missing by design until a GPU is attached: `runs/office-loop-v02/geometry.npz`
- Baseline disk requirement: at least `10 GiB` free under `/root/autodl-tmp`

Run `python .agents/skills/vggt-instance-handoff/scripts/audit_instance.py` after every instance migration. Missing ignored assets mean the new server is incomplete, not that Git-tracked source was lost.

## Concrete next task

1. On the next GPU instance, run the documented D3 schema-0.2 command and require raw confidence availability.
2. Use `runs/office-loop-mv-d8-trash-can` (or its tracked evidence copy) to implement same-class candidate generation and the explainable 3D center-distance/overlap D9 gate.
3. Add labelled same/different pairs, pairwise precision/recall/F1, failure cases, strict serialization, and an independent D9 validator.

## Publication history

- 2026-08-20: PR #1 merged to private `main` at `5946f3e30bf75c556802085ebeb6374004898a88`; D3 integration and first validated run were published, and the local repeat run subsequently confirmed strict reproducibility.
- 2026-08-22: D4 PE/SAM 3 integration, BF16 inference fix, local checkpoint workflow, two matching real B0 runs, validator evidence, and disposable-instance continuity support were published to private `main` in the commit containing this entry (parent `5946f3e30bf75c556802085ebeb6374004898a88`).
- 2026-08-23: D5 deterministic PE Top-K retrieval and D6 multi-frame SAM 3/robust lifting, real RTX 4090 evidence, validators, 31-test regression, documentation, memory, and baseline were published in the commit containing this entry (parent `d46b3f7d3ba2b91bed86e943273e99be7b2e48ad`).
- 2026-08-24: D7 frozen self-contained observation cache, dynamic four-stage evidence video, independent cache/video validator, 38-test regression, documentation, memory, and baseline were published to public `main` in the commit containing this entry (parent `745d82c1be62a94025b168de4630040edd4d69fe`).
- 2026-08-25: Corrected D4/D6 controlled baselines, true-multiview pose gate and query evidence, D8 frozen object-memory schema, 56-test regression, documentation, memory, and baseline were published to public `main` in the commit containing this entry (parent `7b646deda5e8a01d128cb95f274f97316d01f53c`).
- 2026-08-25: Closed the D7/D8 stride and portability gaps, added query-specific same-pair validation, upgraded D3 source to confidence-preserving schema 0.2, published 12 MiB of D4–D8 evidence, and recorded a 64-test no-GPU regression in the commit containing this entry (parent `d0399ffb872c78ec09eae6ab9168f92d47a1fbce`).

## Scope reminder

The project is currently a semantic-navigation perception front end. It does not yet include path planning, control, or closed-loop navigation. Ground-truth depth, poses, and OBBs belong only in evaluation/oracle paths, not the primary inference input.
