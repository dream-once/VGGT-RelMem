# VGGT-RelMem project memory

This tracked file is the recoverable handoff snapshot for disposable cloud instances. Update it in every user-authorized GitHub publication. Never put credentials, GitHub device codes, tokens, private keys, cookies, authentication files, or private dataset contents here.

## Current snapshot

- Updated: 2026-08-24 publication session
- Public repository: `dream-once/VGGT-RelMem`
- Publication target and pre-publication HEAD: `main` at `745d82c1be62a94025b168de4630040edd4d69fe`
- Current milestone: document D1 through D7 are complete.
- Next milestone: correct the historical D4 B0/B1 naming and controlled baseline inputs before starting D8.
- Publication state: the completed D7 source, tests, documentation, memory, and baseline are included in the publication commit containing this entry. Environments, weights, datasets, upstream checkouts, and generated run artifacts remain ignored and local-only.

## Document-day progress

| Day | State | Evidence / remaining work |
| --- | --- | --- |
| D1 | Complete | All five upstream sources are pinned; disk/GPU checked; local PE and ModelScope SAM 3 checkpoints are verified. |
| D2 | Complete | `vggt_geom` and Python 3.12 `open_vocab` are isolated; geometry/mask schemas, adapters, and smoke tests pass. |
| D3 | Complete | Two independent 8-frame `office_loop` geometry runs passed validation and produced byte-identical geometry plus run manifests. |
| D4 | Complete | Two real `trash can` B0 runs produced identical masks, four valid 3D OBB observations, previews, manifests, and validator `PASS`. |
| D5 | Complete | Real PE Top-K retrieval exported deterministic K=1/3/5 results with temporal redundancy suppression and B0-compatible top-1. |
| D6 | Complete | Four D5-selected frames ran through SAM 3 and robust 3D lifting; all four produced valid observations and the independent validator passed. |
| D7 | Complete | `ObjectObservation` schema 1.0 is frozen; 15 observations from four frames are cached self-contained and reload without any model or GPU; the independent validator accepts the 40-second dynamic evidence video. |
| D8+ | Pending | Correct the controlled B0/B1 single-view baselines first, then freeze `ObjectMemory`; cross-frame association starts at D9. |

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
- Real `PE-Core-L14-336` query `printer` ran twice and selected `frame_0006` both times with cosine `0.1550685453894363`.
- First run: `961.779 s` including the 2.68 GB PE download; cached repeat: `10.914 s`; peak GPU memory: `2797.170 MB`.
- Both top-1 previews have SHA-256 `a1ccb8d1c9107e7c730cbd4074df4069de5599fad1360d7a7f159535a571e0d6`.
- ModelScope `facebook/sam3` checkpoint: `3,450,062,241` bytes, SHA-256 `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`; strict model loading has no missing or unexpected keys.
- SAM 3 image and prompt inference run under the official BF16 CUDA autocast context; a regression test covers the dtype boundary.
- Real query `trash can` selected `frame_0004` with cosine `0.20180504907322777`; SAM 3 returned six masks and four were lifted to valid 3D OBB observations.
- First and fully local repeat runs both passed `scripts.validate_open_vocab`; runtime was about `20 s` and peak GPU memory about `5090 MB`.
- Matching hashes across the two runs: `masks.json` `713f8b12e11fe36244200bc83dde61cc9f6e4e5fc8736931bb5ffa5376f4497c`, `observations.json` `f71ed5d6354998f3a2efc09f3c43f9d3573a1c4126837bbad93c118eb9505527`, and `preview.png` `e5c612a6a88662de942c7b01349dfdd4c5076fadfcde0d83a36f192ab3b02fa3`.
- `printer` is retained as a negative D4 example: the first eight frames do not show a printer, and all raw SAM candidates stay far below the official `0.50` threshold.
- A later audit found that this historical D4 runner is not a strict official B0: it feeds original-resolution images to SAM 3, resizes masks onto VGGT geometry, and uses `Robust3DLifter`. Its artifacts remain valid reproducibility evidence but will be relabelled as the legacy B1 robust single-view result; a controlled `B0-official`/`B1-robust-single-view` split is the next task.

## D5 acceptance evidence

- One PE pass over the eight D3 frames exported deterministic K=1/3/5 selections for query `trash can`.
- Temporal suppression with `min_frame_gap=2` retained 1/3/4 frames for K=1/3/5; K=5 correctly reports exhausted nonredundant candidates instead of silently backfilling duplicates.
- Top-1 is `frame_0004` at score `0.20180504907322777`, exactly matching D4 B0; all K outputs are prefix-consistent.
- Real run time was `9.640 s`, peak GPU memory was `2797.170 MB`, and `scripts.validate_topk_retrieval` reported `PASS`.

## D6 acceptance evidence

- `scripts.run_sam_topk_lifting` consumes the saved D5 selection and does not rerun PE; it loads SAM 3 once and processes all selected frames.
- Per-frame SAM/lifted counts were `frame_0004` 6/4, `frame_0001` 6/4, `frame_0006` 5/4, and `frame_0008` 5/3.
- All four selected frames produced masks and valid world-space 3D observations. In total, 15 of 22 SAM instances passed confidence filtering, MAD outlier removal, minimum-point checks, and coarse OBB fitting.
- Seven instances were explicitly rejected because fewer than 30 trustworthy VGGT points remained.
- Real run time was `14.374 s`, peak GPU memory was `5089.975 MB`, and `scripts.validate_d6` reported `PASS`.
- A synthetic regression independently confirms that the radial MAD filter removes a far 3D outlier.

## D7 acceptance evidence

- `ObjectObservation` schema 1.0 freezes the frame/query/score, source mask, sampled world-space points, confidence summary, and world-space OBB fields required by later memory stages.
- The cache copies masks and sampled points into a self-contained directory, records an inventory plus hashes, and reloads independently of D3-D6 artifacts, PE, SAM 3, VGGT, CUDA, or a GPU.
- The real D6 input produced four cached frames and 15 observations containing 20,814 finite sampled points; every stored mask has the VGGT geometry resolution `(294, 518)`.
- The replacement evidence video is a dynamic 40-second, 10 FPS four-stage rendering (source masks, per-frame lifting, accumulated scene, rotating world view), not a four-image slideshow.
- The independent video check measured a motion ratio of `0.9493670886075949`, above its `0.25` threshold. It rejects the retained local slideshow backup, whose motion ratio is about `0.048`.
- Cache construction took `8.484 s`, required no GPU, and the complete local D7 output occupies about `11 MiB`.

## Publication verification

- `python -m unittest discover -v tests`: all 38 tests passed.
- D5 real artifact: `python -m scripts.validate_topk_retrieval runs/office-loop-d5-trash-can` reported `PASS`.
- D6 real artifact: `python -m scripts.validate_d6 runs/office-loop-d6-trash-can` reported `PASS`.
- D7 real artifact: `python -m scripts.validate_d7_cache runs/office-loop-d7-trash-can` reported `PASS` for four frames, 15 observations, 20,814 points, and motion ratio `0.9493670886075949`.
- Base Python reloaded the D7 cache without importing model frameworks; schema version, frame count, and observation count were `1.0`, `4`, and `15`.
- D6 no-model check resolved four selected images, the geometry point-map shape `(294, 518, 3)`, the fixed SAM 3 source commit, and the 3,450,062,241-byte local checkpoint.
- `python -m compileall` over project modules and `git diff --check` passed.
- The publication instance has an NVIDIA GeForce RTX 4090 with 24,564 MiB total memory.
- `/root/autodl-tmp` has about `25 GiB` free, above the `10 GiB` baseline.

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
- First/repeat passing B0 results: `runs/office-loop-b0-trash-can` and `runs/office-loop-b0-trash-can-repeat`
- Passing D5 retrieval result: `runs/office-loop-d5-trash-can/retrieval.json`
- D5 maximum nonredundant selection: `runs/office-loop-d5-trash-can/topk_5.json`
- Passing D6 multi-frame result: `runs/office-loop-d6-trash-can/d6_result.json`
- D6 mask/observation/point-cloud/previews directory: `runs/office-loop-d6-trash-can`, about `6.3 MiB`
- Passing D7 self-contained cache: `runs/office-loop-d7-trash-can/scene_cache.json` and `observations.json`
- Passing D7 dynamic evidence video: `runs/office-loop-d7-trash-can/stage_video.mp4`, `5,110,187` bytes; the full D7 output is about `11 MiB`
- Baseline disk requirement: at least `10 GiB` free under `/root/autodl-tmp`

Run `python .agents/skills/vggt-instance-handoff/scripts/audit_instance.py` after every instance migration. Missing ignored assets mean the new server is incomplete, not that Git-tracked source was lost.

## Concrete next task

1. Preserve the historical D4 artifacts while relabelling their method as legacy `B1-robust-single-view`.
2. Implement controlled `B0-official` and `B1-robust-single-view` paths that share the same VGGT-preprocessed image and SAM 3 masks; only the lifting method may differ.
3. Save and validate the resize/crop/pad mapping and harden D4 validation for non-finite data and frame/query/class mismatches.
4. After the baseline correction, start D8 `ObjectMemory`; expand D3 to genuinely separated views before making D9 multi-view claims.

## Publication history

- 2026-08-20: PR #1 merged to private `main` at `5946f3e30bf75c556802085ebeb6374004898a88`; D3 integration and first validated run were published, and the local repeat run subsequently confirmed strict reproducibility.
- 2026-08-22: D4 PE/SAM 3 integration, BF16 inference fix, local checkpoint workflow, two matching real B0 runs, validator evidence, and disposable-instance continuity support were published to private `main` in the commit containing this entry (parent `5946f3e30bf75c556802085ebeb6374004898a88`).
- 2026-08-23: D5 deterministic PE Top-K retrieval and D6 multi-frame SAM 3/robust lifting, real RTX 4090 evidence, validators, 31-test regression, documentation, memory, and baseline were published in the commit containing this entry (parent `d46b3f7d3ba2b91bed86e943273e99be7b2e48ad`).
- 2026-08-24: D7 frozen self-contained observation cache, dynamic four-stage evidence video, independent cache/video validator, 38-test regression, documentation, memory, and baseline were published to public `main` in the commit containing this entry (parent `745d82c1be62a94025b168de4630040edd4d69fe`).

## Scope reminder

The project is currently a semantic-navigation perception front end. It does not yet include path planning, control, or closed-loop navigation. Ground-truth depth, poses, and OBBs belong only in evaluation/oracle paths, not the primary inference input.
