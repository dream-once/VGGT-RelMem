# VGGT-RelMem project memory

This tracked file is the recoverable handoff snapshot for disposable cloud instances. Update it in every user-authorized GitHub publication. Never put credentials, GitHub device codes, tokens, private keys, cookies, authentication files, or private dataset contents here.

## Current snapshot

- Updated: 2026-08-22 publication session (`2026-08-22T06:15:33Z` pre-publication audit)
- Private repository: `dream-once/VGGT-RelMem`
- Publication target and pre-publication HEAD: `main` at `5946f3e30bf75c556802085ebeb6374004898a88`
- Current milestone: document D3 and D4 are complete.
- Next milestone: begin D5 multi-view observation fusion and persistent object memory.
- Publication state: the completed D4 source, tests, documentation, continuity skill, memory, and baseline are included in the publication commit containing this entry. Environments, weights, datasets, upstream checkouts, and generated run artifacts remain ignored and local-only.

## Document-day progress

| Day | State | Evidence / remaining work |
| --- | --- | --- |
| D1 | Complete | All five upstream sources are pinned; disk/GPU checked; local PE and ModelScope SAM 3 checkpoints are verified. |
| D2 | Complete | `vggt_geom` and Python 3.12 `open_vocab` are isolated; geometry/mask schemas, adapters, and smoke tests pass. |
| D3 | Complete | Two independent 8-frame `office_loop` geometry runs passed validation and produced byte-identical geometry plus run manifests. |
| D4 | Complete | Two real `trash can` B0 runs produced identical masks, four valid 3D OBB observations, previews, manifests, and validator `PASS`. |
| D5+ | Pending | Multi-view observation fusion, persistent memory, relation evaluation, calibration, and full experiment matrix follow D4. |

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

## Publication verification

- `python -m unittest discover -v`: all 20 tests passed in no-card mode.
- Open-vocabulary environment: `python -m unittest tests.test_open_vocab tests.test_validate_open_vocab -v` passed all 6 focused tests.
- `bash -n scripts/bootstrap_open_vocab.sh` and `git diff --check` passed.
- The pre-publication instance audit found no deleted or missing required files; all five ignored upstream source checkouts match their fixed commits; every required local-only asset passed size/hash checks.
- `/root/autodl-tmp` had `24.55 GiB` free, above the `10 GiB` baseline.
- GPU was not detected in this no-card instance. Its empty `/usr/bin/nvidia-smi` placeholder previously raised `Exec format error`; the audit now records unavailable probes as `not detected` and continues instead of aborting.
- `git fetch --prune origin` succeeded, and `main...origin/main` was `0 0` before creating the publication commit.

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
- Baseline disk requirement: at least `10 GiB` free under `/root/autodl-tmp`

Run `python .agents/skills/vggt-instance-handoff/scripts/audit_instance.py` after every instance migration. Missing ignored assets mean the new server is incomplete, not that Git-tracked source was lost.

## Concrete next task

1. Re-read the recommendation document's exact D5 acceptance criteria before implementation.
2. Extend the validated single-view observations into multi-view association and persistent `ObjectMemory`.
3. Keep inference inputs separate from oracle/evaluation-only depth, poses, and OBBs.
4. Add deterministic fixtures and a small real-scene D5 validation before expanding the experiment matrix.

## Publication history

- 2026-08-20: PR #1 merged to private `main` at `5946f3e30bf75c556802085ebeb6374004898a88`; D3 integration and first validated run were published, and the local repeat run subsequently confirmed strict reproducibility.
- 2026-08-22: D4 PE/SAM 3 integration, BF16 inference fix, local checkpoint workflow, two matching real B0 runs, validator evidence, and disposable-instance continuity support were published to private `main` in the commit containing this entry (parent `5946f3e30bf75c556802085ebeb6374004898a88`).

## Scope reminder

The project is currently a semantic-navigation perception front end. It does not yet include path planning, control, or closed-loop navigation. Ground-truth depth, poses, and OBBs belong only in evaluation/oracle paths, not the primary inference input.
