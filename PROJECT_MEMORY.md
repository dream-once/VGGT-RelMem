# VGGT-RelMem project memory

This tracked file is the recoverable handoff snapshot for disposable cloud instances. Update it in every user-authorized GitHub publication. Never put credentials, GitHub device codes, tokens, private keys, cookies, authentication files, or private dataset contents here.

## Current snapshot

- Updated: 2026-08-30 D16–D21 and portable GPU-evidence publication session
- Public repository: `dream-once/VGGT-RelMem`
- Publication target and pre-publication HEAD: `main` at `806c84558d6234fc5952a0ef74094c46e934c52e`
- Current milestone: D16–D21 are complete at CPU/source scope, with fail-closed Clio readiness, relation reliability, frozen Q×A experiments, ablation audit, reproducible result tables and a final claim-audited result card. D15 floating replay and public GPU/D15/D15.5 evidence are hardened separately.
- Next milestone: verify the Clio data licence and reliable archive/extracted sizes before any download, then run real calibration, held-out and any required GPU experiments. Do not turn the legacy D15 observation-count signal into an object/spatial-coverage claim.
- Publication state: seven ordered commits publish D16 through D21 and the D15/public-evidence correction. Large run trees, geometry, masks, point clouds and videos remain local-only; one hash-pinned overview PNG is retained under the explicit evidence-policy exception.

## Document-day progress

| Day | State | Evidence / remaining work |
| --- | --- | --- |
| D1 | Complete | All five upstream sources are pinned; disk/GPU checked; local PE and ModelScope SAM 3 checkpoints are verified. |
| D2 | Complete | `vggt_geom` and Python 3.12 `open_vocab` are isolated; geometry/mask schemas, adapters, and smoke tests pass. |
| D3 | Complete | Two schema-0.1 GPU runs were byte-identical; a real schema-0.2 RTX 4090 export preserves raw confidence plus valid masks and passes the independent validator. |
| D4 | Complete | Historical output is relabelled legacy B1; controlled `B0-official` and `B1-robust-single-view` now share the exact PE/SAM/VGGT inputs and differ only in lifting. |
| D5 | Complete | Temporal and six interval-sequence hybrid GPU runs exported deterministic K=1/3/5 results; all hybrid outputs pass the independent validator. |
| D6 | Complete | Controlled B2 preprocessing removes post-SAM mask resizing; four selected frames produced 21 SAM instances, 15 valid observations, and six explicit rejections. |
| D7 | Complete | Stride inputs resolve from the geometry manifest; the current true-multiview cache has 10 observations over four frames and a validated 40-second dynamic video. |
| D8 | Complete | The true-multiview D7 cache produces 10 pending observations, zero objects/decisions, exact round trip, and a relative source reference that survives bundle moves. |
| D9 | Complete | Exact-class plus center-distance/AABB-overlap gating reached pairwise F1=1.0 on 45 manually labelled pairs; only the cross-frame component became permanent. |
| D10 | Complete | Prediction is label-free and deterministic; evaluation alone reads manual labels. Both real JSON bundles pass independent validators. |
| D11 | Complete at engineering scope | The historical 4+4 partial cache is retained; GPU materialization now provides 8/8 available outcomes, 21 observations and eight rejections without retained-outcome drift. PE embedding binaries remain not retained. |
| D12 | Complete at engineering scope | A2 complete-link and evidence contracts remain frozen; label-free CPU prediction over the complete real GPU outcome cache passes. No new labelled evaluator or F1 claim was added. |
| D13 | Complete at engineering scope | Q0 remains upstream-aligned; 10/10 static checks and real `frame_0001` B0/B1 single-view GPU acceptance pass. |
| D14 | Complete at engineering scope | Q1 prediction replay over the complete real GPU outcome cache passes and K=1 matches Q0. No new labelled recall result was produced. |
| D15 | Complete at engineering scope | Q2 completes five real-cache steps and records 14 new frame-scoped observations under the legacy `observed_gain` field; this is neither object nor spatial coverage. Float replay now tolerates harmless cross-environment ULP drift. |
| D15.5 | Complete | A 95-anchor long-trajectory run yields 40 3D observations, eight predicted objects, a 132,204-point PLY and a validated 10-second MP4; three objects pass strict object-centric multiview and five remain diagnostic. |
| D16 | CPU complete; data blocked | Fail-closed Clio manifest and disk gate pass; download remains blocked by unknown archive/extracted sizes, unverified data licence and missing checksum. No Clio data or ROS stack was downloaded. |
| D17 | CPU complete; real calibration pending | Formal relation prediction is label-free, negative-query abstention is evaluated correctly, and ECE/Brier/coverage-risk/AURC paths pass synthetic and office-loop replay. Default threshold `0.60` is not claimed as calibrated. |
| D18 | CPU complete; Clio held-out pending | The Q×A matrix, shared-cache rule, frozen hashes/budgets and blocked-outcome behavior pass development and complete-synthetic replay. No held-out score is reported. |
| D19 | CPU complete; real ablation pending | One-factor Q2/A2 ablations and six-stage failure accounting pass synthetic evaluation; office-loop results remain engineering structure only. Unimplemented history-success scoring is recorded honestly. |
| D20 | CPU complete; optional binaries pending | Canonical JSON now rebuilds Q×A, relation and ablation tables automatically; clean-clone, relocation, hash, path and evidence-policy checks pass. |
| D21 | CPU complete | The final result card and README claim audit pass, positioning the project as an auditable semantic-localization reliability layer over VGGT-SLAM rather than closed-loop navigation. |

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
- Geometry schema 0.2 saves `raw_confidence_maps`, boolean `valid_masks`, and a legacy binary alias; 0.1 archives remain loadable.
- The real schema-0.2 RTX 4090 run passed with 8 frames, finite raw confidence `1.0–14.375`, mean valid ratio `0.7498456912`, `16.270 s` runtime, and `4008.549 MB` peak VRAM. Its generated NPZ was intentionally purged after validation under the new lightweight policy.

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
- Git now retains only the D7 observation/inventory/run JSON and saved validator report. Dense masks, sampled points, previews, and the MP4 were deliberately removed; the full ignored run can be regenerated locally.
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

## D9 acceptance evidence

- Manual visual labels group the 10 real D8 observations into three physical trash-can instances with component sizes 2/6/2.
- The D9 gate uses normalized exact class equality and requires center distance `<=0.15` unscaled reconstruction units or positive AABB IoU; it does not use learned semantic similarity or confidence in matching.
- All 45 labelled pairs were evaluated: 17 true same-instance pairs and 28 different-instance pairs produced precision/recall/F1/accuracy of `1.0` with no saved failure cases.
- The predicted graph has three components. Only the six-observation component covering `frame_0001/0041/0021` is promoted to `obj_0001`; two same-frame-only duplicate components remain as four pending observations.
- The real model-free run took `0.0125 s`, recorded no GPU allocation, round-tripped exactly, and passed an independent validator that recomputes every pair and rejects tampering or absolute source paths.
- The generated D9 run directory was intentionally deleted after validation. Source, manual labels, tests, README metrics, and the tracked D8 JSON input are sufficient to reproduce it.

## D10 acceptance evidence

- `run_d9_association` no longer accepts labels or F1 thresholds and saves only label-free pair predictions, components, ObjectMemory, provenance, and integrity acceptance.
- `evaluate_d9_association` independently attaches frozen manual labels and is the only D9 path that writes pairwise metrics, expected relations, or failure cases.
- The real prediction bundle preserves all 10 observations, produces 45 pairs, 17 predicted edges, three components, one permanent object, four pending observations, and six association decisions.
- The separate development evaluation contains 17 positive and 28 negative pairs with precision/recall/F1 `1.0`; this remains a regression fixture, not held-out evidence.
- Prediction and evaluation validators both report `PASS`; tests cover label non-interference, observation-order invariance, zero-match legality, A1 bridge behavior, tampering, and path escape.
- Week 2 D10 evidence is self-contained, JSON/Markdown-only, and about 86 KiB.

## D11 acceptance evidence

- `VisualMemoryManifest 0.1` records frame IDs, geometry indices, image hashes, camera centers/directions, embedding rows, PE provenance, and relative embedding-artifact metadata.
- The retained real manifest honestly sets `embedding_status=not_retained`; no embedding binary or fabricated vector is published.
- `CandidateOutcomeCache 0.1` conserves all eight D5 candidates. Four D6/D7 candidates are `available`; four are `unmaterialized`, not zero detections.
- Available outcomes contain 10 observations and five explicit lifting rejections in total. The policy-independent cache contains no GT, labels, answers, metrics, or policy trace.
- The independent replay validator reports `PASS_WITH_UNMATERIALIZED_OUTCOMES`; a complete synthetic cache reports the strict `complete` contract in CPU tests.
- The published D11 evidence bundle remains the 67,029-byte JSON/Markdown-only partial snapshot. The 2026-08-29 local GPU addendum completes all candidate outcomes; PE embedding binaries remain intentionally not retained.


## D12 acceptance evidence

- Frozen A1 prediction and ObjectMemory hashes remain `6aabd271…0362` and `1d8029e8…9f3`; both D10 validators still report `PASS`.
- A2 pair records save semantic mode/similarity, center distance, AABB IoU, sorted OBB extent ratios, both observation qualities, fixed thresholds, weighted score, gate result, final cluster membership, and reasons.
- Default weights are semantic `0.25`, center `0.25`, overlap `0.20`, OBB shape `0.15`, and quality `0.15`; quality must be at least `0.25`, with center `<=0.15` or positive overlap.
- Deterministic complete-link blocks A–B–C bridge merges unless every cross-cluster pair passes. Same-frame duplicates may cluster, but promotion still requires two distinct frames.
- Real D8 prediction has 45 pairs, 17 gate passes/final matches, seven merges, three clusters, one permanent object, four pending observations, and exact conservation/round-trip.
- Separate frozen-label evaluation is precision/recall/F1 `1.0`, equal to A1 on this development fixture; it is recorded as parity, not improvement or held-out evidence.
- Both independent validators report `PASS`; 15 dedicated tests cover bridge splitting, conflicts, quality, fallback, ordering, conservation, leakage, paths, and tampering.
- The published D12 evidence bundle remains the 111,875-byte JSON/Markdown development snapshot. The 2026-08-29 label-free A2 prediction over complete real GPU outcomes passes without a new labelled evaluation.


## D13 acceptance evidence

- Protocol ID is `Q0-vggt-slam-upstream-top1`; status remains `upstream-aligned`. Both FOUND-IT-official and VGGT-SLAM-official-reproduction claims are explicitly false.
- Q0 freezes non-negative PE cosine Top-1, VGGT 518 crop/14 alignment/white batch padding, SAM threshold `0.5`, same-grid direct mask indexing, finite-only points, and ordinary PCA OBB.
- Confidence gating, radial MAD, minimum-point gating, robust PCA, and post-SAM mask resize are explicitly forbidden.
- Nine pinned source files match SHA-256 and all 10 semantic source checks pass; the four upstream commits also match the instance baseline.
- D5 `trash can` raw rank 1 and upstream Top-1 both select `frame_0001` with score `0.1963950286`.
- Retained D4 JSON records `frame_0004`, `294×518`, threshold `0.5`, no mask resize, and the direct finite-only PCA method.
- The historical D4 validator report remains `PASS`, but its strict rerun now reports only missing `masks.json` and `preview.png`; this lightweight-publication gap is preserved as a limitation.
- The D13 validator reports `PASS`; eight tests cover claim boundaries, source semantics, D5 agreement, the D4 gap, and preprocess/SAM/lifting/OBB/hash tampering.
- The published D13 evidence bundle remains the 8,725-byte static/retained snapshot. The 2026-08-29 real Q0 B0/B1 single-view GPU replay passes while protocol status remains `upstream-aligned`.


## D14 acceptance evidence

- `Q1-fixed-topk-hybrid` freezes budgets `1/3/5`, frame gap `2`, camera distance `0.15`, and view angle `3°`.
- Selection input contains only rank/frame/geometry/pose/retrieval metadata; a cached outcome is revealed only after its frame is selected.
- The real partial cache selects `frame_0001/0071/0041/0021`; requested K=1/3/5 yields 1/3/4 frames, with explicit nonredundant-candidate exhaustion at K=5.
- Real replay costs are 1/3/4 SAM calls and 4/8/10 lifted observations. All selected outcomes exist; no unmaterialized candidate is treated as a zero detection.
- Separate development evaluation reports observed-instance recall `0.667/1.0/1.0`; this is one labelled development query, not held-out or new-GPU evidence.
- The complete synthetic cache executes full 1/3/5 paths. Missing selected outcomes return `BLOCKED_MISSING_OUTCOME` without skip-ahead.
- The D14 validator reports `PASS`; seven dedicated tests cover prefix determinism, Q0 compatibility, metadata-only selection, redundancy exhaustion, missing outcomes, label isolation, and tampering.
- The D14 evidence bundle remains the historical JSON/Markdown-only partial/synthetic publication. The 2026-08-29 complete real-outcome prediction replay passes separately without a new labelled evaluator.

## D15 acceptance evidence

- Historical schema 0.1 retains policy ID `Q2-gain-based-sequential-search`; the corrected method name is `retrieval-pose-novelty-sequential-search`, with max budget `5`, empty-observation threshold `1`, patience `2`, and retrieval/novelty weights `0.65/0.35`.
- Retrieval uses candidate-universe min-max normalization; pose novelty averages translation clipped at `0.15 m` and view angle clipped at `3°`.
- Step one is retrieval-only and selects Q0 `frame_0001`; later selection reads metadata only and reveals exactly one selected outcome afterward.
- The retained partial-cache trace reveals 4/2/2 new observations from `frame_0001/0071/0041`, then selects unmaterialized `frame_0061` and immediately returns `BLOCKED_MISSING_OUTCOME`.
- The retained trace has `performance_claim=null`; it does not skip to `frame_0021` or report partial-cache performance improvement.
- Complete synthetic replay selects five frames and stops at `max_budget_reached`. Zero-gain, exhaustion, missing-outcome, tie-break, and budget-one paths are independently tested.
- Synthetic Q0/Q1/Q2 comparison contains selected frames and SAM/lifting counts only; Q1 and Q2 are equal on this fixture, so no improvement is claimed.
- The D15 validator reports `PASS`; nine dedicated tests and 136 total CPU tests pass. Evidence is JSON/Markdown-only at about 46 KiB.
- The 2026-08-29 complete real-outcome trace runs five steps through `frame_0001/0071/0041/0061/0031`, stops at `max_budget_reached`, records 14 new frame-scoped observations through the legacy `observed_gain` field and makes no coverage/performance claim.


## 2026-08-29 GPU completion addendum

- The additive validator command is `conda run --no-capture-output -p /root/autodl-tmp/envs/vggt_geom env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 python -m scripts.validate_gpu_acceptance runs/gpu-acceptance-20260829`; the OpenCV-capable environment is required for the independent D7 video probe.
- The final report is `PASS / COMPLETE` with scope `ENGINEERING_REPLAY_NO_NEW_MANUAL_LABELS`. D3/D5/D6/D13 execute real GPU model inference; D7/D8/D11/D12/D14/D15 are CPU assembly/replay over those outcomes.
- The D11 candidate universe and raw ranking stay frozen. All eight outcomes are available with 21 observations and eight rejections; retained frames `0001/0071/0041/0021` do not drift, and `0061/0031/0011/0051` are newly materialized. PE embedding binaries remain `not_retained`.
- D12 label-free prediction passes without a new labelled evaluation. D13 Q0 selects `frame_0001` and both B0/B1 pass. D14 complete-cache prediction passes with K=1 equal to Q0.
- D15 complete-cache trace selects `0001/0071/0041/0061/0031`, stops at `max_budget_reached`, records 14 new frame-scoped observations, and keeps `performance_claim=null` and `coverage_aware=false`.
- Creation-time manifests inside the ignored local bundle may still contain the earlier pending snapshot. The additive validator/report is authoritative for completion; those large artifacts are not published.

## D15.5 acceptance evidence

- Long geometry uses 95 anchors selected at stride 5 from the 473-frame office-loop sequence, covers turns/return, and intentionally runs with `max_loops=0`.
- Top-24 retrieval plus real SAM 3/Robust3DLifter produces 49 SAM instances, 40 world-space observations and nine explicit rejections over 20 frames with lifted evidence.
- A2 produces 13 clusters and eight promoted predicted objects. Three objects pass strict object-centric multiview; five are diagnostic, and none is silently upgraded from weak evidence.
- Frozen strict evidence requires at least three distinct frames, at least two same-frame-pair gates jointly satisfying angle `>=15°` and baseline/mean-depth `>=0.20`, and at least three covered frames.
- The D15.5 validator reports `PASS`: overview `2931x1010`, MP4 `10.0 s / 12 FPS / 120 frames` with motion ratio `1.0`, and binary colored PLY with `132,204` vertices.
- Eight dedicated tests reject in-place rotation, same-ray translation, split-pair angle/ratio evidence, duplicate same-frame masks, path escape, and nondeterministic OBB output.
- The MP4 is an offline virtual orbit around the final static scene, not a robot-trajectory video or real 360° object coverage. Camera frusta are schematic, coordinates are reconstruction units, and no loop-closure/geometric-accuracy improvement is claimed.
- The fixed upstream provides Viser incremental mapping, trajectory walkthrough, Top-1 single-frame OBB and PCD export but no built-in MP4 exporter at the pinned commit. D15.5's own contribution is the object-memory evidence package, audit, portable artifacts and independent validation, not the underlying SLAM geometry or video encoding primitive.

## Publication verification

- `PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v`: all 190 tests passed after D16–D21 and the D15/public-evidence corrections.
- `python -m scripts.validate_d8_memory evidence/week1/runs/office-loop-mv-d8-trash-can`: `PASS` after binary evidence removal.
- The real D3 schema-0.2 GPU validator passed with raw confidence available and finite.
- The real D9 validator passed for 45 pairs, one permanent cross-frame object, four pending observations, and exact JSON round trip before generated runs were purged.
- D9 tests cover class mismatch, distance/overlap alternatives, insufficient frame support, explicit failure cases, label coverage, pair tampering, and source-path escape.
- Project compileall, `git diff --check`, and `git diff --cached --check` passed.
- Git evidence is JSON/text-only and about 296 KiB; 50 binary evidence files were removed from the current tree.
- Both tracked D10 validators pass directly against `evidence/week2/d9-office-loop-trash-can`.
- The D10 Week 2 bundle adds about 86 KiB; evidence policy rejects non-JSON/Markdown files and ground-truth fields in prediction.
- The tracked D11 validator reports `PASS_WITH_UNMATERIALIZED_OUTCOMES` with eight candidates, four available outcomes, four unmaterialized outcomes, 10 observations, and five rejections.
- D11 tests cover strict schema round-trip, complete synthetic replay, universe conservation, duplicate frames, illegal states, path escape, source tampering, and GT/policy leakage.
- Both tracked D12 validators report `PASS`; the real A2 development evaluation remains precision/recall/F1 `1.0` with no post-label threshold tuning.
- D12 tests freeze A1 hashes and cover complete-link bridge splitting, embedding fallback/conflict, low quality, same-frame deferral, order invariance, conservation, label isolation, paths, and tampering.
- The tracked D13 validator reports `PASS` with 10/10 source checks, D5 Top-1 agreement, upstream-aligned claim boundaries, and the retained-D4 gap accounted for.
- D13 tests reject changes to preprocess size, SAM threshold, robust lifting, OBB method, and source hashes while preserving the explicit missing-binary limitation.
- The tracked D14 validator reports `PASS` for both the real partial-cache development replay and complete synthetic replay; K=1 matches Q0.
- D14 tests cover deterministic prefix selection, redundancy exhaustion, missing-outcome blocking, unselected-outcome non-interference, evaluator separation, and tampering.
- The tracked D15 validator reports `PASS`: real evidence is an explicit blocked readiness trace, while synthetic replay and comparison complete.
- D15 tests cover Q0 budget-one reduction, frozen novelty math, deterministic ties, low-gain/exhaustion stops, missing outcome, future-outcome non-interference, replay, and tampering.
- The local GPU acceptance validator replays D3/D5/D6/D7/D8/D11/D12/D13 plus D14/D15 contracts and reports `PASS / COMPLETE`; its three focused anti-drift/leakage tests pass.
- The local D15.5 validator reports `PASS` for hashes, PNG/MP4 decoding, video motion, PLY vertices and independently reclassified recorded object-centric pair metrics; its eight focused tests pass.
- D16, D17, D18, D19, D20 and D21 dedicated validators report `PASS` with deferred real-data work represented only by explicit `*_PENDING` or `BLOCKED_*` states.
- D15 replay accepts a one-ULP finite drift, still rejects material drift and NaN/Inf, and the retained D15 validator reports `PASS` across environments.
- The portable public evidence validator reports `PASS` for the sanitized GPU report, complete candidate cache and D15 trace, D15.5 validation report and hash-pinned overview PNG; it contains no absolute paths.
- Week 3 evidence remains below the frozen 768 KiB limit. Structured daily bundles stay below 128 KiB; the sole PNG exception is exact-path, SHA-256 and size constrained.
- Week 2 evidence remains JSON/Markdown-only at about 323 KiB total; the D15 daily bundle is about 46 KiB.
- The current instance has an NVIDIA GeForce RTX 4090 with 24,564 MiB VRAM and about 24.39 GiB free under `/root/autodl-tmp`, above the 10 GiB baseline.
- Generated runs remain optional and ignored: the current local GPU acceptance bundle is about 27 MiB and the D15.5 long-trajectory tree is about 142 MiB. Neither is an instance-baseline requirement or Git publication payload.

## Fixed upstream sources

- VGGT-SLAM 2.0: `35327ac28b7d193df9ccc39ba6346052bb6f1207`
- SALAD: `33ca9c0ca1e10cbb21efc0d6a5fcb6d45688e42d`
- VGGT_SPARK: `6e6e16107b88e8e76c751826af10d4295d87ecd2`
- Perception Encoder: `3e352cca660658d4b5c90f42a7808b11469e4c66`
- SAM 3: `8f0b7f4d4e7eda2ed606ebde6702c93359ad01da`

These checkouts are intentionally ignored under `third_party/VGGT-SLAM`; a clean clone must restore them at the exact commits above.

## Current local-only instance assets

The lightweight policy does not treat generated `runs/` as required instance assets. Two regenerated bundles are currently retained locally for inspection, but a fresh clone may omit them and reproduce them from source, weights, data, and retained JSON evidence.

Assets currently present:

- Geometry environment: `/root/autodl-tmp/envs/vggt_geom/bin/python`
- Open-vocabulary environment: `/root/autodl-tmp/envs/open_vocab/bin/python`
- VGGT-1B weight: `/root/autodl-tmp/cache/torch/hub/checkpoints/model.pt`, `5,026,874,952` bytes
- PE-Core-L14-336 checkpoint revision `bafb0f76541d399057e980a25947f67acec76575`, `2,684,747,432` bytes
- SAM 3 checkpoint: `/root/autodl-tmp/cache/modelscope/facebook-sam3/sam3.pt`, `3,450,062,241` bytes, SHA-256 `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`
- Official `data/office_loop`: 473 files
- Local GPU completion bundle: `runs/gpu-acceptance-20260829`, about 27 MiB, ignored and optional
- Local D15.5 long-trajectory tree: `runs/office-loop-d15_5-s5`, about 142 MiB, ignored and optional

Run `python .agents/skills/vggt-instance-handoff/scripts/audit_instance.py` after migration. Missing generated runs are expected; missing tracked files, upstream pins, environments, weights, or the dataset require attention.

## Concrete next task

1. Resolve `DATA_LICENSE_UNVERIFIED`, archive/extracted-size uncertainty and checksum availability before authorizing a Clio scene download; retain at least 10 GiB free space.
2. If the data gate later passes, materialize the frozen development/held-out protocol and run real relation calibration, Q×A evaluation and ablations without tuning on held-out labels.
3. Design any genuine object/spatial-coverage policy as a new protocol version. Preserve the legacy D15 trace for compatibility and do not relabel observation count as coverage gain.

## Publication history

- 2026-08-27: Published D14 Q1 Fixed Top-K metadata-only replay, real 1/3/4 and synthetic 1/3/5 budget curves, 127 CPU tests, and an approximately 39 KiB JSON/Markdown development bundle (pre-publication HEAD `3ee7e98`).
- 2026-08-27: Published D15 Q2 gain-based sequential search, real blocked readiness trace, complete synthetic Q0/Q1/Q2 engineering comparison, 136 CPU tests, and an approximately 46 KiB JSON/Markdown bundle (pre-publication HEAD `94bcb81`).
- 2026-08-29: Published the D11–D15 GPU completion validator, D15.5 long-trajectory scene-memory visualization/audit/validator, 147-test regression, updated documentation, memory and baseline without publishing binary run artifacts (pre-publication HEAD `3073b46`).
- 2026-08-30: Published D16 fail-closed Clio data protocol without downloading data (`db44e9f`).
- 2026-08-30: Published D17 label-free relation prediction, abstention evaluation and calibration boundaries (`5ea7b86`).
- 2026-08-30: Published D18 frozen Q×A experiment protocol and development/synthetic replay (`79d7d12`).
- 2026-08-30: Published D19 one-factor ablation and complete failure-accounting audit (`3ad14ad`).
- 2026-08-30: Published D20 reproducible package and automatically rebuilt result tables (`849fba1`).
- 2026-08-30: Published D21 final result card, claim audit and explicit held-out boundaries (`5dc3a59`).
- 2026-08-30: Published tolerant D15 replay, corrected Q2 semantics and portable GPU/D15/D15.5 evidence; 190 tests and final validators pass (commit containing this entry, pre-publication parent `5dc3a59`).

## Scope reminder

The project is currently a semantic-navigation perception front end. It does not yet include path planning, control, or closed-loop navigation. Ground-truth depth, poses, and OBBs belong only in evaluation/oracle paths, not the primary inference input.
