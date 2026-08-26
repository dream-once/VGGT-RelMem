# D13 Q0 protocol evidence

This static CPU audit freezes `Q0-vggt-slam-upstream-top1` with status `upstream-aligned`. It does not claim an official FOUND-IT implementation or a byte-identical reproduction of the interactive VGGT-SLAM process.

`q0_protocol.json` records pinned source hashes and verifies ten source/evidence properties: non-negative PE cosine Top-1, VGGT 518 resize/center-crop/white batch-pad, SAM threshold `0.5`, direct same-grid mask indexing, finite-only points, ordinary PCA OBB, and the absence of confidence/MAD/minimum-point/post-SAM-resize operations in Q0. The D5 development Top-1 is the raw rank-1 frame `frame_0001`.

The original D4 strict validator cannot rerun from the lightweight Git bundle because `masks.json` and `preview.png` were intentionally removed. The historical retained report is `PASS`; the current strict rerun fails only for those two missing artifacts. This limitation is part of the protocol and validator output rather than being hidden or converted into a pass.

Status: `CPU_COMPLETE / GPU_ACCEPTANCE_PENDING`. A future GPU replay may restore full binary validation, but it is not required for this static protocol freeze.
