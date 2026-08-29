# Final result card

**Status:** CPU_COMPLETE

**Positioning:** VGGT-SLAM 几何之上的可审计语义定位可靠性层

**Conclusion:** 完成可复现基准、查询策略与关联策略隔离、可靠拒答协议和失败分析

| result | sample size | budget | validation | scope |
|---|---|---|---|---|
| D15.5-scene-visualization | {"anchor_cameras": 95, "observations": 40, "predicted_objects": 8, "selected_cameras": 20} | {"retrieval_top_k": 24} | PASS | office_loop_engineering_visualization |
| D16-clio-readiness | {"scene_roles": 2} | {"available_bytes": 26189692928, "maximum_peak_bytes": 15452274688, "safety_reserve_bytes": 10737418240} | PASS | metadata_readiness_no_download |
| D17-relation-reliability | {"negative": 3, "positive": 2, "queries": 5} | {"engineering_threshold": 0.6} | PASS | synthetic_correctness_real_calibration_pending |
| D18-QxA-protocol | {"frozen_queries": 1, "matrix_rows": 6} | {"Q0": 1, "Q1": [1, 3, 5], "Q2": 5} | PASS | synthetic_correctness_office_development_replay |
| D19-ablation-audit | {"a2_variants": 5, "frozen_queries": 1, "q2_variants": 3} | {"association_input_q1_k": 5} | PASS | synthetic_correctness_real_ablation_pending |
| D20-reproduction-package | {"canonical_inputs": 7, "derived_tables": 3, "stage_validators": 4} | {"cpu_only": true} | PASS | tracked_json_markdown_reproduction |

## Explicit gaps

- clio_download: DATA_DOWNLOAD_BLOCKED_SIZE_UNKNOWN
- data_license: DATA_LICENSE_UNVERIFIED
- clio_held_out: CLIO_HELD_OUT_PENDING
- real_calibration: REAL_DATA_CALIBRATION_PENDING
- real_ablation: REAL_ABLATION_PENDING
- new_gpu_inference: PENDING
- optional_binary_release: OPTIONAL_BINARY_RELEASE_PENDING

No Clio held-out or superiority number is claimed.
