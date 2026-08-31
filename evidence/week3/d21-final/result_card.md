# Final result card

**Status:** GPU_AND_CPU_COMPLETE

**Positioning:** VGGT-SLAM 几何之上的可审计语义定位可靠性层

**Conclusion:** 完成可复现基准、查询策略与关联策略隔离、可靠拒答协议、跨场景固定确认和失败分析

| result | sample size | budget | validation | scope |
|---|---|---|---|---|
| D15.5-scene-visualization | {"anchor_cameras": 95, "observations": 40, "predicted_objects": 8, "selected_cameras": 20} | {"retrieval_top_k": 24} | PASS | office_loop_engineering_visualization |
| D16-clio-readiness | {"scene_roles": 2, "selected_rgb_frames": 24, "task_metadata_files": 3} | {"available_bytes": 26189692928, "maximum_peak_bytes": 15452274688, "safety_reserve_bytes": 10737418240} | PASS | apartment_development_subset_only_full_modalities_and_cubicle_pending |
| Clio-apartment-GPU-acceptance | {"candidate_outcomes": 24, "evidence_frames": 2, "geometry_frames": 24, "lifted_observations": 3} | {"query": "pillow", "sam_calls": 24} | PASS | real_gpu_development_replay_not_performance |
| D17-relation-reliability | {"negative": 3, "positive": 2, "queries": 5} | {"engineering_threshold": 0.6} | PASS | synthetic_selective_answer_correctness_real_calibration_pending |
| D18-QxA-protocol | {"clio_development_matrix_rows": 6, "frozen_queries": 1, "office_complete_cache_matrix_rows": 6, "synthetic_matrix_rows": 6} | {"Q0": 1, "Q1": [1, 3, 5], "Q2": 5} | PASS | office_and_clio_apartment_complete_cache_development_replay_plus_synthetic_correctness_not_performance |
| D19-ablation-audit | {"a2_variants": 5, "clio_a2_variants": 5, "clio_q2_variants": 3, "frozen_queries": 1, "q2_variants": 3} | {"association_input_q1_k": 5} | PASS | synthetic_correctness_plus_clio_unlabelled_engineering_ablation_real_metrics_pending |
| D20-reproduction-package | {"canonical_inputs": 10, "derived_tables": 3, "stage_validators": 4} | {"cpu_only": true} | PASS | tracked_json_markdown_reproduction |
| Clio-final-object-grounding-association | {"apartment_geometry_frames": 192, "apartment_tasks": 18, "cubicle_geometry_frames": 172, "cubicle_tasks": 18} | {"Q0_sam_calls": 1, "Q1_sam_calls": 5} | PASS | metric_specific_frozen_cubicle_object_grounding_not_general_superiority |
| Clio-final-relation-abstention | {"apartment_queries": 272, "cubicle_negative": 149, "cubicle_positive": 149, "cubicle_queries": 298} | {"uncalibrated_answer_threshold": 0.6} | PASS | fixed_confirmatory_real_relations_calibration_pending |

## Explicit gaps

- clio_download: APARTMENT_AND_CUBICLE_LOCAL_COMPLETE_RAW_NOT_REDISTRIBUTED
- data_license: DATA_LICENSE_UNVERIFIED
- clio_held_out: CUBICLE_OBJECT_GROUNDING_FROZEN_RELATION_FIXED_CONFIRMATORY
- real_calibration: REAL_DATA_CALIBRATION_PENDING
- real_ablation: ASSOCIATION_LABELLED_COMPLETE_QUERY_POLICY_ABLATION_PENDING
- new_gpu_inference: CLIO_APARTMENT_AND_CUBICLE_COMPLETE
- found_it_comparison: OUT_OF_SCOPE_BY_PROJECT_DEFINITION
- optional_binary_release: OPTIONAL_BINARY_RELEASE_PENDING

No Clio held-out or superiority number is claimed.
