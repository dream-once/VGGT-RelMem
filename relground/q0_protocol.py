"""D13 frozen Q0 upstream-aligned protocol and static audit helpers."""

from __future__ import annotations

import ast
import copy
from pathlib import Path
from typing import Any, Mapping
import json

from .observation_cache import sha256_file


Q0_SCHEMA_VERSION = "0.1"
Q0_PROTOCOL_ID = "Q0-vggt-slam-upstream-top1"
Q0_PROTOCOL_STATUS = "upstream-aligned"
Q0_FIELDS = (
    "schema_version",
    "protocol_id",
    "status",
    "scope",
    "claims",
    "selection",
    "preprocess",
    "segmentation",
    "lifting",
    "obb",
    "forbidden",
    "source_pins",
    "source_files",
    "source_checks",
    "retained_d4",
    "development_selection",
    "limitations",
    "created_at",
)
SOURCE_PATHS = (
    "third_party/VGGT-SLAM/main.py",
    "third_party/VGGT-SLAM/vggt_slam/map.py",
    "third_party/VGGT-SLAM/vggt_slam/solver.py",
    "third_party/VGGT-SLAM/vggt_slam/submap.py",
    "third_party/VGGT-SLAM/vggt_slam/slam_utils.py",
    "third_party/VGGT-SLAM/third_party/vggt/vggt/utils/load_fn.py",
    "adapters/open_vocab.py",
    "relground/single_view.py",
    "scripts/run_single_view_baselines.py",
)
EVIDENCE_PATHS = {
    "result": "evidence/week1/d4-single-view/single_view_result.json",
    "preprocess": "evidence/week1/d4-single-view/preprocess.json",
    "run_manifest": "evidence/week1/d4-single-view/run_manifest.json",
    "saved_validation": "evidence/week1/validation/d4-single-view.json",
    "d5_retrieval": "evidence/week1/d5-multiview/trash-can/retrieval.json",
    "d5_validation": "evidence/week1/validation/d5-trash-can.json",
}
FORBIDDEN_Q0_OPERATIONS = (
    "confidence_gate",
    "radial_mad_filter",
    "minimum_point_gate",
    "post_sam_mask_resize",
    "robust_pca",
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} root must be an object")
    return payload


def _function_source(path: Path, name: str) -> str:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            lines = text.splitlines()
            return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    raise ValueError(f"missing function {name} in {path}")


def audit_source_semantics(project_root: str | Path) -> dict[str, bool]:
    root = Path(project_root).resolve()
    main = (root / SOURCE_PATHS[0]).read_text(encoding="utf-8")
    map_text = (root / SOURCE_PATHS[1]).read_text(encoding="utf-8")
    solver = (root / SOURCE_PATHS[2]).read_text(encoding="utf-8")
    submap_fn = _function_source(root / SOURCE_PATHS[3], "get_points_in_mask")
    upstream_pca = _function_source(
        root / SOURCE_PATHS[4], "compute_obb_from_points"
    )
    load_fn = _function_source(
        root / SOURCE_PATHS[5], "load_and_preprocess_images"
    )
    local_top1 = _function_source(root / SOURCE_PATHS[6], "select_top1")
    local_scores = _function_source(root / SOURCE_PATHS[6], "score_frames")
    local_lift = _function_source(root / SOURCE_PATHS[7], "official_pca_lift")
    runner = (root / SOURCE_PATHS[8]).read_text(encoding="utf-8")
    local_lift_lower = local_lift.lower()
    return {
        "upstream_run_os_entry": (
            'parser.add_argument("--run_os"' in main
            and "retrieve_best_semantic_frame" in main
        ),
        "upstream_nonnegative_top1": all(
            token in map_text
            for token in (
                "overall_best_score = 0.0",
                "best_score_id = np.argmax(scores)",
                "if best_score > overall_best_score",
            )
        ),
        "local_nonnegative_top1": (
            "np.clip(cosine, 0.0, 1.0)" in local_scores
            and "np.argmax" in local_top1
        ),
        "upstream_vggt_preprocessed_sam_image": all(
            token in solver
            for token in (
                "load_and_preprocess_images(image_names)",
                "new_submap.add_all_frames(images)",
            )
        ) and all(
            token in main
            for token in (
                "get_frame_at_index(overall_best_frame_index)",
                "processor.set_image(best_img)",
            )
        ),
        "upstream_518_crop_batch_pad": all(
            token in load_fn
            for token in (
                'mode="crop"',
                "target_size = 518",
                "start_y = (new_height - target_size) // 2",
                'mode="constant", value=1.0',
                "images = torch.stack(images)",
            )
        ),
        "upstream_sam_threshold_0_5": (
            "Sam3Processor(sam3_model, confidence_threshold=0.50)" in main
        ),
        "upstream_direct_mask_indexing": all(
            token in submap_fn
            for token in (
                "points.reshape(-1, 3)",
                "mask.reshape(-1)",
                "points_flat[mask_flat]",
            )
        ),
        "upstream_finite_only_pca_obb": all(
            token in upstream_pca
            for token in (
                "np.isfinite(points).all(axis=1)",
                "np.cov(centered, rowvar=False)",
                "np.linalg.eigh(cov)",
            )
        ),
        "local_q0_direct_finite_pca": all(
            token in local_lift
            for token in (
                "points_array.reshape(-1, 3)[mask_array.reshape(-1)]",
                "np.all(np.isfinite(points), axis=1)",
                "np.cov(centered, rowvar=False)",
                "np.linalg.eigh(covariance)",
            )
        ) and not any(
            token in local_lift_lower
            for token in ("confidence", "mad", "min_points", "robust")
        ),
        "local_q0_no_post_sam_resize": all(
            token in runner
            for token in (
                "controlled SAM masks must directly match the VGGT point grid",
                '"mask_resizing_after_sam": False',
                "make_official_observation(",
            )
        ),
    }


def validate_q0_payload(payload: Mapping[str, Any]) -> None:
    if tuple(payload) != Q0_FIELDS or set(payload) != set(Q0_FIELDS):
        raise ValueError("Q0 protocol fields are not frozen")
    if payload["schema_version"] != Q0_SCHEMA_VERSION:
        raise ValueError("unsupported Q0 protocol schema")
    if payload["protocol_id"] != Q0_PROTOCOL_ID:
        raise ValueError("unexpected Q0 protocol id")
    if payload["status"] != Q0_PROTOCOL_STATUS:
        raise ValueError("Q0 must remain upstream-aligned")
    claims = payload["claims"]
    if (
        not isinstance(claims, Mapping)
        or claims.get("found_it_official") is not False
        or claims.get("vggt_slam_official_reproduction") is not False
        or claims.get("local_b0_label_scope")
        != "audited single-view lifting baseline only"
    ):
        raise ValueError("Q0 claim boundaries changed")
    if payload["selection"] != {
        "encoder": "PE-Core-L14-336",
        "similarity": "cosine",
        "score_rule": "clip cosine to [0,1] / upstream zero floor",
        "selection": "Top-1",
        "tie_break": "first geometry-order frame",
    }:
        raise ValueError("Q0 selection protocol changed")
    if payload["preprocess"] != {
        "mode": "VGGT crop",
        "target_size": 518,
        "patch_size": 14,
        "resize": "width=518 preserve aspect ratio",
        "center_crop": "height only when >518",
        "batch_pad": "symmetric white padding to batch max shape",
        "sam_input": "stored VGGT-preprocessed frame",
    }:
        raise ValueError("Q0 preprocess protocol changed")
    if payload["segmentation"] != {
        "model": "SAM3",
        "confidence_threshold": 0.5,
        "mask_grid": "same as VGGT point grid",
        "post_sam_mask_resize": False,
    }:
        raise ValueError("Q0 segmentation protocol changed")
    if payload["lifting"] != {
        "mask_indexing": "direct",
        "point_filter": "finite-only",
        "confidence_gate": False,
        "mad_filter": False,
        "minimum_point_gate": False,
    }:
        raise ValueError("Q0 lifting protocol changed")
    if payload["obb"] != {
        "method": "ordinary PCA",
        "covariance": "np.cov centered points",
        "eigendecomposition": "np.linalg.eigh descending eigenvalues",
    }:
        raise ValueError("Q0 OBB protocol changed")
    if tuple(payload["forbidden"]) != FORBIDDEN_Q0_OPERATIONS:
        raise ValueError("Q0 forbidden operations changed")
    files = payload["source_files"]
    if not isinstance(files, list) or [item.get("path") for item in files] != list(SOURCE_PATHS):
        raise ValueError("Q0 source-file inventory changed")
    checks = payload["source_checks"]
    if not isinstance(checks, Mapping) or not checks or not all(
        value is True for value in checks.values()
    ):
        raise ValueError("Q0 source checks are incomplete")
    if not payload["development_selection"].get("top1_matches_d5_first"):
        raise ValueError("Q0 Top-1 no longer matches D5 raw rank 1")
    limitations = payload["limitations"]
    if not isinstance(limitations, list) or len(limitations) < 2:
        raise ValueError("Q0 audit limitations must remain explicit")
    if not str(payload["created_at"]).strip():
        raise ValueError("Q0 created_at is required")


def build_q0_protocol(
    project_root: str | Path,
    *,
    created_at: str,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    d4 = _read_json(root / EVIDENCE_PATHS["result"])
    preprocess = _read_json(root / EVIDENCE_PATHS["preprocess"])
    run_manifest = _read_json(root / EVIDENCE_PATHS["run_manifest"])
    saved_validation = _read_json(root / EVIDENCE_PATHS["saved_validation"])
    d5 = _read_json(root / EVIDENCE_PATHS["d5_retrieval"])
    d5_validation = _read_json(root / EVIDENCE_PATHS["d5_validation"])
    source_checks = audit_source_semantics(root)
    source_files = [
        {
            "path": path,
            "sha256": sha256_file(root / path),
        }
        for path in SOURCE_PATHS
    ]
    first = d5["raw_ranking"][0]
    upstream_top1 = d5["upstream_top1"]
    retained_refs = {
        key: {
            "path": path,
            "sha256": sha256_file(root / path),
        }
        for key, path in EVIDENCE_PATHS.items()
    }
    payload = {
        "schema_version": Q0_SCHEMA_VERSION,
        "protocol_id": Q0_PROTOCOL_ID,
        "status": Q0_PROTOCOL_STATUS,
        "scope": "VGGT-SLAM --run_os aligned Top-1 single-view query path",
        "claims": {
            "found_it_official": False,
            "vggt_slam_official_reproduction": False,
            "local_b0_label_scope": "audited single-view lifting baseline only",
        },
        "selection": {
            "encoder": "PE-Core-L14-336",
            "similarity": "cosine",
            "score_rule": "clip cosine to [0,1] / upstream zero floor",
            "selection": "Top-1",
            "tie_break": "first geometry-order frame",
        },
        "preprocess": {
            "mode": "VGGT crop",
            "target_size": 518,
            "patch_size": 14,
            "resize": "width=518 preserve aspect ratio",
            "center_crop": "height only when >518",
            "batch_pad": "symmetric white padding to batch max shape",
            "sam_input": "stored VGGT-preprocessed frame",
        },
        "segmentation": {
            "model": "SAM3",
            "confidence_threshold": 0.5,
            "mask_grid": "same as VGGT point grid",
            "post_sam_mask_resize": False,
        },
        "lifting": {
            "mask_indexing": "direct",
            "point_filter": "finite-only",
            "confidence_gate": False,
            "mad_filter": False,
            "minimum_point_gate": False,
        },
        "obb": {
            "method": "ordinary PCA",
            "covariance": "np.cov centered points",
            "eigendecomposition": "np.linalg.eigh descending eigenvalues",
        },
        "forbidden": list(FORBIDDEN_Q0_OPERATIONS),
        "source_pins": {
            "vggt_slam": "35327ac28b7d193df9ccc39ba6346052bb6f1207",
            "vggt": "6e6e16107b88e8e76c751826af10d4295d87ecd2",
            "perception_models": "3e352cca660658d4b5c90f42a7808b11469e4c66",
            "sam3": "8f0b7f4d4e7eda2ed606ebde6702c93359ad01da",
        },
        "source_files": source_files,
        "source_checks": source_checks,
        "retained_d4": {
            "artifacts": retained_refs,
            "query": d4["query"],
            "top1_frame": d4["top1"]["frame_id"],
            "sam_threshold": run_manifest["config"]["sam_threshold"],
            "sam_mask_shape": d4["controlled_inputs"]["sam_mask_shape"],
            "mask_resizing_after_sam": d4["controlled_inputs"]["mask_resizing_after_sam"],
            "preprocess_mode": preprocess["transform"]["mode"],
            "preprocess_target_size": preprocess["transform"]["target_size"],
            "b0_method": d4["baselines"]["B0-official"]["method"],
            "saved_validator_status": saved_validation["status"],
            "strict_validator_rerun": "FAIL_MISSING_MASKS_AND_PREVIEW",
        },
        "development_selection": {
            "query": d5["query"],
            "raw_rank_1_frame": first["frame_id"],
            "raw_rank_1_score": first["retrieval_score"],
            "upstream_top1_frame": upstream_top1["frame_id"],
            "upstream_top1_score": upstream_top1["retrieval_score"],
            "top1_matches_d5_first": (
                first["frame_id"] == upstream_top1["frame_id"]
                and first["geometry_index"] == upstream_top1["geometry_index"]
                and first["retrieval_score"] == upstream_top1["retrieval_score"]
                and d5_validation["status"] == "PASS"
            ),
        },
        "limitations": [
            "The retained lightweight D4 bundle intentionally omits masks.json and preview.png, so the original strict binary validator cannot rerun; the saved historical report is PASS while the current rerun fails only for those missing artifacts.",
            "The audit proves alignment to the pinned VGGT-SLAM --run_os path and retained local JSON, not an official FOUND-IT implementation or byte-identical interactive upstream execution.",
        ],
        "created_at": created_at,
    }
    validate_q0_payload(payload)
    return copy.deepcopy(payload)
