# D16 Clio feasibility and data protocol

> **历史快照（D16，2026-08-29）**：本文件记录下载前的 fail-closed 可行性门槛，
> 不代表当前运行状态。Apartment/Cubicle 后续已完成本地获取、192/172 帧几何、
> 18+18 task 物化与评测；当前状态以 README 和 `evidence/final-clio/` 为准。

## D16 historical decision

At the D16 gate, CPU/source work completed before downloading or installing
Clio. The decision at that historical point was
`DATA_DOWNLOAD_BLOCKED_SIZE_UNKNOWN`; dataset redistribution licensing remains
`DATA_LICENSE_UNVERIFIED`.

The official Clio README confirms that the custom Office, Apartment, Cubicle
and Building scenes contain RGB, depth, a rosbag with poses and task ground
truth. All except Building may also include COLMAP dense reconstruction. The
repository code is BSD-2-Clause, but its README does not explicitly state that
the same licence applies to the linked dataset.

Sources:

- https://github.com/MIT-SPARK/Clio#datasets
- https://github.com/MIT-SPARK/Clio/blob/main/LICENSE
- the official Dropbox URL retained in `configs/clio_dataset_manifest.json`

## Frozen roles and GT boundary

- `apartment`: development scene;
- `cubicle`: historically named held-out, now reported as fixed-confirmatory;
- historical query status: `PENDING_DATA_METADATA` (later resolved locally);
- Clio task labels, OBBs, poses used for alignment, and answers are
  evaluator-only and cannot enter prediction, candidate caches, ObjectMemory
  or policy traces.

The scene roles were frozen at D16. Tracked query manifests were added after
the official task YAML and metadata became locally available.

## Disk gate

The gate is recomputed from live free bytes before any future download:

```text
available_bytes
  - (archive_bytes + extracted_bytes + temporary_bytes)
  >= 10 GiB
```

Unknown size, licence, or checksum blocks the download. The audit does not
delete cached weights, Conda environments, user files, or existing runs.

## CPU acceptance

```bash
cd /root/autodl-tmp/VGGT-RelMem
python -m scripts.audit_clio_feasibility \
  --output /tmp/clio-feasibility.json
python -m scripts.validate_d16 evidence/week3/d16-clio-feasibility
```

A validator `PASS` means the fail-closed contract is correct. It does not
mean that Clio data was downloaded or that held-out experiments were run.
