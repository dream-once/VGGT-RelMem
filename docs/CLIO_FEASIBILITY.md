# D16 Clio feasibility and data protocol

## Decision

D16 is complete at CPU/source scope without downloading data or installing
Clio. The current download decision is
`DATA_DOWNLOAD_BLOCKED_SIZE_UNKNOWN`; the dataset licence is separately
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
- `cubicle`: held-out scene;
- query status: `PENDING_DATA_METADATA`;
- Clio task labels, OBBs, poses used for alignment, and answers are
  evaluator-only and cannot enter prediction, candidate caches, ObjectMemory
  or policy traces.

Only scene roles are frozen. Query IDs are deliberately empty until the
official task YAML and metadata are legally and safely available.

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
