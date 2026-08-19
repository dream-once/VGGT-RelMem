# Third-party source

VGGT-SLAM is a local checkout of the official MIT-SPARK/VGGT-SLAM repository.
It is ignored by the parent project and must not be presented as original code.

| Component | Pinned commit |
| --- | --- |
| VGGT-SLAM 2.0 | 35327ac28b7d193df9ccc39ba6346052bb6f1207 |
| Dominic101/salad | 33ca9c0ca1e10cbb21efc0d6a5fcb6d45688e42d |
| MIT-SPARK/VGGT_SPARK | 6e6e16107b88e8e76c751826af10d4295d87ecd2 |

The official setup.sh also installs Perception Encoder and SAM3. This project
keeps those optional dependencies out of the geometry-only environment.
