# D21.1 Clio pillow diagnostic

This lightweight bundle records a real RTX 4090 development diagnostic on the
24-frame Clio `apartment` subset. It isolates SAM prompt/threshold sensitivity
from 3D association. It is not a formal prompt-policy change, held-out evidence,
segmentation recall, Grounding Acc@1, or a performance-improvement claim.

The frozen `pillow@0.5` run bitwise replays the retained three masks in two
frames. `dinosaur pillow@0.5` yields five visually reviewed masks in five other
frames; all cover the same physical green dinosaur pillow. The task phrase
`bring me a pillow` yields zero masks. Across all six experiments, only 8/24
frames contain a mask; frame visibility is still awaiting manual annotation.

With the five clean instance-specific masks, frozen A2 and development A2.1 both
form one permanent object and classify all ten positive pairs correctly. There
are no negative pairs, so A2 remains the formal method and A2.1 is not promoted.

The recovered COLMAP database lists 1,845 RGB images, but only 786 RGB files are
local and no sparse poses or rosbag are available. Full-trajectory evaluator-only
Sim(3) alignment therefore remains blocked. Raw data, masks, points, images and
video are not redistributed.

Validate with:

```bash
python -m scripts.validate_d21_1_pillow_diagnostic \
  evidence/week4/d21_1-pillow-diagnostic/public_report.json
```
