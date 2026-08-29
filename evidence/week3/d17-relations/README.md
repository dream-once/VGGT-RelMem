# D17 relation and abstention evidence

The prediction bundle reads only a synthetic ObjectMemory, anchor poses,
label-free queries and the frozen uncalibrated engineering manifest. Labels are
stored separately and are opened only by the evaluator.

This fixture has two answerable directional queries and three negative queries
covering missing target, missing reference and missing anchor. Correct negative
rejections count as task success. Metrics validate definitions and isolation;
they are not real-scene performance or calibration evidence.
