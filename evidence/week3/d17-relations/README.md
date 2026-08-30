# D17 relation and abstention evidence

The prediction bundle reads only a synthetic ObjectMemory, anchor poses,
label-free queries and the frozen uncalibrated engineering manifest. Labels are
stored separately and are opened only by the evaluator.

This fixture has two answerable directional queries and three structural
negative queries covering missing target, missing reference and missing anchor.
Correct negative rejections count as task success, but are excluded from the
selective **answer** coverage curve. Coverage is the number of answered queries
divided by all five frozen queries; the retained curve therefore ends at `0.4`.
Raw `answer_confidence` is never inverted after an abstention.

The reported Brier/ECE values separate answerability-proxy calibration from
correctness calibration on answered queries. The fixture validates definitions
and label isolation only; it is not real-scene performance or an independently
fitted calibration result.
