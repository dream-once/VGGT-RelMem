import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from relground.segmentation_sweep import (
    build_sweep_plan,
    derive_selection,
    validate_prompt_config,
    validate_sweep_plan,
)


class SegmentationSweepTests(unittest.TestCase):
    def config(self):
        return {
            "schema_version": "0.1",
            "scene_id": "clio-apartment",
            "split_role": "development_calibration_only",
            "source_task": "bring me a pillow",
            "candidate_universe": "same frozen 24-frame apartment geometry",
            "experiments": [
                {
                    "experiment_id": "baseline-pillow-050",
                    "query": "pillow",
                    "sam_threshold": 0.5,
                    "role": "frozen_upstream_threshold_baseline",
                },
                {
                    "experiment_id": "task-phrase-050",
                    "query": "bring me a pillow",
                    "sam_threshold": 0.5,
                    "role": "generic_task_phrase_diagnostic",
                },
            ],
            "guards": {
                "q0_threshold_remains_0_5": True,
                "no_cubicle_access_or_tuning": True,
                "same_geometry_and_candidate_universe": True,
                "formal_prompt_policy_must_not_use_image_specific_description": True,
            },
        }

    def selection(self):
        return {
            "stage": "D5",
            "query": "pillow",
            "requested_k": 2,
            "selected_count": 2,
            "frames": [
                {"rank": 1, "frame_id": "rgb_1", "geometry_index": 0},
                {"rank": 2, "frame_id": "rgb_2", "geometry_index": 1},
            ],
        }

    def make_plan(self, root: Path):
        config_path = root / "config.json"
        selection_path = root / "selection.json"
        output = root / "runs/sweep"
        config = self.config()
        selection = self.selection()
        config_path.write_text(json.dumps(config))
        selection_path.write_text(json.dumps(selection))
        for experiment in validate_prompt_config(config):
            path = output / "selections" / f"{experiment['experiment_id']}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(derive_selection(selection, experiment["query"])))
        plan = build_sweep_plan(
            project_root=root,
            config_path=config_path,
            source_selection_path=selection_path,
            output_root=output,
            created_at="fixed",
        )
        return plan, output

    def test_plan_preserves_candidate_universe_and_validates(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            plan, _ = self.make_plan(root)
            self.assertEqual(plan["candidate_frame_ids"], ["rgb_1", "rgb_2"])
            self.assertEqual(validate_sweep_plan(plan, project_root=root)["status"], "PASS")

    def test_derived_selection_tampering_fails(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            plan, _ = self.make_plan(root)
            path = root / plan["experiments"][1]["derived_selection_ref"]
            payload = json.loads(path.read_text())
            payload["frames"].reverse()
            path.write_text(json.dumps(payload))
            self.assertEqual(validate_sweep_plan(plan, project_root=root)["status"], "FAIL")

    def test_baseline_or_held_out_guard_cannot_change(self):
        config = self.config()
        config["experiments"][0]["sam_threshold"] = 0.4
        with self.assertRaises(ValueError):
            validate_prompt_config(config)
        config = self.config()
        config["guards"]["no_cubicle_access_or_tuning"] = False
        with self.assertRaises(ValueError):
            validate_prompt_config(config)


if __name__ == "__main__":
    unittest.main()
