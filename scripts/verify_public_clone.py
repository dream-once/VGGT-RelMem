"""Run the test set that a tracked-files-only public clone can reproduce."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import unittest


EXCLUDED_MODULES = {
    "test_q0_protocol": (
        "requires the pinned, Git-ignored third_party/VGGT-SLAM source tree; "
        "run the full suite after bootstrapping upstream sources"
    ),
}


def _iter_tests(suite: unittest.TestSuite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _iter_tests(item)
        else:
            yield item


def _module_name(test: unittest.TestCase) -> str:
    parts = test.id().split(".")
    return next(
        (part for part in parts if part.startswith("test_")),
        parts[0],
    )


def run(*, project_root: str | Path, verbosity: int = 1) -> dict[str, object]:
    root = Path(project_root).resolve()
    discovered = unittest.defaultTestLoader.discover(str(root / "tests"))
    selected = unittest.TestSuite()
    skipped: dict[str, dict[str, object]] = {}
    selected_count = 0
    for test in _iter_tests(discovered):
        module = _module_name(test)
        if module in EXCLUDED_MODULES:
            row = skipped.setdefault(module, {
                "reason": EXCLUDED_MODULES[module],
                "test_count": 0,
            })
            row["test_count"] = int(row["test_count"]) + 1
            continue
        selected.addTest(test)
        selected_count += 1
    result = unittest.TextTestRunner(
        stream=sys.stderr,
        verbosity=verbosity,
    ).run(selected)
    report = {
        "schema_version": "0.1",
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "stage": "public-clean-clone-validation",
        "selected_test_count": selected_count,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "excluded_modules": skipped,
        "scope": "tracked_files_only_cpu_validation",
        "full_suite_command": "python -m unittest discover -s tests -v",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(root))
    parser.add_argument("--verbosity", type=int, default=1)
    args = parser.parse_args()
    return 0 if run(
        project_root=args.project_root,
        verbosity=args.verbosity,
    )["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
