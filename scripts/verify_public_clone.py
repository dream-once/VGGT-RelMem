"""Run the curated core tests that a tracked-files-only clone can reproduce."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import unittest



def run(*, project_root: str | Path, verbosity: int = 1) -> dict[str, object]:
    root = Path(project_root).resolve()
    discovered = unittest.defaultTestLoader.discover(str(root / "tests"))
    selected_count = discovered.countTestCases()
    result = unittest.TextTestRunner(
        stream=sys.stderr,
        verbosity=verbosity,
    ).run(discovered)
    report = {
        "schema_version": "0.1",
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "stage": "public-clean-clone-validation",
        "selected_test_count": selected_count,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "excluded_modules": {},
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
