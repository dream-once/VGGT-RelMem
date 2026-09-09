#!/usr/bin/env python3
"""Read-only continuity audit for a VGGT-RelMem cloud instance."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def run(command: list[str], cwd: Path | None = None) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError as error:
        return 127, f"{command[0]} unavailable: {error}"
    return completed.returncode, completed.stdout.strip()


def git(root: Path, *args: str) -> str | None:
    code, output = run(["git", *args], cwd=root)
    return output if code == 0 else None


def find_root(value: str | None) -> Path:
    start = Path(value).expanduser().resolve() if value else Path.cwd().resolve()
    code, output = run(["git", "rev-parse", "--show-toplevel"], cwd=start)
    if code == 0:
        return Path(output).resolve()
    for candidate in (start, *start.parents):
        if (candidate / ".agents" / "vggt-instance-baseline.json").is_file():
            return candidate
    raise RuntimeError(f"cannot locate repository root from {start}")


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def count_files(path: Path) -> int:
    return sum(1 for item in path.rglob("*") if item.is_file())


def inspect_asset(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    path = resolve_path(root, spec["path"])
    result: dict[str, Any] = {
        "path": spec["path"],
        "purpose": spec.get("purpose", ""),
        "present": False,
        "ok": False,
    }
    kind = spec.get("kind", "file")
    if kind == "file":
        if not path.is_file():
            result["reason"] = "missing file"
            return result
        result["present"] = True
        result["size"] = path.stat().st_size
        if result["size"] < int(spec.get("min_size", 0)):
            result["reason"] = f"file is smaller than {spec['min_size']} bytes"
            return result
        if spec.get("sha256"):
            result["sha256"] = sha256(path)
            if result["sha256"] != spec["sha256"]:
                result["reason"] = "SHA-256 mismatch"
                return result
    elif kind == "directory":
        if not path.is_dir():
            result["reason"] = "missing directory"
            return result
        result["present"] = True
        result["files"] = count_files(path)
        if result["files"] < int(spec.get("min_files", 0)):
            result["reason"] = f"directory has fewer than {spec['min_files']} files"
            return result
    else:
        result["reason"] = f"unsupported asset kind: {kind}"
        return result
    result["ok"] = True
    return result


def audit(root: Path, baseline: dict[str, Any]) -> dict[str, Any]:
    status_porcelain = git(root, "status", "--short")
    deleted_output = git(root, "ls-files", "--deleted") or ""
    branch = git(root, "branch", "--show-current") or "(detached)"
    head = git(root, "rev-parse", "HEAD") or "unknown"
    origin = git(root, "remote", "get-url", "origin") or "missing"
    expected_remote = baseline.get("repository", {}).get("remote")
    missing_tracked = [
        value
        for value in baseline.get("tracked_required", [])
        if not (root / value).exists()
    ]

    upstream: list[dict[str, Any]] = []
    upstream_mismatches: list[str] = []
    upstream_missing: list[str] = []
    for spec in baseline.get("upstream_repositories", []):
        path = resolve_path(root, spec["path"])
        row: dict[str, Any] = {
            "path": spec["path"],
            "role": spec.get("role", ""),
            "expected": spec["commit"],
        }
        actual = git(path, "rev-parse", "HEAD") if path.is_dir() else None
        row["actual"] = actual
        row["ok"] = actual == spec["commit"]
        if actual is None:
            row["reason"] = "checkout missing or not a Git repository"
            upstream_missing.append(spec["path"])
        elif actual != spec["commit"]:
            row["reason"] = "commit mismatch"
            upstream_mismatches.append(spec["path"])
        upstream.append(row)

    assets = [inspect_asset(root, spec) for spec in baseline.get("local_assets", [])]
    missing_assets = [row["path"] for row in assets if not row["ok"]]

    disk_spec = baseline.get("disk", {})
    disk_path = resolve_path(root, disk_spec.get("path", str(root)))
    disk: dict[str, Any] = {"path": str(disk_path), "ok": False}
    if disk_path.exists():
        usage = shutil.disk_usage(disk_path)
        free_gib = usage.free / (1024**3)
        disk.update(
            {
                "total_gib": round(usage.total / (1024**3), 2),
                "free_gib": round(free_gib, 2),
                "min_free_gib": disk_spec.get("min_free_gib", 0),
                "ok": free_gib >= float(disk_spec.get("min_free_gib", 0)),
            }
        )
    else:
        disk["reason"] = "disk path missing"

    gpu = "not detected"
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        code, output = run([nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader"])
        if code == 0 and output:
            gpu = output.replace("\n", "; ")

    critical = bool(missing_tracked or deleted_output or upstream_mismatches)
    incomplete = bool(upstream_missing or missing_assets or not disk["ok"])
    dirty = bool(status_porcelain)
    if critical:
        overall = "FAIL"
    elif incomplete:
        overall = "INCOMPLETE"
    elif dirty:
        overall = "DIRTY"
    else:
        overall = "PASS"

    return {
        "status": overall,
        "root": str(root),
        "repository": {
            "branch": branch,
            "head": head,
            "origin": origin,
            "expected_origin": expected_remote,
            "origin_matches": expected_remote is None or origin.rstrip("/") == expected_remote.rstrip("/"),
            "dirty": dirty,
            "status_short": status_porcelain.splitlines() if status_porcelain else [],
            "deleted_tracked": deleted_output.splitlines() if deleted_output else [],
            "missing_required": missing_tracked,
        },
        "upstream_repositories": upstream,
        "local_assets": assets,
        "disk": disk,
        "gpu": gpu,
        "exit_code": 2 if critical else 0,
    }


def print_human(result: dict[str, Any]) -> None:
    repo = result["repository"]
    print(f"VGGT-RelMem instance audit: {result['status']}")
    print(f"repository: {repo['branch']} @ {repo['head']}")
    print(f"origin: {repo['origin']} (match={repo['origin_matches']})")
    print(f"worktree dirty: {repo['dirty']}")
    if repo["status_short"]:
        for line in repo["status_short"]:
            print(f"  {line}")
    print(f"tracked required: {'OK' if not repo['missing_required'] else 'MISSING'}")
    for value in repo["missing_required"]:
        print(f"  missing: {value}")
    for value in repo["deleted_tracked"]:
        print(f"  deleted tracked: {value}")

    print("upstream checkouts:")
    for row in result["upstream_repositories"]:
        marker = "OK" if row["ok"] else "MISSING/MISMATCH"
        print(f"  [{marker}] {row['path']} actual={row['actual']} expected={row['expected']}")

    print("local-only assets:")
    for row in result["local_assets"]:
        marker = "OK" if row["ok"] else "MISSING/INVALID"
        detail = f"size={row['size']}" if "size" in row else f"files={row.get('files', 0)}"
        print(f"  [{marker}] {row['path']} {detail} ({row['purpose']})")
        if row.get("reason"):
            print(f"    {row['reason']}")

    disk = result["disk"]
    print(
        f"disk: {disk['path']} free={disk.get('free_gib', 'unknown')} GiB "
        f"minimum={disk.get('min_free_gib', 'unknown')} GiB ok={disk['ok']}"
    )
    print(f"gpu: {result['gpu']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", help="repository root (auto-detected by default)")
    parser.add_argument("--baseline", help="baseline JSON (defaults to the repository baseline)")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    try:
        root = find_root(args.root)
        baseline_path = (
            Path(args.baseline).expanduser().resolve()
            if args.baseline
            else root / ".agents" / "vggt-instance-baseline.json"
        )
        with baseline_path.open("r", encoding="utf-8") as handle:
            baseline = json.load(handle)
        result = audit(root, baseline)
    except (OSError, RuntimeError, json.JSONDecodeError, KeyError, ValueError) as exc:
        print(f"audit setup failed: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print_human(result)
    return int(result["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
