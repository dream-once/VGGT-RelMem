---
name: vggt-instance-handoff
description: Audit VGGT-RelMem after an AutoDL instance is cloned, replaced, or migrated, and maintain the project's recoverable progress memory during authorized GitHub publishes. Use before starting the next task when the user mentions a new server or possible missing files, and whenever the user explicitly asks to push or upload project updates to GitHub.
---

# VGGT instance handoff

Keep project continuity across disposable cloud instances. Treat `PROJECT_MEMORY.md` as the human-readable progress source and `.agents/vggt-instance-baseline.json` as the machine-readable instance baseline.

## New-instance workflow

1. Locate the repository root and read `PROJECT_MEMORY.md` and `.agents/vggt-instance-baseline.json` completely.
2. Before changing project code, run:

   ```bash
   python .agents/skills/vggt-instance-handoff/scripts/audit_instance.py
   ```

3. Report the following to the user before continuing the requested task:
   - branch, HEAD, origin, dirty state, and deleted tracked files;
   - missing required tracked files;
   - missing or incorrectly pinned upstream source checkouts;
   - missing local-only environments, weights, datasets, and run artifacts;
   - available data-disk space and detected GPU.
4. Treat missing tracked files or a wrong upstream commit as a failed audit. Treat missing ignored environments, weights, data, and generated runs as an incomplete instance, not as lost Git history.
5. Continue only when the files needed by the next task are present. Optional missing assets may be deferred after they are reported.
6. Do not silently download multi-gigabyte weights or datasets. State the expected download and disk impact first, and restore them only when needed for the authorized task.
7. Preserve any dirty worktree changes. Do not reset, delete, reclone, or overwrite them while auditing.

The audit is deliberately read-only and does not fetch from GitHub. If remote freshness matters, inspect or fetch the remote only when network access and the user's task authorize it.

## GitHub publication workflow

Run this workflow only after the user explicitly asks to push, upload, publish, or merge project updates.

1. Inspect `git status`, relevant diffs, the current branch, and the target remote before staging.
2. Update `PROJECT_MEMORY.md` so it records the state being published:
   - current date, branch, and pre-publication HEAD;
   - completed, partial, and next document-day milestones;
   - validation commands and concise evidence;
   - fixed upstream commits;
   - important local-only assets that a fresh clone will not contain;
   - blockers and the concrete next task.
3. Update `.agents/vggt-instance-baseline.json` whenever expected files, pins, asset paths, or minimum disk requirements change.
4. Stage the Skill, memory, and baseline with the code they describe. Keep commits reviewable; do not use `git add .` blindly.
5. Publish through the user's requested branch/PR flow. Then verify the remote branch or merged commit and confirm the memory file is included.
6. Add a short entry to the publication history. Keep only the latest ten entries rather than an unbounded activity diary.

If an authorized upload was completed without its memory update, use the same publication request to make a focused follow-up memory commit unless the user explicitly limited the request.

## Memory safety

- Never record passwords, access tokens, GitHub one-time/device codes, private keys, cookies, authentication files, or private dataset contents.
- Record reproducible facts: commits, paths, sizes, hashes, test results, commands, milestones, and blockers.
- Do not claim a milestone is complete unless its documented acceptance checks actually passed.
