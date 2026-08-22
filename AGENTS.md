# Project continuity

- When the user says this is a new, cloned, replaced, or migrated AutoDL instance, use the `vggt-instance-handoff` skill before starting the requested project task.
- When the user explicitly asks to push or upload project updates to GitHub, use that skill so `PROJECT_MEMORY.md` and the instance baseline are updated in the same publication.
- Never publish merely because the memory was updated. GitHub writes still require the user's explicit request.
- Never store credentials, GitHub device codes, tokens, private keys, or authentication files in project memory.
