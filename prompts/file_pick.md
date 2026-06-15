# Phase 1 — File Pick Prompt (placeholder)

Used by `localize.py` to ask the LLM to choose the top-N most suspicious files
from a repository skeleton + issue title/body.

Inputs (filled in at runtime):
- `{{repo}}`
- `{{issue_title}}`
- `{{issue_body}}`
- `{{skeleton}}` — directory tree + per-file outlines (exported decls only)
- `{{top_n}}`

Expected output: JSON array of file paths (length <= top_n), each path drawn
verbatim from the skeleton. No prose, no commentary.

> Replace this placeholder with the final prompt in a later task.
