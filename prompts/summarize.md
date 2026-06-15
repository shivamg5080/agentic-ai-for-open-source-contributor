# Phase 4 — PR Summary Prompt (placeholder)

Used by `summarize.py` to draft a PR title + body from the winning diff,
matching the repository's recent-PR style.

Inputs (filled in at runtime):
- `{{issue_number}}`
- `{{issue_title}}`
- `{{issue_body}}`
- `{{winning_diff}}`
- `{{recent_prs}}` — 2-3 recent merged PR titles/bodies for style cues

Expected output:
- Line 1: a single-line PR title.
- Blank line.
- Body that references the issue number (e.g. `Fixes #{{issue_number}}`)
  and briefly describes the change.

> Replace this placeholder with the final prompt in a later task.
