# Phase 2 — Repair Prompt (placeholder)

Used by `repair.py` to ask the LLM for a fix expressed as strict
SEARCH/REPLACE blocks (chosen over unified diffs since they apply more
reliably).

Inputs (filled in at runtime):
- `{{issue_title}}`
- `{{issue_body}}`
- `{{edit_locations}}` — file paths + surrounding code context
- `{{conventions_hint}}` — short note on Go style / package conventions

Expected output: one or more SEARCH/REPLACE blocks of the form

```
<<<<<<< SEARCH
<exact existing text>
=======
<replacement text>
>>>>>>> REPLACE
```

with the file path on the line immediately above each block. No prose
between blocks.

> Replace this placeholder with the final prompt in a later task.
