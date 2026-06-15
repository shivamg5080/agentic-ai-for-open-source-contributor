# Phase 1 — Edit-Location Narrow Prompt (placeholder)

Used by `localize.py` to narrow a chosen file to specific declarations / line
ranges that the issue most likely refers to.

Inputs (filled in at runtime):
- `{{file_path}}`
- `{{declarations}}` — list of exported decls with line numbers
- `{{issue_title}}`
- `{{issue_body}}`

Expected output: JSON array of `{symbol, start_line, end_line}` objects
referencing only declarations from the provided list.

> Replace this placeholder with the final prompt in a later task.
