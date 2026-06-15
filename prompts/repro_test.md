# Phase 3 — Reproduction Test Prompt (placeholder)

Used by `validate.py` to (optionally) generate a Go test that reproduces the
issue — failing on the base checkout and passing on a correct fix.

Inputs (filled in at runtime):
- `{{issue_title}}`
- `{{issue_body}}`
- `{{relevant_files}}` — minimal context drawn from edit locations
- `{{package}}` — target Go package

Expected output: a single Go test file body. Must compile against the
target package and use only the standard `testing` package unless an
import is clearly justified by the issue.

> Replace this placeholder with the final prompt in a later task.
