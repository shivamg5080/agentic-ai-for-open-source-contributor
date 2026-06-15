# Implementation Plan: Agentless Go Contributor

## Overview

This plan builds the four-phase Agentless pipeline (localize → repair → validate/select → summarize) plus the evaluation harness in **Python 3.11+**, as a sequence of small, runnable, test-driven slices. Each slice adds one component, wires it into the growing system, and (where the design defines a correctness property) is covered by a Hypothesis property test placed next to the implementation it guards. The smallest runnable slice — shared config + LLM wrapper — comes first; the README is written last once the behavior is settled.

Requirement references use the five reviewer criteria and the reproducibility/reliability goals documented in the design (the `requirements.md` document is currently empty). Property tasks additionally cite their numbered Correctness Property from `design.md`.

## Tasks

- [x] 1. Project scaffolding and shared infrastructure
  - [x] 1.1 Create project structure and dependency manifests
    - Create `src/` package (`src/__init__.py`), `prompts/` directory (placeholder prompt templates for file-pick, narrow, repair, repro-test, summarize), `eval/cases/`, and `outputs/`
    - Author `requirements.txt` (anthropic, PyGithub/requests, PyYAML, hypothesis, pytest)
    - Author `config.yaml` with repo allowlist, default `issue_number: 4460`, `top_n_files`, and `llm` block (model/temperature/sample_count/max_retries/request_timeout_s)
    - _Design: Dependencies, Components/config.py; Requirements: reproducibility_
  - [x] 1.2 Implement `src/config.py`
    - Define frozen `LLMConfig` and `Config` dataclasses per the design interface
    - Implement `load_config(path, overrides)`: parse YAML, apply CLI overrides, source `ANTHROPIC_API_KEY` from env only, validate required fields and repo allowlist, fail fast with actionable messages
    - _Design: Components/config.py, Security Considerations; Requirements: reproducibility_
  - [ ]* 1.3 Write unit tests for `config.py`
    - Required-field validation, CLI override precedence, env-var sourcing, allowlist rejection
    - _Design: Testing Strategy/Unit; Requirements: reproducibility_
  - [x] 1.4 Implement `src/llm.py`
    - Define `LLMResponse`; implement `complete(prompt, cfg, temperature, n)` and `complete_json(prompt, cfg, schema)`
    - Centralize retry/backoff (`max_retries`), JSON-parse-with-repair, `n`-sample support, and prompt/response logging to `outputs/`
    - _Design: Components/llm.py, Error Handling/Malformed LLM output_
  - [ ]* 1.5 Write unit tests for `llm.py`
    - Mock client: retry/backoff path, JSON repair, `n`-sample fan-out, log emission
    - _Design: Testing Strategy/Unit_

- [x] 2. Issue ingestion
  - [x] 2.1 Implement `src/issue.py`
    - Define `Issue` and `MergedPRRef`; implement `fetch_issue(repo, number, offline, cache_dir)` (GitHub REST + disk cache) and `load_issue_json(path)` offline fallback
    - Enforce validation rules: `repo` matches `owner/name` and is in allowlist, positive `number`, non-empty `title`, string `body`
    - _Design: Components/issue.py, Data Models/Issue, Error Handling/Issue not found; Requirements: Criterion 1, reviewability_
  - [ ]* 2.2 Write unit tests for `issue.py`
    - Offline JSON load, cache hit/miss, validation failures, 404/network fail-fast message
    - _Design: Testing Strategy/Unit, Error Handling_

- [x] 3. Repository preparation
  - [x] 3.1 Implement `src/repo.py`
    - Define `RepoCheckout`; implement `prepare_repo(repo, base_commit, workdir)` (clone or reuse cached clone, `git checkout` pinned base commit), `clean_worktree(checkout)` (isolated fresh copy), `make_diff(checkout_path)` (unified diff vs base)
    - Use argument arrays for all git invocations (no shell interpolation)
    - _Design: Components/repo.py, Security Considerations; Requirements: reproducibility_
  - [ ]* 3.2 Write property test for base immutability
    - **Property 1: Base immutability** — across operations the base checkout at `base_commit` is byte-identical and every edit lands only in an isolated worktree
    - **Validates: reproducibility / fair comparison**
    - _Design: Correctness Properties/Property 1_
  - [ ]* 3.3 Write unit tests for `repo.py`
    - Worktree isolation (edits don't leak to base), `make_diff` produces valid unified diff, pinned checkout uses the requested commit; use a tiny checked-in fixture Go module
    - _Design: Testing Strategy/Unit + fixtures; Requirements: reproducibility_

- [x] 4. Phase 1 — hierarchical localization
  - [x] 4.1 Implement skeleton builder in `src/localize.py`
    - Define `FileOutline`, `RepoSkeleton`, `EditLocation`, `LocalizationResult`; implement `build_skeleton(checkout)` producing directory tree + per-file outlines (package + exported decls only, never full bodies) and `approx_tokens`
    - _Design: Components/localize.py, Key Functions/build_skeleton; Requirements: Criterion 1_
  - [ ]* 4.2 Write unit tests for `build_skeleton`
    - Outlines contain exported decls only, tree lists every `.go` file, `approx_tokens` non-negative
    - _Design: Testing Strategy/Unit, Key Functions/build_skeleton_
  - [x] 4.3 Implement `localize()` with LLM file-pick, edit-location narrowing, and ripgrep fallback
    - Within-budget path: LLM picks top-N files from skeleton; over-budget path: ripgrep seeds over symbols/error strings then LLM pick (`used_fallback=True`)
    - Narrow each ranked file to declarations/line ranges via LLM and attach surrounding context
    - _Design: Algorithmic Pseudocode/Phase 1, Components/localize.py, Error Handling/Skeleton budget; Requirements: Criterion 1_
  - [ ]* 4.4 Write property test for localization soundness
    - **Property 2: Localization soundness** — every `EditLocation.file_path` exists, ends in `.go`, and `1 <= start_line <= end_line <= file_line_count`; no hallucinated files
    - **Validates: Criterion 1 (right files identified)**
    - _Design: Correctness Properties/Property 2_
  - [ ]* 4.5 Write unit tests for `localize`
    - Within-budget vs ripgrep-fallback branch selection, `ranked_files` length ≤ `top_n_files`, only skeleton/seed-derived files iterated
    - _Design: Testing Strategy/Unit, Algorithmic Pseudocode/Phase 1_

- [x] 5. Checkpoint
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Phase 2 — repair
  - [x] 6.1 Implement `src/patch.py` (SEARCH/REPLACE parse + atomic apply)
    - Define `SearchReplaceBlock`, `ApplyResult`; implement `parse_blocks(llm_text)` (strict format) and `apply_blocks(worktree, blocks)` with exact-match-once semantics and atomic rollback on any zero/non-unique match
    - _Design: Components/patch.py, Key Functions/apply_blocks; Requirements: Criterion 2_
  - [ ]* 6.2 Write property test for patch atomicity
    - **Property 3: Patch atomicity** — for block lists where any block's search is non-matching, `apply_blocks` leaves the worktree unchanged; otherwise all blocks apply
    - **Validates: Criterion 2 (relevant code changes)**
    - _Design: Correctness Properties/Property 3, Testing Strategy/Property-Based_
  - [ ]* 6.3 Write unit tests for `patch.py`
    - Exact match, non-unique match, zero match, multi-file blocks, rollback leaves worktree byte-identical
    - _Design: Testing Strategy/Unit_
  - [x] 6.4 Implement `src/repair.py` (multi-candidate generation + format prefilter)
    - Define `Candidate`; implement `generate_candidates(...)`: LLM emits `n` SEARCH/REPLACE samples at moderate temperature, parse, apply each to a fresh `clean_worktree`, run `gofmt`/`goimports`, normalize diff, mark `applied_clean`; discard non-applying/malformed samples
    - _Design: Algorithmic Pseudocode/Phase 2, Components/repair.py, Data Models/Candidate_
  - [ ]* 6.5 Write property test for applied-implies-formatted
    - **Property 4: Applied implies formatted** — for every candidate with `applied_clean == True`, `gofmt -l` over the changed files lists nothing
    - **Validates: Criterion 3 (project conventions followed)**
    - _Design: Correctness Properties/Property 4_
  - [ ]* 6.6 Write unit tests for `repair.py`
    - Fresh-worktree-per-candidate isolation, malformed-sample skip, `applied_clean` candidates have non-empty normalized diff
    - _Design: Testing Strategy/Unit, Algorithmic Pseudocode/Phase 2_

- [x] 7. Phase 3 — validation and selection
  - [x] 7.1 Implement `src/validate.py` (layered Go-toolchain checks + repro test)
    - Define `CheckName`, `CheckResult`, `ValidationReport`; implement `generate_repro_test(...)` (optional) and `validate(...)`: per candidate run REPRO_TEST then BUILD → VET → FMT → TEST → LINT in fixed order, short-circuit remaining checks on BUILD failure, mark LINT skipped when `golangci-lint` absent, compute `breaks_existing_tests`
    - Use argument arrays for all tool invocations
    - _Design: Algorithmic Pseudocode/Phase 3, Components/validate.py, Key Functions/run_check, Data Models/ValidationReport; Requirements: Criterion 4_
  - [ ]* 7.2 Write property test for validation layering
    - **Property 7: Validation layering** — if BUILD fails, VET/TEST/LINT are never reported as passed for that candidate
    - **Validates: Criterion 4 (appropriate validation run)**
    - _Design: Correctness Properties/Property 7, Testing Strategy/Property-Based_
  - [ ]* 7.3 Write unit tests for `validate.run_check`
    - Exit-code → passed mapping, `gofmt -l` lists-files semantics, skipped-tool (`golangci-lint`) behavior, BUILD short-circuit
    - _Design: Testing Strategy/Unit, Key Functions/run_check_
  - [x] 7.4 Implement `src/select.py` (ranking, tie-break, regression drop)
    - Define `RankedCandidate`; implement `score_key` (repro > existing tests > vet/build > lint > fmt, lexicographic) and `select(...)`: drop candidates that break existing tests, sort descending, break top-score ties by majority vote over normalized diffs then stable order, return single winner or None
    - _Design: Algorithmic Pseudocode/Phase 3, Components/select.py; Requirements: Criterion 4, reliability_
  - [ ]* 7.5 Write property test for ranking monotonicity
    - **Property 6: Ranking monotonicity** — if A's check outcomes dominate B's in priority order, A is ranked at least as high as B
    - **Validates: Criterion 4 (appropriate validation run)**
    - _Design: Correctness Properties/Property 6, Testing Strategy/Property-Based_
  - [ ]* 7.6 Write property test for deterministic selection
    - **Property 8: Deterministic selection given reports** — shuffling the input order of a fixed report set yields the same winner
    - **Validates: reliability**
    - _Design: Correctness Properties/Property 8, Testing Strategy/Property-Based_
  - [ ]* 7.7 Write property test for no-regressions-selected
    - **Property 5: No regressions selected** — any returned winner does not break a test that passed on the base checkout
    - **Validates: Criteria 3 & 4**
    - _Design: Correctness Properties/Property 5_
  - [ ]* 7.8 Write unit tests for `select.py`
    - Tie-break majority vote, regression candidates excluded, `None` when all break tests
    - _Design: Testing Strategy/Unit_

- [x] 8. Checkpoint
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Phase 4 — PR summary
  - [x] 9.1 Implement `src/summarize.py`
    - Define `PRSummary`; implement `fetch_recent_merged_prs(repo, k, offline)`, `derive_style(recent_prs)`, `summarize(winning_diff, issue, recent_prs, cfg)`, and `parse_title_body(...)` — single-line non-empty title, body references the issue number
    - _Design: Algorithmic Pseudocode/Phase 4, Components/summarize.py; Requirements: Criterion 5_
  - [ ]* 9.2 Write property test for summary-references-issue
    - **Property 9: Summary references issue** — for successful runs the PR body references the issue number and the title is a single non-empty line
    - **Validates: Criterion 5 (reasonable PR summary)**
    - _Design: Correctness Properties/Property 9_
  - [ ]* 9.3 Write unit tests for `summarize.py`
    - `parse_title_body` extraction, style derivation from recent PRs, empty-`recent_prs` handling
    - _Design: Testing Strategy/Unit_

- [x] 10. CLI orchestration
  - [x] 10.1 Implement `src/main.py` (`python -m src.main --issue N`)
    - Wire `run_pipeline(cfg)`: fetch issue → prepare repo → localize → generate/filter candidates → repro test + validate → select → fetch recent PRs → summarize → `write_outputs`
    - Parse CLI args/overrides; emit per-phase artifacts and logs to `outputs/`; return terminal `RunResult` status (`success` / `no_applicable_patch` / `no_valid_patch`)
    - _Design: Architecture, Algorithmic Pseudocode/Main pipeline, Example Usage; Requirements: Criteria 1–5_
  - [ ]* 10.2 Write integration test (offline fixture, end-to-end)
    - Run fully offline against a checked-in fixture Go repo + synthetic issue; assert `success` status and a non-empty diff + PR summary written to `outputs/`
    - _Design: Testing Strategy/Integration_
  - [ ]* 10.3 Write property test for offline equivalence
    - **Property 10: Offline equivalence** — for cached issues, an offline run yields the same `status` class as an online run with identical inputs
    - **Validates: reviewability**
    - _Design: Correctness Properties/Property 10_

- [x] 11. Checkpoint
  - Ensure all tests pass, ask the user if questions arise.

- [x] 12. Evaluation harness
  - [x] 12.1 Create ground-truth eval cases under `eval/cases/`
    - Author `gin-4460.json` plus 2 additional approved-repo cases (repo, issue_number, base_commit, gold_changed_files, gold_diff)
    - _Design: Components/eval/harness.py, Example Usage/Example 4_
  - [x] 12.2 Implement `eval/harness.py`
    - Define `EvalCase`, `EvalReport`; implement `run_eval(case, cfg)`: drive `run_pipeline`, compute file-level localization precision/recall (our changed files vs gold), report build/vet/test pass, render side-by-side our-diff vs gold-diff
    - _Design: Components/eval/harness.py; Requirements: Criteria 1 & 4_
  - [ ]* 12.3 Write unit tests for the harness
    - Precision/recall computation on known file sets, side-by-side diff rendering
    - _Design: Testing Strategy/Unit_
  - [ ]* 12.4 Write integration smoke test on `gin-4460`
    - Assert precision/recall computed and side-by-side diff renders for the gin #4460 case
    - _Design: Testing Strategy/Integration_

- [x] 13. Documentation
  - [x] 13.1 Write `README.md`
    - Setup/prerequisites (Python, Go toolchain, ripgrep, `ANTHROPIC_API_KEY`), one-command run (`python -m src.main --issue 4460`), Agentless design rationale (no agent loop / fixed stages), explicit mapping to the five grading criteria, and a sample run walkthrough
    - _Design: Overview, Architecture; Requirements: Criteria 1–5_

- [x] 14. Final checkpoint
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional test sub-tasks and can be skipped for a faster MVP; core implementation tasks are never optional.
- Each task references specific design sections; the implementation language is Python 3.11+ (the design's pseudocode is Python).
- `requirements.md` is currently empty, so requirement references use the five reviewer criteria and the reproducibility/reliability goals documented in the design.
- Property tests use Hypothesis and are placed next to the implementation they guard so correctness regressions surface early.
- Checkpoints (tasks 5, 8, 11, 14) provide incremental validation breaks.
- Each property test sub-task cites its numbered Correctness Property from `design.md` and the criterion it validates.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "12.1"] },
    { "id": 1, "tasks": ["1.2", "1.4", "2.1", "3.1", "6.1"] },
    { "id": 2, "tasks": ["1.3", "1.5", "2.2", "3.2", "3.3", "4.1", "6.2", "6.3", "9.1"] },
    { "id": 3, "tasks": ["4.2", "4.3", "6.4", "9.2", "9.3"] },
    { "id": 4, "tasks": ["4.4", "4.5", "6.5", "6.6", "7.1"] },
    { "id": 5, "tasks": ["7.2", "7.3", "7.4"] },
    { "id": 6, "tasks": ["7.5", "7.6", "7.7", "7.8", "10.1"] },
    { "id": 7, "tasks": ["10.2", "10.3", "12.2"] },
    { "id": 8, "tasks": ["12.3", "12.4"] },
    { "id": 9, "tasks": ["13.1"] }
  ]
}
```
