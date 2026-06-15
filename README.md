# Agentless Go Contributor

A deterministic, four-phase pipeline that turns a GitHub issue from an approved
Go repository into a production-quality fix: a unified diff plus a PR title and
body. The pipeline localizes the relevant files, generates several candidate
patches, validates and ranks them with the native Go toolchain, picks a single
winner, and writes a PR summary that matches recent merged-PR conventions in the
target repo.

This system is intentionally **agentless**. There is no autonomous tool-calling
loop, no planner that decides its own actions, and no multi-agent setup. A
Python orchestrator owns all control flow; the LLM is used only for tightly
scoped sub-tasks (file selection, edit-location narrowing, patch generation,
PR prose). The Go toolchain (`go build`, `go vet`, `gofmt`, `goimports`,
`go test`, optionally `golangci-lint`) is the source of truth for validation.
Nothing is fed back into an "agent" to re-plan. That intentional simplicity is
what makes runs reproducible and reviewable.

## Approved repositories

The example issue ships against [`gin-gonic/gin#4460`](https://github.com/gin-gonic/gin/issues/4460).
The repo allowlist in `config.yaml` also accepts `spf13/cobra` and
`labstack/echo`.

## Setup

### Prerequisites

- **Python 3.11+** — required by the dataclass and typing features used in `src/`.
- **Go toolchain** on `PATH` — `go build`, `go vet`, `go test`, `gofmt`, `goimports`.
  - Install Go from [go.dev/dl](https://go.dev/dl/).
  - Install `goimports` once with `go install golang.org/x/tools/cmd/goimports@latest`
    and ensure `$(go env GOPATH)/bin` is on `PATH`.
- **git** on `PATH` — used to clone the target repo and check out the pinned base commit.
- **`ANTHROPIC_API_KEY`** environment variable — read only from the environment,
  never from the config file. The `claude-sonnet-4-5` model is used by default
  (see `config.yaml`).
- **`golangci-lint`** *(optional)* — the LINT check is reported as `skipped`
  when the binary is absent; it does not block selection.
- **`ripgrep`** *(optional but recommended)* — used by Phase 1's fallback when
  the repo skeleton exceeds the token budget; install via your package manager
  (`brew install ripgrep`, `choco install ripgrep`, `apt install ripgrep`).

### Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

## One-command run

```bash
python -m src.main --issue 4460
```

That single command runs the full pipeline against the configured target repo
(default: `gin-gonic/gin`), pinned base commit, and `outputs_dir`. The CLI also
accepts `--repo`, `--config`, `--workdir`, `--outputs-dir`, `--offline`, and
`-v/--verbose`; everything else lives in `config.yaml`.

For an entirely offline run (uses the cached issue JSON under `eval/cases/`):

```bash
python -m src.main --issue 4460 --offline
```

## Agentless design rationale

The pipeline is strictly linear; each phase consumes the output of the previous
phase and writes a traceable artifact under `outputs/`. No phase calls back into
a prior phase. Failures short-circuit with a clear status or a documented
fallback.

```
CLI (--issue N)
  ↓
config.py → issue.py → repo.py
  ↓
Phase 1: localize.py            (skeleton + LLM file pick + LLM narrow,
                                 ripgrep fallback when over budget)
  ↓
Phase 2: repair.py + patch.py   (n SEARCH/REPLACE candidates, atomic apply,
                                 gofmt/goimports prefilter)
  ↓
Phase 3: validate.py            (BUILD → VET → FMT → TEST → LINT, layered)
         select.py              (rank + tie-break, drop regressions)
  ↓
Phase 4: summarize.py           (PR title + body, matching recent-PR style)
  ↓
outputs/                        (diff, PR summary, per-phase artifacts, logs)
```

Why this shape:

- **Bounded LLM scope.** Each LLM call has one job and a well-defined input/output
  contract. No prompt has to "decide what tool to call next."
- **Reproducibility.** The base checkout is never mutated; every candidate edit
  lands in an isolated worktree. Pinning to a base commit means two runs against
  the same issue see byte-identical source.
- **Toolchain as ground truth.** Validation is `go build`, `go vet`, `gofmt`,
  `go test`, `golangci-lint` — exit codes drive selection. The model never grades
  its own output.
- **Reviewability.** Every phase persists a JSON artifact, so a reviewer can
  inspect intermediate stages even when a run terminates early
  (`no_applicable_patch`, `no_valid_patch`).

## Mapping to the five reviewer criteria

| # | Criterion                           | Pipeline phase that delivers it                                                                                                                                                  |
|---|-------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1 | Right files identified              | **Phase 1 — `src/localize.py`** builds a repo skeleton (tree + per-file outlines, exported decls only), the LLM picks top-N suspicious files, and ripgrep is the over-budget fallback. |
| 2 | Relevant code changes               | **Phase 2 — `src/repair.py` + `src/patch.py`** generate `n` SEARCH/REPLACE candidates, then atomically apply each to a fresh worktree (all-or-nothing, with rollback on any miss).      |
| 3 | Project conventions followed        | **Phase 2 prefilter** runs `gofmt` and `goimports` on every applying candidate; **Phase 3 FMT / LINT checks** confirm formatting and (when available) `golangci-lint`.                  |
| 4 | Appropriate validation run          | **Phase 3 — `src/validate.py` + `src/select.py`** run BUILD → VET → FMT → TEST → LINT in fixed order, short-circuit on BUILD failure, drop regressions, and rank lexicographically.     |
| 5 | Reasonable PR summary               | **Phase 4 — `src/summarize.py`** fetches 2–3 recent merged PRs to derive style, then drafts a single-line title and a body that references the issue number.                            |

The corresponding correctness properties (Property 2 for criterion 1, Property
3/4 for criterion 2/3, Property 5/6/7 for criterion 4, Property 9 for
criterion 5) are documented in `.kiro/specs/agentless-go-contributor/design.md`.

## Sample run walkthrough

After `python -m src.main --issue 4460` finishes successfully, `outputs/`
contains:

```
outputs/
├── issue.json           # validated issue payload (title, body, labels, base commit)
├── localization.json    # ranked files + edit locations (file, symbol, line range, context)
├── candidates.json      # one entry per Phase-2 sample (blocks, applied_clean, diff)
├── validation.json      # per-candidate ValidationReport (BUILD/VET/FMT/TEST/LINT, breaks_existing_tests)
├── diff.patch           # the winning normalized unified diff
├── pr_title.txt         # single-line PR title
├── pr_body.md           # PR body markdown referencing the issue number
├── winner.json          # chosen candidate id + score + worktree path
├── run.json             # terminal RunResult snapshot (status + key data)
└── run.log              # tee'd Python logging output for the whole run
```

Reading the artifacts in order tells the full story:

1. **`issue.json`** — what the system actually saw (after allowlist + validation).
2. **`localization.json`** — which `.go` files were picked and why; whether the
   ripgrep fallback was used (`used_fallback`).
3. **`candidates.json`** — every SEARCH/REPLACE sample, with `applied_clean=true`
   on the ones that landed in a worktree without conflict.
4. **`validation.json`** — for each applying candidate, the layered Go-toolchain
   results, in fixed order.
5. **`diff.patch`** + **`pr_title.txt`** + **`pr_body.md`** — the deliverable:
   a unified diff plus PR prose that matches recent-PR style.
6. **`run.json`** — terminal status (`success`, `no_applicable_patch`, or
   `no_valid_patch`) for scripting and CI integration.
7. **`run.log`** — full per-phase trace for post-mortem review.

If a run terminates early (e.g. no candidate applied cleanly), the partial
artifacts (`issue.json`, `localization.json`, `candidates.json`) are still
written, so the reviewer can see exactly where the pipeline stopped and why.

## Project layout

```
.
├── src/                  # Pipeline implementation
│   ├── config.py         #   YAML + env config, allowlist enforcement
│   ├── issue.py          #   GitHub REST + offline JSON ingestion
│   ├── repo.py           #   clone, pinned checkout, isolated worktrees, diff
│   ├── llm.py            #   Anthropic Claude wrapper (retry, JSON repair, n samples)
│   ├── localize.py       #   Phase 1: skeleton + LLM pick + LLM narrow + ripgrep fallback
│   ├── patch.py          #   SEARCH/REPLACE parser + atomic apply
│   ├── repair.py         #   Phase 2: n candidates + gofmt/goimports prefilter
│   ├── validate.py       #   Phase 3a: layered Go-toolchain checks + repro test
│   ├── select.py         #   Phase 3b: ranking + tie-break + regression drop
│   ├── summarize.py      #   Phase 4: PR title + body matching repo style
│   └── main.py           #   CLI entrypoint and run_pipeline orchestration
├── eval/
│   ├── harness.py        # File-level precision/recall + side-by-side diff vs gold
│   └── cases/            # Ground-truth eval cases (gin-4460, cobra-1734, echo-1921)
├── prompts/              # Per-phase prompt templates
├── outputs/              # Per-run artifacts and logs (gitignored)
├── config.yaml           # Default config; CLI flags override these values
└── requirements.txt      # Pinned Python dependencies
```

## Evaluation

Run the harness against a checked-in ground-truth case to get file-level
localization precision/recall plus a side-by-side our-diff-vs-gold-diff view:

```bash
python -m eval.harness --case eval/cases/gin-4460.json
```

The other shipped cases are `eval/cases/cobra-1734.json` and
`eval/cases/echo-1921.json`.

## Configuration reference

`config.yaml` controls non-secret defaults. The most useful knobs:

- `repo` and `repo_allowlist` — target and approved repos.
- `issue_number` — default issue (CLI `--issue` overrides).
- `top_n_files` — Phase 1 localization breadth.
- `offline` — use cached issue/PR JSON instead of GitHub.
- `llm.model`, `llm.temperature`, `llm.sample_count`, `llm.max_retries`,
  `llm.request_timeout_s` — Anthropic call settings; `sample_count` is the
  primary cost/quality trade-off (number of repair candidates per run).

`ANTHROPIC_API_KEY` is read only from the environment — never put it in
`config.yaml` or in any file under `outputs/`.
