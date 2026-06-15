"""CLI orchestration — wire the four-phase pipeline end-to-end.

This module is the binding glue between the typed component modules
(:mod:`src.issue`, :mod:`src.repo`, :mod:`src.localize`, :mod:`src.repair`,
:mod:`src.validate`, :mod:`src.select`, :mod:`src.summarize`) and the
command-line surface documented in design.md (Example Usage):

    python -m src.main --issue 4460

Responsibilities
----------------

* Parse CLI arguments and merge them as overrides on top of ``config.yaml``.
* Drive :func:`run_pipeline`: fetch issue → prepare repo → localize → generate
  candidates → repro test + validate → select → fetch recent PRs → summarize
  → :func:`write_outputs`.
* Emit per-phase artifacts to ``cfg.outputs_dir`` so a reviewer can inspect
  *every* intermediate stage even when the run terminates early. Concretely:

  ``outputs/issue.json``           the validated issue payload
  ``outputs/localization.json``    ranked files + edit locations
  ``outputs/candidates.json``      one entry per Phase-2 sample
  ``outputs/validation.json``      per-candidate validation reports
  ``outputs/diff.patch``           the winning unified diff (success only)
  ``outputs/pr_title.txt``         single-line PR title (success only)
  ``outputs/pr_body.md``           PR body referencing the issue (success only)
  ``outputs/run.json``             terminal :class:`RunResult` snapshot
  ``outputs/run.log``              tee'd Python logging output

* Return a :class:`RunResult` with one of the three terminal statuses
  documented in design.md (``"success"``, ``"no_applicable_patch"``,
  ``"no_valid_patch"``). The CLI ``main`` translates that into a process
  exit code.

Design alignment
----------------

* Architecture diagram and Sequence Diagrams / Main end-to-end run.
* Algorithmic Pseudocode / Main pipeline orchestration (the inline pseudocode
  in this docstring is the ``run_pipeline`` here verbatim, modulo logging).
* Example Usage examples 1–3 (CLI invocation + config overrides).
* Requirements: the five reviewer criteria (1) right files, (2) relevant
  code changes, (3) project conventions, (4) appropriate validation,
  (5) reasonable PR summary — each phase the orchestrator drives is
  responsible for one criterion; this module's job is to drive them all
  in the fixed agentless order.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

# Load .env BEFORE importing any pipeline module that reads ANTHROPIC_API_KEY.
# python-dotenv is a hard dependency declared in requirements.txt; if it is
# missing we fall back to a tiny inline parser so the user gets an actionable
# error rather than an opaque ImportError at startup.
def _load_dotenv_if_present() -> None:
    """Load ``./.env`` into ``os.environ`` without overriding existing vars.

    Existing env vars win over file values, matching the python-dotenv default.
    Quietly returns if no .env file is present — running with the env already
    populated (e.g. in CI) is the supported pattern.
    """
    env_path = Path(".env")
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv  # type: ignore[import-not-found]
        load_dotenv(env_path, override=False)
        return
    except ImportError:
        pass
    # Minimal fallback: KEY=VALUE per line, '#' comments, no shell expansion.
    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError:
        pass


_load_dotenv_if_present()

from src.config import Config, ConfigError, load_config
from src.issue import Issue, IssueError, fetch_issue
from src.llm import LLMError
from src.localize import LocalizationResult, localize
from src.repair import Candidate, RepairError, generate_candidates
from src.repo import RepoCheckout, RepoError, prepare_repo
from src.select import RankedCandidate, SelectionError, select
from src.summarize import PRSummary, SummarizeError, fetch_recent_merged_prs, summarize
from src.validate import ValidateError, ValidationReport, generate_repro_test, validate


__all__ = [
    "RunResult",
    "run_pipeline",
    "write_outputs",
    "main",
]


logger = logging.getLogger(__name__)


# Process exit codes mirror the terminal RunResult statuses so a shell
# caller (e.g. the eval harness) can branch without parsing stdout.
_EXIT_OK = 0
_EXIT_USAGE = 1
_EXIT_NO_APPLICABLE = 2
_EXIT_NO_VALID = 3


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunResult:
    """Terminal outcome of a single :func:`run_pipeline` invocation.

    Attributes:
        status: One of ``"success"``, ``"no_applicable_patch"``,
            ``"no_valid_patch"`` — the three terminal states documented in
            the design's Algorithmic Pseudocode / Main pipeline.
        winner: The chosen :class:`~src.select.RankedCandidate` on
            ``"success"``; ``None`` otherwise.
        summary: The :class:`~src.summarize.PRSummary` on ``"success"``;
            ``None`` otherwise.
        localization: The Phase-1 :class:`~src.localize.LocalizationResult`
            when Phase 1 ran. Always populated unless an upstream step
            (issue fetch, repo prep) failed before Phase 1 could start.
        reports: Per-candidate validation reports. Empty tuple when
            validation did not run (e.g. zero applying candidates).
    """

    status: str
    winner: Optional[RankedCandidate] = None
    summary: Optional[PRSummary] = None
    localization: Optional[LocalizationResult] = None
    reports: tuple[ValidationReport, ...] = ()


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


def run_pipeline(cfg: Config) -> RunResult:
    """Drive the four-phase pipeline end-to-end.

    Mirrors the design.md Algorithmic Pseudocode / Main pipeline block:

    1. ``fetch_issue`` (online or cached) — validate input.
    2. ``prepare_repo`` — clone (or reuse) and pin to ``base_commit``.
    3. Phase 1 — :func:`localize`. Per-phase artifact written before any
       short-circuit so reviewers always see what we localized.
    4. Phase 2 — :func:`generate_candidates`. Filter to ``applied_clean``.
       If none apply, return ``no_applicable_patch``.
    5. Phase 3 — :func:`generate_repro_test` (best-effort) →
       :func:`validate` → :func:`select`. If no winner, return
       ``no_valid_patch``.
    6. Phase 4 — fetch recent merged PRs → :func:`summarize` → emit final
       artifacts → return ``success``.

    Per-phase artifacts are written incrementally so an aborted run still
    leaves traceable evidence on disk:

      * ``issue.json`` (after step 1)
      * ``localization.json`` (after step 3)
      * ``candidates.json`` (after step 4)
      * ``validation.json`` (after step 5, even on no_valid_patch)
      * ``diff.patch`` / ``pr_title.txt`` / ``pr_body.md`` (after step 6)
      * ``run.json`` (always, last)

    Args:
        cfg: Validated :class:`~src.config.Config`.

    Returns:
        :class:`RunResult` with one of the three terminal statuses.
    """
    outputs_dir = Path(cfg.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "pipeline: starting run for %s issue=%d (offline=%s)",
        cfg.repo,
        cfg.issue_number,
        cfg.offline,
    )

    # Phase 0 — issue ingestion ---------------------------------------------
    issue = fetch_issue(
        cfg.repo,
        cfg.issue_number,
        offline=cfg.offline,
        cache_dir=Path(cfg.workdir),
    )
    _emit_artifact(outputs_dir / "issue.json", issue)
    logger.info("pipeline: fetched issue '%s'", issue.title)

    checkout = prepare_repo(cfg.repo, issue.base_commit, Path(cfg.workdir))
    logger.info(
        "pipeline: repo prepared at %s (base_commit=%s)",
        checkout.path,
        checkout.base_commit,
    )

    # Phase 1 — localize ----------------------------------------------------
    loc = localize(checkout, issue, cfg)
    _emit_artifact(outputs_dir / "localization.json", loc)
    logger.info(
        "pipeline: localization picked %d files, %d edit locations (fallback=%s)",
        len(loc.ranked_files),
        len(loc.locations),
        loc.used_fallback,
    )

    # Phase 2 — repair ------------------------------------------------------
    candidates = generate_candidates(checkout, issue, loc.locations, cfg)
    _emit_artifact(outputs_dir / "candidates.json", _candidates_summary(candidates))
    applying = [c for c in candidates if c.applied_clean]
    logger.info(
        "pipeline: generated %d candidates, %d applied cleanly",
        len(candidates),
        len(applying),
    )
    if not applying:
        logger.warning(
            "pipeline: no candidate applied cleanly; emitting no_applicable_patch"
        )
        result = RunResult(
            status="no_applicable_patch",
            localization=loc,
        )
        _emit_artifact(outputs_dir / "run.json", _run_result_summary(result))
        return result

    # Phase 3 — validate + select ------------------------------------------
    repro = generate_repro_test(issue, checkout, cfg)
    if repro is None:
        logger.info("pipeline: no reproduction test generated (continuing)")
    else:
        logger.info("pipeline: reproduction test generated (%d chars)", len(repro))

    reports = validate(applying, repro, cfg)
    _emit_artifact(outputs_dir / "validation.json", reports)
    logger.info("pipeline: produced %d validation reports", len(reports))

    winner = select(applying, reports, cfg)
    if winner is None:
        logger.warning(
            "pipeline: every candidate broke at least one existing test; "
            "emitting no_valid_patch"
        )
        result = RunResult(
            status="no_valid_patch",
            localization=loc,
            reports=tuple(reports),
        )
        _emit_artifact(outputs_dir / "run.json", _run_result_summary(result))
        return result
    logger.info(
        "pipeline: winner is %s (score=%s)", winner.candidate.id, winner.score
    )

    # Phase 4 — summarize ---------------------------------------------------
    recent = fetch_recent_merged_prs(
        cfg.repo,
        k=3,
        offline=cfg.offline,
        cache_dir=Path(cfg.workdir) / "prs",
    )
    summary = summarize(winner.candidate.diff, issue, recent, cfg)
    logger.info("pipeline: PR summary drafted (title=%r)", summary.title)

    write_outputs(outputs_dir, winner, summary, reports, loc)

    result = RunResult(
        status="success",
        winner=winner,
        summary=summary,
        localization=loc,
        reports=tuple(reports),
    )
    _emit_artifact(outputs_dir / "run.json", _run_result_summary(result))
    logger.info("pipeline: run finished with status=success")
    return result


# ---------------------------------------------------------------------------
# Output emission
# ---------------------------------------------------------------------------


def write_outputs(
    outputs_dir: Path,
    winner: RankedCandidate,
    summary: PRSummary,
    reports: Sequence[ValidationReport],
    loc: LocalizationResult,
) -> None:
    """Emit the success-path artifacts under ``outputs_dir``.

    Writes (or overwrites):

    * ``diff.patch`` — the normalized winning unified diff (Phase 2 output).
    * ``pr_title.txt`` — single-line title plus trailing newline.
    * ``pr_body.md`` — multi-line markdown body referencing the issue.
    * ``validation.json`` — per-candidate :class:`ValidationReport` list.
    * ``localization.json`` — Phase-1 :class:`LocalizationResult`.
    * ``winner.json`` — small summary of the chosen candidate (id, score,
      worktree path) so reviewers can locate the source artifact.

    The function is idempotent — calling it twice with the same inputs
    yields the same files. It is safe to call after :func:`run_pipeline`
    has already emitted the per-phase ``localization.json`` and
    ``validation.json`` files; the rewrite is byte-equivalent in that case.

    Args:
        outputs_dir: Destination directory; created if missing.
        winner: The chosen :class:`RankedCandidate` from Phase 3b.
        summary: PR title/body produced by Phase 4.
        reports: Validation reports for every applying candidate.
        loc: Phase-1 localization result.
    """
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    diff_path = outputs_dir / "diff.patch"
    title_path = outputs_dir / "pr_title.txt"
    body_path = outputs_dir / "pr_body.md"
    winner_path = outputs_dir / "winner.json"

    diff_text = winner.candidate.diff or ""
    if diff_text and not diff_text.endswith("\n"):
        diff_text += "\n"
    diff_path.write_text(diff_text, encoding="utf-8")

    title_path.write_text(summary.title.rstrip() + "\n", encoding="utf-8")

    body_text = summary.body.rstrip()
    if body_text:
        body_text += "\n"
    body_path.write_text(body_text, encoding="utf-8")

    _emit_artifact(outputs_dir / "validation.json", list(reports))
    _emit_artifact(outputs_dir / "localization.json", loc)
    _emit_artifact(
        winner_path,
        {
            "candidate_id": winner.candidate.id,
            "score": list(winner.score),
            "worktree": str(winner.candidate.worktree),
            "applied_clean": winner.candidate.applied_clean,
            "diff_chars": len(winner.candidate.diff or ""),
        },
    )

    logger.info("pipeline: success artifacts written to %s", outputs_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for ``python -m src.main``.

    Mirrors the override semantics in :mod:`src.config`: top-level keys
    flow through directly, and ``--model`` / ``--sample-count`` /
    ``--temperature`` map onto the ``llm.<field>`` dotted overrides.
    """
    p = argparse.ArgumentParser(
        prog="python -m src.main",
        description=(
            "Agentless Go Contributor — turn a GitHub issue into a "
            "validated patch and PR summary."
        ),
    )
    p.add_argument(
        "--issue",
        type=int,
        metavar="N",
        help="GitHub issue number (overrides config.yaml issue_number).",
    )
    p.add_argument(
        "--repo",
        metavar="OWNER/NAME",
        help="Target repo slug (must be in repo_allowlist).",
    )
    p.add_argument(
        "--config",
        default="config.yaml",
        metavar="PATH",
        help="Path to config.yaml (default: ./config.yaml).",
    )
    p.add_argument(
        "--base-commit",
        metavar="SHA_OR_REF",
        help="Pin the base commit; resolved via git rev-parse if a ref.",
    )
    p.add_argument(
        "--top-n-files",
        type=int,
        metavar="N",
        help="Phase 1 file-pick breadth (overrides top_n_files).",
    )
    p.add_argument(
        "--workdir",
        metavar="PATH",
        help="Working directory for clones and caches.",
    )
    p.add_argument(
        "--outputs-dir",
        metavar="PATH",
        help="Directory for per-phase artifacts and run logs.",
    )
    offline = p.add_mutually_exclusive_group()
    offline.add_argument(
        "--offline",
        dest="offline",
        action="store_true",
        default=None,
        help="Run fully offline; require cached issue/PR JSON.",
    )
    offline.add_argument(
        "--no-offline",
        dest="offline",
        action="store_false",
        help="Force online mode (overrides offline=true in config).",
    )

    # LLM overrides → dotted keys consumed by load_config.
    p.add_argument(
        "--model",
        metavar="NAME",
        help="LLM model identifier (overrides llm.model).",
    )
    p.add_argument(
        "--sample-count",
        type=int,
        metavar="N",
        help="Number of repair candidates to sample (overrides llm.sample_count).",
    )
    p.add_argument(
        "--temperature",
        type=float,
        metavar="T",
        help="LLM sampling temperature (overrides llm.temperature).",
    )

    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging on stderr.",
    )
    return p


def _overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Project parsed CLI ``args`` into the override dict shape ``load_config`` accepts.

    Only keys the user actually set are included so they don't clobber the
    YAML defaults with ``None``.
    """
    overrides: dict[str, Any] = {}
    if args.issue is not None:
        overrides["issue_number"] = args.issue
    if args.repo is not None:
        overrides["repo"] = args.repo
    if args.base_commit is not None:
        overrides["base_commit"] = args.base_commit
    if args.top_n_files is not None:
        overrides["top_n_files"] = args.top_n_files
    if args.workdir is not None:
        overrides["workdir"] = args.workdir
    if args.outputs_dir is not None:
        overrides["outputs_dir"] = args.outputs_dir
    if args.offline is not None:
        overrides["offline"] = args.offline
    if args.model is not None:
        overrides["llm.model"] = args.model
    if args.sample_count is not None:
        overrides["llm.sample_count"] = args.sample_count
    if args.temperature is not None:
        overrides["llm.temperature"] = args.temperature
    return overrides


def _configure_logging(outputs_dir: Path, verbose: bool) -> None:
    """Wire Python logging to stderr plus a rotating-free run log file.

    The file handler is attached only after ``outputs_dir`` exists so a
    bad config path doesn't accidentally create a stray ``run.log``.
    """
    outputs_dir.mkdir(parents=True, exist_ok=True)

    level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger()

    # Tear down any handlers a previous in-process run installed so
    # repeated invocations from a long-running test process don't double
    # log. Stdlib idiom: list(root.handlers) to avoid mutation-during-iter.
    for h in list(root.handlers):
        root.removeHandler(h)

    root.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    stream = logging.StreamHandler(sys.stderr)
    stream.setLevel(level)
    stream.setFormatter(fmt)
    root.addHandler(stream)

    log_path = outputs_dir / "run.log"
    file_h = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_h.setLevel(level)
    file_h.setFormatter(fmt)
    root.addHandler(file_h)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    overrides = _overrides_from_args(args)

    try:
        cfg = load_config(Path(args.config), overrides)
    except ConfigError as exc:
        # Logging is not yet configured (we don't know outputs_dir without cfg);
        # write straight to stderr so the failure is visible.
        print(f"error: {exc}", file=sys.stderr)
        return _EXIT_USAGE

    _configure_logging(Path(cfg.outputs_dir), args.verbose)

    try:
        result = run_pipeline(cfg)
    except (
        ConfigError,
        IssueError,
        RepoError,
        RepairError,
        ValidateError,
        SelectionError,
        SummarizeError,
        LLMError,
    ) as exc:
        logger.error("pipeline failed: %s", exc)
        return _EXIT_USAGE
    except KeyboardInterrupt:
        logger.warning("interrupted by user")
        return _EXIT_USAGE

    if result.status == "success":
        return _EXIT_OK
    if result.status == "no_applicable_patch":
        return _EXIT_NO_APPLICABLE
    if result.status == "no_valid_patch":
        return _EXIT_NO_VALID
    # Defensive: an unknown status is a programmer error, not a user error.
    logger.error("pipeline returned unknown status %r", result.status)
    return _EXIT_USAGE


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _emit_artifact(path: Path, payload: Any) -> None:
    """Best-effort JSON dump of ``payload`` to ``path``.

    Uses :func:`_to_jsonable` so dataclasses, enums, ``Path`` instances,
    and tuples all serialize cleanly. Any I/O failure is logged at WARNING
    rather than raised — artifact emission is auxiliary and must not sink
    a successful run.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(_to_jsonable(payload), fh, indent=2, ensure_ascii=False)
            fh.write("\n")
    except OSError as exc:
        logger.warning("failed to write artifact %s: %s", path, exc)


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses / enums / paths into JSON-friendly types.

    Loop invariant: every value returned from this function is one of
    ``None``, ``bool``, ``int``, ``float``, ``str``, ``list``, or ``dict``
    whose contents are themselves JSON-friendly.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: _to_jsonable(getattr(obj, f.name))
            for f in dataclasses.fields(obj)
        }
    if isinstance(obj, Mapping):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_to_jsonable(v) for v in obj]
    # Last resort: fall back to repr so the artifact is still readable.
    return repr(obj)


def _candidates_summary(candidates: Sequence[Candidate]) -> list[dict[str, Any]]:
    """Lightweight summary of Phase-2 candidates for the artifact log.

    We deliberately avoid serializing each ``SearchReplaceBlock``'s search
    and replace text — those can be large and are already on disk inside
    the worktree. Instead we record the structural shape callers actually
    need to triage a run: id, applied flag, block count, target files,
    diff size.
    """
    out: list[dict[str, Any]] = []
    for c in candidates:
        files = sorted({b.file_path for b in c.blocks})
        out.append(
            {
                "id": c.id,
                "applied_clean": c.applied_clean,
                "block_count": len(c.blocks),
                "files": files,
                "worktree": str(c.worktree),
                "diff_chars": len(c.diff or ""),
            }
        )
    return out


def _run_result_summary(result: RunResult) -> dict[str, Any]:
    """Compact JSON-friendly snapshot of a :class:`RunResult`."""
    payload: dict[str, Any] = {"status": result.status}
    if result.localization is not None:
        payload["localization"] = {
            "ranked_files": list(result.localization.ranked_files),
            "location_count": len(result.localization.locations),
            "used_fallback": result.localization.used_fallback,
        }
    if result.reports:
        payload["reports"] = _to_jsonable(list(result.reports))
    if result.winner is not None:
        payload["winner"] = {
            "candidate_id": result.winner.candidate.id,
            "score": list(result.winner.score),
        }
    if result.summary is not None:
        payload["summary"] = {
            "title": result.summary.title,
            "body_chars": len(result.summary.body),
        }
    return payload


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
