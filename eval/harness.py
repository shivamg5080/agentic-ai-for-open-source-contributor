"""Evaluation harness — measure pipeline quality against a known merged PR.

This module implements ``Components / eval/harness.py`` from ``design.md``:
it ingests a ground-truth :class:`EvalCase` (a JSON file under
``eval/cases/``), drives :func:`src.main.run_pipeline` against the case's
pinned base commit, and produces an :class:`EvalReport` with:

* file-level localization **precision** and **recall** (our changed files
  vs. the merged PR's gold changed files),
* the **build / vet / tests** pass-flags pulled from the winning
  candidate's :class:`~src.validate.ValidationReport` (so reviewers can
  see whether the chosen patch actually compiles, vets clean, and keeps
  the existing test suite green), and
* a **side-by-side diff** rendering of our patch next to the upstream
  fix, intended for at-a-glance reviewer comparison.

The harness is deliberately thin: it is not a runner that decides which
case to use, parallelises across cases, or persists its own artifacts —
those are the orchestrator's job. Its single responsibility is to turn
``(case, cfg)`` into a deterministic :class:`EvalReport`.

Design alignment
----------------

* Components / eval/harness.py: :class:`EvalCase`, :class:`EvalReport`,
  :func:`run_eval` signatures match the design verbatim.
* Example Usage / Example 4: the public surface
  (:func:`load_eval_case`, :func:`run_eval`) mirrors the snippet
  documented there.
* Requirements: Criterion 1 (right files identified) is measured by
  precision/recall; Criterion 4 (appropriate validation) is measured by
  the build/vet/tests pass-flags.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional, Sequence

from src.main import RunResult, run_pipeline
from src.validate import CheckName, ValidationReport

if TYPE_CHECKING:  # pragma: no cover - type-only imports
    from src.config import Config


__all__ = [
    "EvalCase",
    "EvalReport",
    "EvalError",
    "load_eval_case",
    "run_eval",
]


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


# Width of each column in the side-by-side diff rendering. Picked to fit
# two columns plus the separator inside a 170-column terminal — wide
# enough to hold typical Go source lines without aggressive truncation
# but narrow enough that the rendering stays readable when piped to a log
# file or pasted into a PR review comment.
_SIDE_BY_SIDE_COLUMN_WIDTH = 80

# Separator placed between the two columns in the rendered diff.
_SIDE_BY_SIDE_SEP = " | "


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class EvalError(ValueError):
    """Raised on malformed eval cases or unrecoverable harness failures.

    Recoverable conditions (the pipeline returning ``no_applicable_patch``
    or ``no_valid_patch``) do **not** raise: they produce an
    :class:`EvalReport` with zero precision/recall and ``False`` pass
    flags so callers can still tabulate results across cases.
    """


@dataclass(frozen=True)
class EvalCase:
    """One ground-truth evaluation case loaded from ``eval/cases/*.json``.

    Mirrors the design's data shape exactly. Each field is the minimum
    needed to drive the pipeline against the same inputs the upstream
    contributor saw, plus the merged-PR gold so we can measure how close
    our output came:

    Attributes:
        repo: ``owner/name`` slug; must be in the running pipeline's
            ``repo_allowlist`` for :func:`run_eval` to succeed.
        issue_number: GitHub issue number the merged PR addresses.
        base_commit: Parent commit of the merged PR's merge commit. The
            pipeline checks this out and applies its candidate edits on
            top so the diff is comparable to the gold.
        gold_changed_files: File paths the merged PR actually modified
            (relative to the repo root, no ``a/`` / ``b/`` prefix).
        gold_diff: The merged PR's unified diff verbatim, used as the
            right-hand column of the side-by-side rendering.
    """

    repo: str
    issue_number: int
    base_commit: str
    gold_changed_files: list[str]
    gold_diff: str


@dataclass(frozen=True)
class EvalReport:
    """Result of one :func:`run_eval` invocation.

    Attributes:
        file_precision: ``|ours ∩ gold| / |ours|``. Defined as ``0.0``
            when our patch changes no files (no positive predictions to
            be precise about).
        file_recall: ``|ours ∩ gold| / |gold|``. Defined as ``0.0`` when
            ``gold`` is empty (which a well-formed case should never
            allow — :func:`load_eval_case` rejects it).
        build_passed: ``go build ./...`` passed for the winning
            candidate. ``False`` when there is no winner.
        vet_passed: ``go vet ./...`` passed for the winning candidate.
        tests_passed: ``go test ./...`` passed for the winning candidate.
        diff_comparison: Side-by-side rendering of our diff next to
            ``case.gold_diff``. Always populated, even when our diff is
            empty (the left column shows blanks).
    """

    file_precision: float
    file_recall: float
    build_passed: bool
    vet_passed: bool
    tests_passed: bool
    diff_comparison: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_eval_case(path: Path) -> EvalCase:
    """Load and validate an eval case JSON file.

    Required JSON shape::

        {
          "repo": "owner/name",
          "issue_number": <positive int>,
          "base_commit": "<sha or ref>",
          "gold_changed_files": ["path/one.go", ...],
          "gold_diff": "diff --git a/...\\n..."
        }

    Extra keys (e.g. ``notes``) are tolerated and ignored — case files
    routinely carry provenance comments.

    Args:
        path: Path to the JSON file (e.g. ``eval/cases/gin-4460.json``).

    Returns:
        A frozen :class:`EvalCase`.

    Raises:
        EvalError: When the file is missing, not valid JSON, not a JSON
            object, missing a required field, or has a field of the
            wrong type / out of range.
    """
    path = Path(path)
    if not path.exists():
        raise EvalError(f"eval case file not found: {path}")

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EvalError(f"could not read eval case at {path}: {exc}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise EvalError(f"eval case at {path} is not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise EvalError(
            f"eval case at {path} must be a JSON object, "
            f"got {type(data).__name__}."
        )

    required = ("repo", "issue_number", "base_commit", "gold_changed_files", "gold_diff")
    missing = [k for k in required if k not in data]
    if missing:
        raise EvalError(
            f"eval case at {path} is missing required field(s): "
            f"{', '.join(missing)}."
        )

    repo = data["repo"]
    if not isinstance(repo, str) or "/" not in repo:
        raise EvalError(
            f"eval case at {path}: 'repo' must be of the form 'owner/name', "
            f"got {repo!r}."
        )

    issue_number = data["issue_number"]
    # Reject bools explicitly: bool is a subclass of int in Python.
    if (
        isinstance(issue_number, bool)
        or not isinstance(issue_number, int)
        or issue_number <= 0
    ):
        raise EvalError(
            f"eval case at {path}: 'issue_number' must be a positive integer, "
            f"got {issue_number!r}."
        )

    base_commit = data["base_commit"]
    if not isinstance(base_commit, str) or not base_commit:
        raise EvalError(
            f"eval case at {path}: 'base_commit' must be a non-empty string."
        )

    gold_changed_files = data["gold_changed_files"]
    if (
        not isinstance(gold_changed_files, list)
        or not gold_changed_files
        or not all(isinstance(f, str) and f for f in gold_changed_files)
    ):
        raise EvalError(
            f"eval case at {path}: 'gold_changed_files' must be a non-empty "
            f"list of non-empty strings."
        )

    gold_diff = data["gold_diff"]
    if not isinstance(gold_diff, str):
        raise EvalError(
            f"eval case at {path}: 'gold_diff' must be a string."
        )

    return EvalCase(
        repo=repo,
        issue_number=issue_number,
        base_commit=base_commit,
        gold_changed_files=list(gold_changed_files),
        gold_diff=gold_diff,
    )


def run_eval(case: EvalCase, cfg: "Config") -> EvalReport:
    """Drive :func:`run_pipeline` for ``case`` and score the result.

    The pipeline is invoked against a config whose ``repo``,
    ``issue_number``, and ``base_commit`` have been replaced by the
    case's values — the rest of ``cfg`` (workdir, outputs_dir, LLM
    settings, offline flag) is preserved so callers retain control over
    where artifacts land and how the LLM is invoked.

    Scoring:

    * **File precision / recall** — extract our changed files from the
      winning candidate's diff via ``+++ b/<path>`` lines, intersect
      with ``case.gold_changed_files``, divide.
    * **build / vet / tests passed** — pulled from the winning
      candidate's :class:`ValidationReport`. When the pipeline returned
      no winner (``no_applicable_patch`` or ``no_valid_patch``) all
      three flags are ``False``.
    * **diff_comparison** — :func:`_render_side_by_side` over our diff
      and ``case.gold_diff``.

    Args:
        case: Ground-truth :class:`EvalCase`.
        cfg: Pipeline :class:`~src.config.Config` to run with.

    Returns:
        A frozen :class:`EvalReport`.
    """
    case_cfg = dataclasses.replace(
        cfg,
        repo=case.repo,
        issue_number=case.issue_number,
        base_commit=case.base_commit,
    )

    result = run_pipeline(case_cfg)

    our_diff = ""
    if result.winner is not None:
        our_diff = result.winner.candidate.diff or ""

    our_files = _extract_changed_files(our_diff)
    gold_files = set(case.gold_changed_files)

    file_precision = _precision(our_files, gold_files)
    file_recall = _recall(our_files, gold_files)

    build_passed, vet_passed, tests_passed = _winner_check_flags(result)

    diff_comparison = _render_side_by_side(our_diff, case.gold_diff)

    return EvalReport(
        file_precision=file_precision,
        file_recall=file_recall,
        build_passed=build_passed,
        vet_passed=vet_passed,
        tests_passed=tests_passed,
        diff_comparison=diff_comparison,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_changed_files(diff: str) -> set[str]:
    """Return the set of target paths from a unified diff.

    Parses ``+++ b/<path>`` lines as the design specifies. Handles two
    real-world wrinkles:

    * ``+++ /dev/null`` — present when the patch deletes a file. Skipped
      (the deletion is captured by the matching ``--- a/...`` line; for
      our purposes we only count files the patch *touches as a target*).
    * ``+++ b/path/to/file.go\\t<timestamp>`` — git emits a tab-prefixed
      timestamp in some configurations. We trim everything from the first
      tab onward.
    """
    files: set[str] = set()
    for line in diff.splitlines():
        if not line.startswith("+++ "):
            continue
        target = line[4:].strip()
        if not target or target == "/dev/null":
            continue
        # Trim trailing ``\t<timestamp>`` if git included one.
        if "\t" in target:
            target = target.split("\t", 1)[0].strip()
        if target.startswith("b/"):
            target = target[2:]
        if target:
            files.add(target)
    return files


def _precision(ours: set[str], gold: set[str]) -> float:
    """``|ours ∩ gold| / |ours|`` with the empty-prediction edge case.

    When ``ours`` is empty the conventional precision is undefined
    (no positive predictions to be precise about). We return ``0.0``
    rather than raising so callers can sort/aggregate without special
    casing — an empty patch trivially fails to match the gold.
    """
    if not ours:
        return 0.0
    return len(ours & gold) / len(ours)


def _recall(ours: set[str], gold: set[str]) -> float:
    """``|ours ∩ gold| / |gold|`` with the empty-gold edge case.

    A well-formed case has non-empty ``gold`` (enforced by
    :func:`load_eval_case`). The defensive ``0.0`` here is purely so
    direct callers passing constructed sets cannot trip a
    ``ZeroDivisionError``.
    """
    if not gold:
        return 0.0
    return len(ours & gold) / len(gold)


def _winner_check_flags(result: RunResult) -> tuple[bool, bool, bool]:
    """Return ``(build_passed, vet_passed, tests_passed)`` for the winner.

    When ``result.winner`` is ``None`` (no applicable / no valid patch),
    all three flags are ``False`` — there is no candidate whose checks
    we could honestly attribute to "the pipeline's chosen fix".
    """
    if result.winner is None:
        return False, False, False

    report = _find_report_for_candidate(
        result.reports, result.winner.candidate.id
    )
    if report is None:
        return False, False, False

    return (
        _check_passed(report, CheckName.BUILD),
        _check_passed(report, CheckName.VET),
        _check_passed(report, CheckName.TEST),
    )


def _find_report_for_candidate(
    reports: Iterable[ValidationReport],
    candidate_id: str,
) -> Optional[ValidationReport]:
    """First report matching ``candidate_id`` or ``None``."""
    for report in reports:
        if report.candidate_id == candidate_id:
            return report
    return None


def _check_passed(report: ValidationReport, name: CheckName) -> bool:
    """``True`` iff ``report`` has a non-skipped, passing check named ``name``.

    Mirrors the design's invariant that ``passed`` and ``skipped`` are
    never both true: a skipped check (e.g. ``golangci-lint`` absent) is
    treated here as "not passed" so a missing toolchain does not flatter
    the eval report.
    """
    for check in report.checks:
        if check.name is name:
            return bool(check.passed and not check.skipped)
    return False


def _render_side_by_side(
    left: str,
    right: str,
    *,
    width: int = _SIDE_BY_SIDE_COLUMN_WIDTH,
) -> str:
    """Render ``left`` and ``right`` diffs as two parallel columns.

    Both inputs are split on newlines and zipped via
    :func:`itertools.zip_longest` so columns stay aligned even when one
    diff is longer than the other. Each left-column line is truncated
    to ``width`` characters and right-padded with spaces so the
    separator lines up vertically; right-column lines are emitted
    untruncated (terminals wrap them naturally and text editors handle
    long lines fine).

    A header row labels the columns and a horizontal rule separates the
    header from the content. The output always ends with a trailing
    newline so it concatenates cleanly into larger reports.
    """
    left_lines = left.splitlines() if left else [""]
    right_lines = right.splitlines() if right else [""]

    header = f"{'OUR DIFF':<{width}}{_SIDE_BY_SIDE_SEP}GOLD DIFF"
    rule = f"{'-' * width}-+-{'-' * width}"

    rendered: list[str] = [header, rule]
    for left_line, right_line in zip_longest(left_lines, right_lines, fillvalue=""):
        # Truncate to width *before* padding so the column boundary is
        # respected even when a line happens to be exactly width chars.
        if len(left_line) > width:
            left_disp = left_line[: width - 1] + "…"
        else:
            left_disp = left_line.ljust(width)
        rendered.append(f"{left_disp}{_SIDE_BY_SIDE_SEP}{right_line}")

    return "\n".join(rendered) + "\n"


# Re-exported as a small convenience for direct callers (e.g. unit tests
# in task 12.3) that want to validate the helpers without going through
# the whole pipeline.
def _internal_helpers() -> Sequence[str]:  # pragma: no cover - introspection only
    return (
        "_extract_changed_files",
        "_precision",
        "_recall",
        "_winner_check_flags",
        "_render_side_by_side",
    )
