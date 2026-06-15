"""Phase 3a — layered Go-toolchain validation.

This module runs the **deterministic** validation pipeline described in
``design.md`` (Algorithmic Pseudocode / Phase 3 and Components / validate.py).
Each candidate produced by Phase 2 is exercised against the native Go
toolchain in a fixed order:

    REPRO_TEST → BUILD → VET → FMT → TEST → LINT

Two ordering invariants enforced here are referenced by the design's
Correctness Properties:

* **BUILD short-circuit** — if ``go build ./...`` fails for a candidate, the
  remaining VET / FMT / TEST / LINT checks are skipped entirely (Property 7
  "Validation layering"). The :class:`ValidationReport` for that candidate
  therefore contains only the checks that actually ran, never a stale
  pass-flag for a check that could not succeed.
* **LINT availability** — ``golangci-lint`` is treated as optional. When the
  binary is not on ``PATH`` we record a :class:`CheckResult` with
  ``skipped=True`` and ``passed=False`` rather than failing the run; this
  keeps Phase 3 reproducible across machines that have only the core Go
  toolchain.

``breaks_existing_tests`` is set iff the ``TEST`` check actually ran (i.e.
was not short-circuited) and reported a failure. Phase 3b (``select.py``)
uses this flag to drop regressions before ranking.

Security
--------
Every subprocess invocation uses an argv array — no shell interpretation,
no user-controlled string interpolation — and is bounded by a wall-clock
timeout. ``cwd`` is always the candidate's isolated worktree, so a
runaway tool cannot reach outside the candidate's sandbox.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

from src import llm as _llm

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from src.config import Config
    from src.issue import Issue
    from src.repair import Candidate
    from src.repo import RepoCheckout


logger = logging.getLogger(__name__)


__all__ = [
    "CheckName",
    "CheckResult",
    "ValidationReport",
    "ValidateError",
    "generate_repro_test",
    "validate",
    "run_check",
]


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class CheckName(str, Enum):
    """Names of the layered Go-toolchain checks, mirroring design.md."""

    REPRO_TEST = "repro_test"
    BUILD = "build"
    VET = "vet"
    FMT = "fmt"
    TEST = "test"
    LINT = "lint"


@dataclass(frozen=True)
class CheckResult:
    """Outcome of a single check.

    Per the design's Data Models / ValidationReport rules:
      * ``passed`` is True iff the underlying tool exited 0 (for FMT:
        ``gofmt -l`` listed no files).
      * ``skipped`` is True iff the tool was unavailable (e.g. ``golangci-lint``
        not on PATH); when skipped, ``passed`` is always False — the two
        flags are never both True.
      * ``output`` is combined stdout/stderr captured for traceability.
    """

    name: CheckName
    passed: bool
    skipped: bool
    output: str


@dataclass(frozen=True)
class ValidationReport:
    """Per-candidate validation outcome.

    ``checks`` contains at most one entry per :class:`CheckName`. Checks
    that were short-circuited (e.g. VET after a BUILD failure) are simply
    absent from the list — this reflects "they did not run", which is
    distinct from "they ran and were skipped because the tool was missing"
    (the latter is a present entry with ``skipped=True``).

    ``breaks_existing_tests`` is True iff the ``TEST`` check ran (was not
    short-circuited) and reported a failure. Phase 3b consumes this flag
    to drop regressions.
    """

    candidate_id: str
    checks: list[CheckResult]
    breaks_existing_tests: bool


class ValidateError(RuntimeError):
    """Raised for unrecoverable failures during validation orchestration.

    Recoverable conditions (a check failing, a tool being absent, the LLM
    being unable to produce a repro test) do not raise — they are reflected
    in the returned :class:`ValidationReport`. Only genuine programmer
    errors (an unknown :class:`CheckName`) surface as ``ValidateError``.
    """


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


# Wall-clock cap (seconds) for any single tool invocation. Picked to be
# generous for moderate-sized Go modules (gin, cobra, echo) yet small
# enough that a hung subprocess fails the run quickly with an actionable
# message rather than blocking indefinitely.
_DEFAULT_TIMEOUT_S = 300

# Filename used to stage the (optional) reproduction test inside a
# candidate worktree. Keeping it deterministic lets ``run_check`` locate
# the test without extra plumbing through the public signature.
_REPRO_TEST_FILENAME = "agentless_repro_test.go"

# Subdirectory under the worktree where the staged test goes when we
# cannot match its package directive to an existing package in the repo.
_REPRO_TEST_FALLBACK_DIR = "agentless_repro"

# Path to the prompt template for repro-test generation. Resolved relative
# to the current working directory at call time so callers running from a
# different cwd can override by chdir'ing — the same convention the rest
# of the pipeline uses.
_REPRO_TEST_PROMPT_PATH = Path("prompts") / "repro_test.md"

# Cap on how much of the issue body we feed into the repro-test prompt.
_REPRO_BODY_CHAR_LIMIT = 4000

# Regex to extract the leading ``package <ident>`` directive from a Go
# source file. We deliberately do NOT match ``package <ident>_test``
# specially because Go allows external test packages and we want to
# faithfully install the file in the matching package directory either way.
_PACKAGE_DIRECTIVE_RE = re.compile(
    r"^\s*package\s+([A-Za-z_][A-Za-z0-9_]*)\s*$",
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_repro_test(
    issue: "Issue",
    checkout: "RepoCheckout",
    cfg: "Config",
) -> Optional[str]:
    """Best-effort: ask the LLM for a Go test that reproduces ``issue``.

    The expected behavior is that the test fails on the unfixed base
    checkout and passes on a correct fix — i.e. it serves as a strong
    discriminator during ranking. Generation is *optional*: if the prompt
    template is missing, the LLM call fails, or the response cannot be
    coerced into something that looks like a Go test file, this function
    returns ``None`` and the rest of Phase 3 proceeds without a repro
    test.

    Args:
        issue: Validated :class:`Issue` whose title/body provide the
            problem description for the prompt.
        checkout: The pinned base checkout. Currently unused — passed in
            for future hooks (e.g. surfacing relevant package context),
            and to keep parity with the design signature.
        cfg: Pipeline config; only ``cfg.llm`` is consulted for the
            request.

    Returns:
        A complete Go test file body (with a ``package`` directive) on
        success, or ``None`` when no usable test could be produced.
    """
    # ``checkout`` is reserved for future expansion (per the design
    # signature). Acknowledge it explicitly so linters do not flag.
    del checkout

    try:
        prompt_template = _REPRO_TEST_PROMPT_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning(
            "validate: repro test prompt template not found at %s: %s",
            _REPRO_TEST_PROMPT_PATH,
            exc,
        )
        return None

    title = (issue.title or "").strip()
    body = (issue.body or "").strip()
    if len(body) > _REPRO_BODY_CHAR_LIMIT:
        body = body[:_REPRO_BODY_CHAR_LIMIT] + "\n…[truncated]"

    prompt = (
        prompt_template
        .replace("{{issue_title}}", title)
        .replace("{{issue_body}}", body if body else "(no body provided)")
        .replace("{{relevant_files}}", "(not provided to this prompt)")
        .replace("{{package}}", _REPRO_TEST_FALLBACK_DIR)
    )
    prompt += (
        "\n\nReturn ONLY the complete Go test file body, including the "
        "`package` directive and any imports. Do NOT include code fences, "
        "prose, or commentary.\n"
    )

    try:
        responses = _llm.complete(prompt, cfg=cfg.llm, temperature=0.0, n=1)
    except Exception as exc:  # noqa: BLE001 - LLM failure is recoverable here
        logger.warning("validate: repro test LLM call failed: %s", exc)
        return None
    if not responses:
        return None

    text = _strip_code_fences((responses[0].text or "").strip())
    if not text:
        return None
    if not _PACKAGE_DIRECTIVE_RE.search(text):
        # No package directive — we cannot safely place it in the worktree.
        logger.info("validate: repro test response had no package directive; discarding")
        return None
    return text


def validate(
    candidates: Sequence["Candidate"],
    repro_test: Optional[str],
    cfg: "Config",
) -> list[ValidationReport]:
    """Run the layered Go checks for each candidate.

    Per design / Algorithmic Pseudocode / Phase 3, for every candidate:

      1. If ``repro_test`` was supplied, install it into the worktree and
         run REPRO_TEST. An install failure is recorded as a skipped
         REPRO_TEST result rather than aborting the whole candidate.
      2. Run BUILD → VET → FMT → TEST → LINT in order. If BUILD fails,
         break out — the remaining checks are not reported.
      3. Compute ``breaks_existing_tests``: True iff TEST ran (was not
         short-circuited) and did not pass.

    Args:
        candidates: Phase 2 candidates. Per the design's preconditions,
            every entry has ``applied_clean == True``.
        repro_test: Optional Go test source to stage in each worktree.
            ``None`` (or empty string) skips the REPRO_TEST step.
        cfg: Pipeline config — currently only used as a forward-compat
            handle (timeouts/tool selection may key off it later).

    Returns:
        One :class:`ValidationReport` per candidate, in input order.
    """
    # cfg is reserved for future per-run tuning of timeouts or tool sets.
    # Acknowledged here so the signature stays aligned with design.md.
    del cfg

    repro_text = repro_test if repro_test and repro_test.strip() else None

    reports: list[ValidationReport] = []
    for candidate in candidates:
        worktree = Path(candidate.worktree)
        checks: list[CheckResult] = []

        # Step 1 — optional repro test.
        if repro_text is not None:
            install_dir = _install_repro_test(worktree, repro_text)
            if install_dir is None:
                checks.append(
                    CheckResult(
                        name=CheckName.REPRO_TEST,
                        passed=False,
                        skipped=True,
                        output=(
                            "repro test could not be installed: missing or "
                            "ambiguous package directive in generated source."
                        ),
                    )
                )
            else:
                checks.append(run_check(CheckName.REPRO_TEST, worktree))

        # Step 2 — layered Go checks with BUILD short-circuit.
        layered = (
            CheckName.BUILD,
            CheckName.VET,
            CheckName.FMT,
            CheckName.TEST,
            CheckName.LINT,
        )
        for name in layered:
            result = run_check(name, worktree)
            checks.append(result)
            if name is CheckName.BUILD and not result.passed:
                # Property 7: a failed BUILD short-circuits VET/FMT/TEST/LINT.
                break

        # Step 3 — derive breaks_existing_tests from the recorded checks.
        breaks = _compute_breaks_existing_tests(checks)
        reports.append(
            ValidationReport(
                candidate_id=candidate.id,
                checks=checks,
                breaks_existing_tests=breaks,
            )
        )

    return reports


def run_check(name: CheckName, worktree: Path) -> CheckResult:
    """Run a single :class:`CheckName` in ``worktree``.

    All tool invocations use argv arrays (never ``shell=True``) and are
    bounded by :data:`_DEFAULT_TIMEOUT_S`. Tool absence is reported as a
    :class:`CheckResult` with ``skipped=True``; this is mandatory for LINT
    (per the design) and applied uniformly to BUILD/VET/TEST/FMT for
    diagnostic clarity if the Go toolchain itself happens to be missing.

    Args:
        name: The check to run.
        worktree: Worktree path; used as ``cwd`` for the subprocess.

    Returns:
        :class:`CheckResult` with ``passed`` / ``skipped`` / ``output``
        populated per the rules in the design's Key Functions / run_check
        section.

    Raises:
        ValidateError: If ``name`` is not a known :class:`CheckName`.
    """
    worktree = Path(worktree)

    if name is CheckName.REPRO_TEST:
        return _run_repro_test(worktree)
    if name is CheckName.BUILD:
        return _run_subprocess(name, ["go", "build", "./..."], worktree)
    if name is CheckName.VET:
        return _run_subprocess(name, ["go", "vet", "./..."], worktree)
    if name is CheckName.FMT:
        return _run_fmt(worktree)
    if name is CheckName.TEST:
        return _run_subprocess(
            name,
            ["go", "test", "-count=1", "./..."],
            worktree,
        )
    if name is CheckName.LINT:
        return _run_lint(worktree)

    raise ValidateError(f"unknown check name: {name!r}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_breaks_existing_tests(checks: list[CheckResult]) -> bool:
    """``True`` iff the TEST check ran (was not skipped) and did not pass.

    The pinned base commit is, by construction, expected to have green
    tests; therefore a TEST failure on the candidate worktree is taken as
    evidence that the candidate broke a previously-passing test. If TEST
    is absent from the list (BUILD short-circuit) or skipped (toolchain
    issue), we conservatively report ``False`` so the candidate is not
    incorrectly flagged as a regression — Phase 3b's ranking will still
    drop it for failing earlier checks.
    """
    for check in checks:
        if check.name is not CheckName.TEST:
            continue
        if check.skipped:
            return False
        return not check.passed
    return False


def _strip_code_fences(text: str) -> str:
    """Remove a single leading/trailing markdown code fence if present."""
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[: -len("```")]
    return text.strip()


def _install_repro_test(worktree: Path, repro_test: str) -> Optional[Path]:
    """Stage ``repro_test`` as a ``_test.go`` file inside ``worktree``.

    Strategy:
      1. Extract the ``package`` directive from the source.
      2. If a directory in the worktree already has Go files declaring
         that package (or its ``_test``-stripped form), install the test
         there so the LLM's imports of internal symbols resolve cleanly.
      3. Otherwise, materialize a fresh ``agentless_repro/`` subdirectory
         and install the file there. (Useful for tests that only call
         exported APIs via the module path.)

    Returns the directory that received the test on success, or ``None``
    when the source lacks a usable package directive.
    """
    match = _PACKAGE_DIRECTIVE_RE.search(repro_test)
    if not match:
        return None
    raw_package = match.group(1)
    # Strip a trailing ``_test`` suffix when looking for the matching
    # package directory: ``package foo_test`` lives alongside ``package foo``.
    pkg_for_lookup = raw_package
    if pkg_for_lookup.endswith("_test"):
        pkg_for_lookup = pkg_for_lookup[: -len("_test")]

    target_dir = _find_package_dir(worktree, pkg_for_lookup)
    if target_dir is None:
        target_dir = worktree / _REPRO_TEST_FALLBACK_DIR

    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / _REPRO_TEST_FILENAME
        target_file.write_text(repro_test, encoding="utf-8")
    except OSError as exc:
        logger.warning(
            "validate: failed to write repro test to %s: %s",
            target_dir,
            exc,
        )
        return None
    return target_dir


def _find_package_dir(worktree: Path, package: str) -> Optional[Path]:
    """Return the first directory whose ``.go`` files declare ``package``.

    Scans the worktree breadth-first, skipping ``vendor/`` and any dir
    whose name starts with ``.`` or ``_`` (Go's own ignore convention).
    Returns ``None`` when no matching directory is found.
    """
    if not package or not worktree.is_dir():
        return None
    skip_names = {"vendor", "node_modules"}

    # Iterative BFS; pathlib.Path.rglob would also work but we want to
    # honor the skip list cleanly.
    queue: list[Path] = [worktree]
    while queue:
        current = queue.pop(0)
        try:
            entries = list(current.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.is_dir():
                name = entry.name
                if name in skip_names or name.startswith(".") or name.startswith("_"):
                    continue
                queue.append(entry)
                continue
            if not entry.is_file() or entry.suffix != ".go":
                continue
            try:
                # Only sniff the first ~4 KB; the package directive is on
                # the first non-comment, non-blank line.
                head = entry.read_text(encoding="utf-8", errors="replace")[:4096]
            except OSError:
                continue
            m = _PACKAGE_DIRECTIVE_RE.search(head)
            if m and m.group(1) == package:
                return current
    return None


def _run_repro_test(worktree: Path) -> CheckResult:
    """Run the staged reproduction test, scoped to its package directory.

    Looks for the staged file (named :data:`_REPRO_TEST_FILENAME`)
    anywhere under the worktree, then invokes ``go test -count=1`` against
    its parent directory. If the file cannot be found we report the check
    as skipped so the rest of validation continues unimpeded.
    """
    matches = list(worktree.rglob(_REPRO_TEST_FILENAME))
    if not matches:
        return CheckResult(
            name=CheckName.REPRO_TEST,
            passed=False,
            skipped=True,
            output=(
                f"no {_REPRO_TEST_FILENAME} found in worktree; "
                "repro test was not installed."
            ),
        )

    test_dir = matches[0].parent
    try:
        rel_parent = test_dir.relative_to(worktree).as_posix()
    except ValueError:
        rel_parent = ""
    if rel_parent in ("", "."):
        target = "./..."
    else:
        target = f"./{rel_parent}/..."

    return _run_subprocess(
        CheckName.REPRO_TEST,
        ["go", "test", "-count=1", target],
        worktree,
    )


def _run_fmt(worktree: Path) -> CheckResult:
    """Run ``gofmt -l .``: passed iff the tool exits 0 and prints nothing.

    ``gofmt -l`` writes the path of every file that differs from canonical
    formatting to stdout and exits 0 even when files are listed; the
    "no formatting issues" condition is therefore "stdout empty AND
    exit 0" rather than "exit 0" alone. This matches the design's
    Key Functions / run_check contract.
    """
    try:
        proc = subprocess.run(  # noqa: S603 - argv array, no shell
            ["gofmt", "-l", "."],
            cwd=str(worktree),
            check=False,
            capture_output=True,
            text=True,
            timeout=_DEFAULT_TIMEOUT_S,
        )
    except FileNotFoundError:
        return CheckResult(
            name=CheckName.FMT,
            passed=False,
            skipped=True,
            output="gofmt not found on PATH",
        )
    except subprocess.TimeoutExpired as exc:
        return CheckResult(
            name=CheckName.FMT,
            passed=False,
            skipped=False,
            output=f"gofmt -l . timed out after {exc.timeout}s",
        )
    except OSError as exc:
        return CheckResult(
            name=CheckName.FMT,
            passed=False,
            skipped=True,
            output=f"gofmt could not be executed: {exc}",
        )

    listed = (proc.stdout or "").strip()
    passed = proc.returncode == 0 and listed == ""
    output = _format_subprocess_output(["gofmt", "-l", "."], proc.stdout, proc.stderr, proc.returncode)
    return CheckResult(name=CheckName.FMT, passed=passed, skipped=False, output=output)


def _run_lint(worktree: Path) -> CheckResult:
    """Run ``golangci-lint run ./...``.

    Per design: when ``golangci-lint`` is not on PATH, the LINT check is
    marked ``skipped=True`` (and ``passed=False``). This covers reviewer
    machines that have only the core Go toolchain.
    """
    if shutil.which("golangci-lint") is None:
        return CheckResult(
            name=CheckName.LINT,
            passed=False,
            skipped=True,
            output="golangci-lint not on PATH; LINT skipped.",
        )
    return _run_subprocess(
        CheckName.LINT,
        ["golangci-lint", "run", "./..."],
        worktree,
    )


def _run_subprocess(
    name: CheckName,
    args: Sequence[str],
    worktree: Path,
    *,
    timeout: int = _DEFAULT_TIMEOUT_S,
) -> CheckResult:
    """Execute ``args`` in ``worktree`` and translate the result to a CheckResult.

    Argv array is used directly; ``shell=True`` is never set. Output is
    captured (text mode) so the full stdout/stderr is available for
    traceability in :attr:`CheckResult.output`.
    """
    try:
        proc = subprocess.run(  # noqa: S603 - argv array, no shell
            list(args),
            cwd=str(worktree),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return CheckResult(
            name=name,
            passed=False,
            skipped=True,
            output=f"{args[0]} not found on PATH",
        )
    except subprocess.TimeoutExpired as exc:
        return CheckResult(
            name=name,
            passed=False,
            skipped=False,
            output=f"{' '.join(args)} timed out after {exc.timeout}s",
        )
    except OSError as exc:
        return CheckResult(
            name=name,
            passed=False,
            skipped=True,
            output=f"{args[0]} could not be executed: {exc}",
        )

    output = _format_subprocess_output(args, proc.stdout, proc.stderr, proc.returncode)
    return CheckResult(
        name=name,
        passed=(proc.returncode == 0),
        skipped=False,
        output=output,
    )


def _format_subprocess_output(
    args: Sequence[str],
    stdout: Optional[str],
    stderr: Optional[str],
    returncode: int,
) -> str:
    """Render captured subprocess output for traceability fields."""
    cmd_repr = " ".join(args)
    return (
        f"$ {cmd_repr}\n"
        f"exit: {returncode}\n"
        f"--- stdout ---\n{stdout or ''}"
        f"--- stderr ---\n{stderr or ''}"
    )
