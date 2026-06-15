"""Phase 2 — multi-candidate patch generation with format prefilter.

This module is the bridge between the localization phase (which produces a
ranked list of :class:`EditLocation`) and the validation phase (which runs
the Go toolchain over each candidate). The strategy is intentionally
"agentless": we ask Claude for ``n`` independent SEARCH/REPLACE samples at
moderate temperature, parse them, and try to apply each to a fresh worktree.
Any sample that parses to zero blocks or fails to apply is marked
``applied_clean=False`` (or dropped entirely if it was malformed at the
parser level). Successful applications are immediately normalized — gofmt
and goimports run inside the worktree, the resulting unified diff is
rewritten so that runs differing only in git's object hashes compare equal —
so downstream phases see uniformly-formatted patches.

Design alignment:
    * ``Candidate`` matches Data Models / Candidate in design.md.
    * ``generate_candidates`` follows Algorithmic Pseudocode / Phase 2.
    * ``MODERATE_TEMP`` is the moderate-temperature sampling constant
      referenced in the pseudocode.

Security:
    Subprocess calls (gofmt, goimports) always use argv arrays with
    ``cwd=worktree`` — no shell interpolation, no path that could escape
    the worktree. Tool absence is treated as a warning rather than a hard
    failure: the candidate is still emitted, just without the formatter
    pass. Validation in Phase 3 will catch any formatting deviation.
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.llm import complete
from src.patch import SearchReplaceBlock, apply_blocks, parse_blocks
from src.repo import make_diff

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from src.config import Config
    from src.issue import Issue
    from src.localize import EditLocation
    from src.repo import RepoCheckout


logger = logging.getLogger(__name__)


__all__ = ["Candidate", "RepairError", "MODERATE_TEMP", "generate_candidates"]


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class RepairError(RuntimeError):
    """Raised for unrecoverable failures during candidate generation.

    Recoverable conditions (a single sample parsing to zero blocks, a single
    sample failing to apply, a missing formatter tool) do not raise — they
    are reflected in the returned :class:`Candidate` list. Only genuine
    infrastructure failures (LLM auth, IO errors against the worktree
    parent) surface as ``RepairError``.
    """


@dataclass(frozen=True)
class Candidate:
    """One candidate fix derived from a single LLM sample.

    Attributes:
        id: Stable identifier of the form ``"c<index>"`` matching the
            sample's position in the LLM response list.
        blocks: Parsed SEARCH/REPLACE blocks from the sample.
        worktree: Isolated worktree where this candidate was applied (or
            attempted). Always a fresh copy — different candidates never
            share a worktree, so later validation phases can run them in
            parallel without interference.
        diff: Normalized unified diff against the base commit. Empty when
            ``applied_clean`` is False.
        applied_clean: True iff every block in ``blocks`` matched and
            replaced exactly once and the formatter pass completed (or was
            skipped because the tool was absent).
    """

    id: str
    blocks: list[SearchReplaceBlock]
    worktree: Path
    diff: str
    applied_clean: bool


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


# Moderate sampling temperature for repair candidates. Higher than the
# near-zero temperature we use for localization (which wants determinism)
# but well below 1.0 so samples remain coherent and grounded in the
# provided context. Calibrated against the design intent in
# Algorithmic Pseudocode / Phase 2.
MODERATE_TEMP: float = 0.6

# Cap individual context blocks (per location) and the overall prompt to
# keep us comfortably under Claude's input window even with many locations.
_PER_LOCATION_CONTEXT_CHARS = 600
_PROMPT_CHAR_BUDGET = 8000

# Pre-compiled regex for the unified diff ``index <sha>..<sha> [mode]`` line.
# We rewrite the SHAs to a canonical zero so two diffs differing only in
# git object hashes (which are content-derived but order-sensitive) compare
# equal at the string level.
_INDEX_LINE_RE = re.compile(
    r"^index\s+[0-9a-f]+\.\.[0-9a-f]+(\s+\d+)?\s*$",
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_candidates(
    checkout: "RepoCheckout",
    issue: "Issue",
    locations: list["EditLocation"],
    cfg: "Config",
) -> list[Candidate]:
    """Generate ``cfg.llm.sample_count`` patch candidates for ``issue``.

    For each LLM sample we:
      1. Parse SEARCH/REPLACE blocks. Drop the sample entirely if zero
         blocks were extracted (the sample is treated as malformed and
         no :class:`Candidate` is produced).
      2. Materialize a fresh worktree via ``clean_worktree`` so candidates
         never observe each other's edits.
      3. Try to apply the blocks atomically. On success run gofmt and
         goimports inside the worktree, compute the normalized diff,
         and emit ``applied_clean=True``. On failure emit
         ``applied_clean=False`` with an empty diff so the caller can see
         which samples were attempted (useful for diagnostics) without
         routing them to validation.

    Args:
        checkout: Pinned base checkout produced by :func:`prepare_repo`.
        issue: Validated issue to patch.
        locations: Edit locations from Phase 1. May be empty.
        cfg: Pipeline config; only ``cfg.llm`` is used for the call and
            ``cfg.llm.sample_count`` for the fan-out width.

    Returns:
        One :class:`Candidate` per non-malformed sample, in sample order.

    Raises:
        RepairError: On unrecoverable failures (LLM auth, repeated
            transient failures, IO errors creating worktrees).
    """
    # Lazy import to avoid a circular module load in environments that wire
    # repo helpers via a registry. Keeping this here rather than at the
    # module level also makes it trivial for tests to monkeypatch
    # ``src.repo.clean_worktree`` after import.
    from src.repo import clean_worktree

    prompt = _build_repair_prompt(issue, locations, cfg)
    try:
        samples = complete(
            prompt,
            cfg=cfg.llm,
            temperature=MODERATE_TEMP,
            n=cfg.llm.sample_count,
        )
    except Exception as exc:  # noqa: BLE001 - re-wrap as RepairError
        raise RepairError(f"LLM sampling failed: {exc}") from exc

    candidates: list[Candidate] = []
    for i, sample in enumerate(samples):
        blocks = parse_blocks(sample.text)
        if not blocks:
            logger.info(
                "repair: sample %d produced zero parseable SEARCH/REPLACE blocks; skipping",
                i,
            )
            continue

        try:
            worktree = clean_worktree(checkout)
        except Exception as exc:  # noqa: BLE001 - re-wrap so the caller sees RepairError
            raise RepairError(
                f"failed to materialize worktree for candidate c{i}: {exc}"
            ) from exc

        result = apply_blocks(worktree, blocks)
        if result.applied:
            _run_gofmt_and_goimports(worktree)
            try:
                raw_diff = make_diff(worktree)
            except Exception as exc:  # noqa: BLE001 - diff is best-effort here
                logger.warning(
                    "repair: candidate c%d applied but make_diff failed: %s",
                    i,
                    exc,
                )
                raw_diff = ""
            diff = _normalize_diff(raw_diff)
            candidates.append(
                Candidate(
                    id=f"c{i}",
                    blocks=blocks,
                    worktree=worktree,
                    diff=diff,
                    applied_clean=True,
                )
            )
        else:
            logger.info(
                "repair: candidate c%d failed to apply: %s",
                i,
                result.reason,
            )
            candidates.append(
                Candidate(
                    id=f"c{i}",
                    blocks=blocks,
                    worktree=worktree,
                    diff="",
                    applied_clean=False,
                )
            )

    return candidates


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_repair_prompt(
    issue: "Issue",
    locations: list["EditLocation"],
    cfg: "Config",
) -> str:
    """Compose the strict SEARCH/REPLACE repair prompt for the LLM.

    The prompt has three sections:
      1. Task framing — fix the issue, return SEARCH/REPLACE blocks only.
      2. Issue title and (truncated) body so the model has the user-visible
         description.
      3. One block per :class:`EditLocation`: file/symbol/line range plus
         a context excerpt truncated to :data:`_PER_LOCATION_CONTEXT_CHARS`.

    A worked example using the exact marker tokens from :mod:`src.patch`
    (``<<<<<<< SEARCH``, ``=======``, ``>>>>>>> REPLACE``) is appended so
    the parser will accept the model's output verbatim.

    The total prompt is truncated to :data:`_PROMPT_CHAR_BUDGET` characters
    by dropping trailing per-location blocks; the example and instructions
    are always preserved.
    """
    # cfg is accepted for parity with the design signature and to leave
    # room for future per-run prompt tuning (e.g. style hints). It is
    # currently unused by the prompt body itself.
    del cfg

    header = (
        "You are an expert Go contributor. Read the issue and the listed "
        "edit locations, then propose a focused fix as one or more "
        "SEARCH/REPLACE blocks.\n"
        "\n"
        "Rules:\n"
        "- Output ONLY SEARCH/REPLACE blocks. No prose, no commentary, no "
        "code fences.\n"
        "- Use the exact markers shown below, on their own lines.\n"
        "- The SEARCH text must match the file byte-for-byte and appear "
        "exactly once.\n"
        "- Keep the change minimal and focused on the reported bug.\n"
        "- You may emit multiple blocks across multiple files when the fix "
        "requires it.\n"
    )

    example = (
        "Format example (use these exact markers):\n"
        "\n"
        "<<<<<<< SEARCH path/to/file.go\n"
        "    if x == nil {\n"
        "        return errors.New(\"x is nil\")\n"
        "    }\n"
        "=======\n"
        "    if x == nil {\n"
        "        return fmt.Errorf(\"x is nil: %w\", ErrInvalid)\n"
        "    }\n"
        ">>>>>>> REPLACE\n"
    )

    issue_title = (issue.title or "").strip()
    issue_body = (issue.body or "").strip()
    if len(issue_body) > 2000:
        issue_body = issue_body[:2000] + "\n…[truncated]"

    issue_section = (
        f"Issue #{issue.number} ({issue.repo}): {issue_title}\n"
        f"\n"
        f"Body:\n{issue_body if issue_body else '(no body)'}\n"
    )

    location_blocks: list[str] = []
    for idx, loc in enumerate(locations):
        ctx = (loc.context or "").strip()
        if len(ctx) > _PER_LOCATION_CONTEXT_CHARS:
            ctx = ctx[:_PER_LOCATION_CONTEXT_CHARS] + "\n…[context truncated]"
        symbol = loc.symbol or "(no symbol)"
        location_blocks.append(
            f"[location {idx + 1}] file={loc.file_path} symbol={symbol} "
            f"lines={loc.start_line}-{loc.end_line}\n"
            f"---\n{ctx}\n---\n"
        )

    locations_section = (
        "Edit locations identified by Phase 1:\n\n" + "\n".join(location_blocks)
        if location_blocks
        else "Edit locations identified by Phase 1: (none — emit no blocks)\n"
    )

    # Assemble while staying within the prompt budget. Drop locations from
    # the tail until the prompt fits; the header, issue section, and example
    # are non-negotiable so the format is always demonstrable.
    fixed = f"{header}\n{example}\n{issue_section}\n"
    while True:
        prompt = fixed + locations_section
        if len(prompt) <= _PROMPT_CHAR_BUDGET or not location_blocks:
            return prompt
        location_blocks.pop()
        locations_section = (
            "Edit locations identified by Phase 1 (truncated to fit):\n\n"
            + "\n".join(location_blocks)
            if location_blocks
            else "Edit locations identified by Phase 1: (none — emit no blocks)\n"
        )


def _run_gofmt_and_goimports(worktree: Path) -> None:
    """Run ``gofmt -w .`` then ``goimports -w .`` inside ``worktree``.

    Both invocations are best-effort:
      * If the tool is not on PATH (``FileNotFoundError``), we log at WARNING
        and continue.
      * If the tool runs but exits non-zero (e.g. a Go file we did not
        author has a syntax error), we log at WARNING with the captured
        stderr and continue. Validation in Phase 3 will reject the
        candidate downstream if formatting is actually broken.

    All commands use argv arrays with ``cwd=worktree`` so paths cannot
    escape the worktree and the shell is never invoked.
    """
    for tool, args in (
        ("gofmt", ["gofmt", "-w", "."]),
        ("goimports", ["goimports", "-w", "."]),
    ):
        try:
            proc = subprocess.run(  # noqa: S603 - argv array, no shell
                args,
                cwd=str(worktree),
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            logger.warning(
                "repair: %s not found on PATH; skipping format pass for %s",
                tool,
                worktree,
            )
            continue
        except OSError as exc:
            logger.warning(
                "repair: %s could not be executed (%s); skipping format pass for %s",
                tool,
                exc,
                worktree,
            )
            continue
        if proc.returncode != 0:
            logger.warning(
                "repair: %s exited %d in %s; stderr=%s",
                tool,
                proc.returncode,
                worktree,
                (proc.stderr or "").strip()[:500],
            )


def _normalize_diff(diff: str) -> str:
    """Normalize a unified diff so equivalent runs compare equal.

    Transformations applied (non-destructive — hunk content is preserved):
      * Convert ``\r\n`` line endings to ``\n``.
      * Strip trailing whitespace from every line.
      * Rewrite ``index <sha>..<sha> [mode]`` lines to
        ``index 0000000..0000000 <mode>`` so two diffs that differ only in
        git's object hashes (a function of file content order) compare
        equal under string equality.

    Args:
        diff: Raw unified diff produced by ``git diff``.

    Returns:
        Normalized diff. Empty input yields empty output.
    """
    if not diff:
        return ""
    text = diff.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    def _rewrite_index(match: re.Match[str]) -> str:
        mode = match.group(1) or ""
        return f"index 0000000..0000000{mode}".rstrip()

    text = _INDEX_LINE_RE.sub(_rewrite_index, text)
    return text
