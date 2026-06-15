"""Phase 3b — rank validation reports and pick a single winning candidate.

This module is the deterministic counterpart to :mod:`src.validate`. Where
``validate.py`` is permitted to be noisy (subprocess timing, environment
quirks), ``select.py`` is a pure function over already-recorded
:class:`~src.validate.ValidationReport` objects: given a fixed report set
(and the candidates they describe) we always pick the same winner, no matter
what order the inputs arrive in.

The ranking lives in two pieces:

* :func:`score_key` produces a lexicographic tuple of pass/fail booleans.
  The order is fixed by the design — reproduction test, existing tests,
  vet-and-build clean, lint, fmt — so a higher-priority pass dominates any
  number of lower-priority passes.
* :func:`select` first drops candidates whose validation report flags
  ``breaks_existing_tests`` (Property 5), then sorts the rest by
  ``score_key`` descending. If the top score is shared by multiple
  candidates the tie is broken by **majority vote over normalized diffs**:
  the most popular normalized diff wins, and within that group we fall back
  to the lexicographically smallest ``candidate.id`` so the result is
  invariant to input ordering (Property 8).

We intentionally do *not* import :class:`~src.validate.CheckName` at
runtime. ``CheckName`` is a ``(str, Enum)`` whose members compare equal to
their string values, so the small set of canonical names we care about is
encoded as plain string constants below. That keeps :mod:`src.select`
importable in isolation (handy for unit tests that build synthetic reports)
and avoids a hard cycle with :mod:`src.validate`.

Design alignment:
    * Component / Phase 3b: select.py.
    * Algorithmic Pseudocode / Phase 3 (``select`` + ``score_key``).
    * Correctness Properties 5 (no regressions selected), 6 (ranking
      monotonicity), and 8 (deterministic selection).
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover - type-only imports
    from src.config import Config
    from src.repair import Candidate
    from src.validate import ValidationReport


logger = logging.getLogger(__name__)


__all__ = [
    "RankedCandidate",
    "SelectionError",
    "score_key",
    "select",
    "CHECK_REPRO_TEST",
    "CHECK_TEST",
    "CHECK_VET",
    "CHECK_BUILD",
    "CHECK_LINT",
    "CHECK_FMT",
]


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class SelectionError(ValueError):
    """Raised for structural problems in the inputs to :func:`select`.

    Examples include a report referencing an unknown ``candidate_id`` or a
    candidate with no matching report. These represent caller bugs (Phase 3
    is supposed to produce one report per candidate); they are *not* the
    same as "no winner exists", which is signalled by ``select`` returning
    ``None``.
    """


@dataclass(frozen=True)
class RankedCandidate:
    """A candidate paired with its validation report and lexicographic score.

    Attributes:
        candidate: The :class:`~src.repair.Candidate` produced by Phase 2.
        report: The :class:`~src.validate.ValidationReport` for that
            candidate produced by Phase 3a.
        score: Lexicographic ranking tuple from :func:`score_key`. Larger
            tuples (under the natural tuple comparison) rank higher.
    """

    candidate: "Candidate"
    report: "ValidationReport"
    score: tuple[bool, bool, bool, bool, bool]


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


# String values mirror :class:`src.validate.CheckName`. Because that enum
# inherits from ``str``, ``CheckName.REPRO_TEST == "repro_test"`` is True,
# so reading ``check.name`` and comparing against these constants works
# regardless of whether the caller hands us enum instances or raw strings.
CHECK_REPRO_TEST: str = "repro_test"
CHECK_TEST: str = "test"
CHECK_VET: str = "vet"
CHECK_BUILD: str = "build"
CHECK_LINT: str = "lint"
CHECK_FMT: str = "fmt"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_key(
    pair: tuple["Candidate", "ValidationReport"],
) -> tuple[bool, bool, bool, bool, bool]:
    """Compute the lexicographic ranking key for one (candidate, report) pair.

    Tuple positions, in priority order from highest to lowest:

    1. ``passed(REPRO_TEST)`` — the candidate fixes the bug per the
       generated reproduction test.
    2. ``passed(TEST)`` — the existing test suite still passes.
    3. ``passed(VET) and passed(BUILD)`` — the candidate is well-formed Go
       code that survives static analysis. Both must pass; either failing
       collapses the slot to ``False``.
    4. ``passed(LINT)`` — the candidate is clean under ``golangci-lint``.
       A skipped LINT (tool absent) counts as ``False`` — we never invent a
       pass.
    5. ``passed(FMT)`` — ``gofmt -l`` listed nothing.

    Tuple comparison in Python is lexicographic, so a candidate that wins
    on a higher-priority slot is automatically ranked above one that loses
    that slot but wins lower slots. This is exactly Property 6 (ranking
    monotonicity).

    Args:
        pair: A ``(candidate, report)`` 2-tuple. The candidate is currently
            unused — kept in the signature so callers can pass the same
            shape into :func:`sorted` and the design pseudocode lines up.

    Returns:
        A 5-tuple of booleans suitable for descending sort.
    """
    _, report = pair
    return (
        _passed(report, CHECK_REPRO_TEST),
        _passed(report, CHECK_TEST),
        _passed(report, CHECK_VET) and _passed(report, CHECK_BUILD),
        _passed(report, CHECK_LINT),
        _passed(report, CHECK_FMT),
    )


def select(
    candidates: list["Candidate"],
    reports: list["ValidationReport"],
    cfg: "Config",
) -> Optional[RankedCandidate]:
    """Rank validated candidates and return the single winner, or ``None``.

    Algorithm (matches design.md Phase 3 pseudocode and Properties 5/6/8):

    1. Pair each report with its candidate by ``candidate_id``. Reports
       without a matching candidate, or candidates without a matching
       report, raise :class:`SelectionError` — that signals a Phase 3 bug,
       not a "no winner" outcome.
    2. Drop pairs whose report has ``breaks_existing_tests == True``
       (Property 5: regressions are never selected).
    3. If no pairs remain, return ``None``.
    4. Sort the remaining pairs by :func:`score_key` descending. Tuple
       comparison gives us the lexicographic ordering required by
       Property 6.
    5. Identify the top-score group (every pair whose key equals the
       maximum). If only one pair shares the top score, it wins.
    6. Otherwise break ties by **majority vote over normalized diffs**:
       count how many tied candidates produced each ``candidate.diff``
       and take the most popular diff. If multiple diffs share the top
       count, choose the lexicographically smallest diff string — this is
       deterministic and independent of input order.
    7. Within the winning diff group, return the candidate with the
       lexicographically smallest ``candidate.id``. This is the
       "stable order" fallback: because ``candidate.id`` is part of the
       fixed report set, the result is invariant under shuffling of the
       input lists, which is exactly Property 8.

    Args:
        candidates: Candidates produced by :mod:`src.repair`. Order does
            not affect the result.
        reports: Validation reports produced by :mod:`src.validate`. Order
            does not affect the result.
        cfg: Pipeline config. Currently unused but accepted for parity
            with the design signature and to leave room for future
            tuning knobs (e.g. weighting LINT differently).

    Returns:
        The :class:`RankedCandidate` for the chosen winner, or ``None`` if
        every candidate breaks at least one previously-passing test.

    Raises:
        SelectionError: When ``candidates`` and ``reports`` cannot be
            paired one-to-one by ``candidate_id``.
    """
    # cfg is reserved for future tuning; explicit ``del`` documents intent.
    del cfg

    pairs = _pair_candidates_with_reports(candidates, reports)

    eligible = [p for p in pairs if not p[1].breaks_existing_tests]
    if not eligible:
        logger.info(
            "select: every candidate (%d total) breaks existing tests; no winner",
            len(pairs),
        )
        return None

    # Decorate each eligible pair with its score so we don't recompute.
    scored = [(score_key(p), p) for p in eligible]

    # Find the maximum score, then the tie group at that score. We compare
    # tuples directly, which gives us the lexicographic order required by
    # Property 6 without needing a dedicated sort.
    top_score = max(score for score, _ in scored)
    tied = [pair for score, pair in scored if score == top_score]

    if len(tied) == 1:
        winner = tied[0]
        logger.info(
            "select: winner is %s with score %s (no ties at top)",
            winner[0].id,
            top_score,
        )
        return RankedCandidate(
            candidate=winner[0], report=winner[1], score=top_score
        )

    winner = _break_tie_by_majority_vote(tied)
    logger.info(
        "select: winner is %s with score %s after majority vote across %d tied candidates",
        winner[0].id,
        top_score,
        len(tied),
    )
    return RankedCandidate(candidate=winner[0], report=winner[1], score=top_score)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _passed(report: "ValidationReport", check_name: str) -> bool:
    """Return True iff ``report`` contains a ``passed`` entry for ``check_name``.

    A check that is missing from the report, or present but skipped/failed,
    counts as not passed. This is conservative on purpose — Property 7
    (validation layering) relies on never reporting a downstream check as
    passed when an upstream one failed (e.g. BUILD failed → TEST is not
    even run, so it can't have ``passed=True``). We never invent a pass.
    """
    for chk in report.checks:
        # ``chk.name`` may be a CheckName enum (which inherits from str) or
        # a plain string in synthetic test fixtures; both compare equal to
        # the canonical lowercase token.
        if chk.name == check_name and chk.passed and not chk.skipped:
            return True
    return False


def _pair_candidates_with_reports(
    candidates: list["Candidate"],
    reports: list["ValidationReport"],
) -> list[tuple["Candidate", "ValidationReport"]]:
    """Pair candidates with reports by ``candidate_id`` (order-independent).

    The design guarantees Phase 3 emits exactly one report per candidate,
    so we enforce a strict one-to-one mapping. Mismatches raise
    :class:`SelectionError` rather than silently dropping data — a Phase 3
    bug should be loud, not selectively forgotten.
    """
    by_id: dict[str, "Candidate"] = {}
    for c in candidates:
        if c.id in by_id:
            raise SelectionError(f"duplicate candidate id: {c.id!r}")
        by_id[c.id] = c

    pairs: list[tuple["Candidate", "ValidationReport"]] = []
    seen_report_ids: set[str] = set()
    for r in reports:
        if r.candidate_id in seen_report_ids:
            raise SelectionError(
                f"duplicate validation report for candidate {r.candidate_id!r}"
            )
        seen_report_ids.add(r.candidate_id)
        cand = by_id.get(r.candidate_id)
        if cand is None:
            raise SelectionError(
                f"validation report references unknown candidate "
                f"{r.candidate_id!r}"
            )
        pairs.append((cand, r))

    missing = set(by_id) - seen_report_ids
    if missing:
        raise SelectionError(
            "no validation report for candidate(s): "
            + ", ".join(sorted(repr(m) for m in missing))
        )

    return pairs


def _break_tie_by_majority_vote(
    tied: list[tuple["Candidate", "ValidationReport"]],
) -> tuple["Candidate", "ValidationReport"]:
    """Pick a single winner from a top-score tie by diff popularity.

    The tied list contains two or more (candidate, report) pairs that all
    share the maximal :func:`score_key`. We:

    1. Tally normalized diffs across the tied candidates.
    2. Find the maximum count. If a single diff string has it, that diff
       wins outright. If multiple diff strings share the max count, choose
       the lexicographically smallest diff — this is deterministic and
       has the pleasant side-effect of being reproducible across machines.
    3. Among the candidates carrying the winning diff, return the one with
       the lexicographically smallest ``candidate.id``. Because
       ``candidate.id`` is set at Phase 2 (``c0``, ``c1``, …) and travels
       with the report, this is invariant under any input shuffle.

    Args:
        tied: Two-or-more (candidate, report) pairs at the maximal score.
            Caller guarantees ``len(tied) >= 2``.

    Returns:
        The chosen (candidate, report) pair.
    """
    assert len(tied) >= 2, "majority-vote tie-break requires at least two pairs"

    # Step 1 — diff frequency across the tied group.
    diff_counts: Counter[str] = Counter(c.diff for c, _ in tied)
    max_count = max(diff_counts.values())

    # Step 2 — pick the winning diff. ``Counter.most_common`` does not
    # guarantee a deterministic order across diffs with equal counts, so
    # we resolve ties by sorted diff string explicitly.
    candidate_diffs = sorted(d for d, n in diff_counts.items() if n == max_count)
    winning_diff = candidate_diffs[0]

    # Step 3 — within the winning diff group, pick by smallest candidate.id.
    in_winning_group = [pair for pair in tied if pair[0].diff == winning_diff]
    in_winning_group.sort(key=lambda pair: pair[0].id)
    return in_winning_group[0]
