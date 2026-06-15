"""SEARCH/REPLACE patch parsing and atomic application.

The repair phase asks the LLM to emit edits in a strict ``SEARCH/REPLACE``
block format (chosen over unified diffs, which fail to apply silently when
the surrounding context drifts). This module parses that format and applies
each block as an exact, single-match string replacement against a writable
worktree.

Block format
------------
Each block is delimited by three markers, each on its own line. The target
file path is placed on the same line as the opening ``SEARCH`` marker,
separated by a single space:

.. code-block:: text

    <<<<<<< SEARCH path/to/file.go
    ... exact text to find ...
    =======
    ... replacement text ...
    >>>>>>> REPLACE

Leading and trailing whitespace around marker *lines* is tolerated, but the
marker tokens themselves (``<<<<<<< SEARCH``, ``=======``, ``>>>>>>> REPLACE``)
must match exactly. A block that is missing the separator or the closing
marker is skipped with a logged warning rather than raising — the caller
treats an empty result as "no applicable patch".

Application semantics
---------------------
:func:`apply_blocks` is **atomic**: either every block applies successfully
or the worktree is restored byte-for-byte to its pre-call state. Each
block's ``search`` text must occur **exactly once** in the current file
contents; zero matches and non-unique matches are both treated as failures
and trigger a full rollback. Multiple blocks may target the same file; in
that case each block matches against the contents produced by the prior
block, but rollback still reverts the file to its original snapshot.

Path escapes (absolute paths, or relative paths that resolve outside the
worktree via ``..``) are rejected before any file is read.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Marker tokens. ``SEARCH_MARKER_PREFIX`` is followed by a single space and
# the file path on the same line.
SEARCH_MARKER_PREFIX = "<<<<<<< SEARCH"
SEPARATOR_MARKER = "======="
REPLACE_MARKER = ">>>>>>> REPLACE"


class PatchError(RuntimeError):
    """Raised for unrecoverable IO failures during apply or rollback.

    Recoverable conditions (zero matches, non-unique matches, path escapes)
    are reported via :class:`ApplyResult` with ``applied=False``. Only
    genuine IO problems surface as ``PatchError``.
    """


@dataclass(frozen=True)
class SearchReplaceBlock:
    """A single exact-match SEARCH/REPLACE edit targeting one file.

    Attributes:
        file_path: Path of the file to edit, relative to the worktree root.
        search:    Exact substring expected to appear once in the file.
        replace:   Replacement text to substitute for ``search``.
    """

    file_path: str
    search: str
    replace: str


@dataclass(frozen=True)
class ApplyResult:
    """Outcome of :func:`apply_blocks`.

    Attributes:
        applied: True iff every block applied with an exact single match.
        reason:  Human-readable explanation when ``applied`` is False;
                 ``None`` on success.
    """

    applied: bool
    reason: Optional[str]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_blocks(llm_text: str) -> list[SearchReplaceBlock]:
    """Parse SEARCH/REPLACE blocks from raw LLM output.

    The parser is tolerant of surrounding prose and stray whitespace around
    marker lines but strict about the marker tokens themselves. Malformed
    blocks (missing separator or closing marker, missing file path, empty
    search text) are skipped with a logged warning. If the input contains
    no valid blocks the function returns an empty list — callers handle
    the empty case.

    Args:
        llm_text: Raw model output that may contain zero or more blocks
            interleaved with arbitrary prose.

    Returns:
        List of :class:`SearchReplaceBlock` in source order.
    """
    if not llm_text:
        return []

    lines = llm_text.splitlines()
    blocks: list[SearchReplaceBlock] = []
    i = 0
    n = len(lines)

    while i < n:
        stripped = lines[i].strip()
        if not stripped.startswith(SEARCH_MARKER_PREFIX):
            i += 1
            continue

        header_start = i
        # Extract file path from the header line. Format:
        #   "<<<<<<< SEARCH path/to/file.go"
        remainder = stripped[len(SEARCH_MARKER_PREFIX):].strip()
        if not remainder:
            logger.warning(
                "skipping SEARCH block at line %d: missing file path on header",
                header_start + 1,
            )
            i += 1
            continue
        file_path = remainder

        # Walk forward collecting search lines until the separator.
        i += 1
        search_lines: list[str] = []
        found_separator = False
        while i < n:
            if lines[i].strip() == SEPARATOR_MARKER:
                found_separator = True
                break
            # A new SEARCH header before the separator means the prior block
            # is malformed; abandon it and let the outer loop pick up the
            # new header on the next iteration.
            if lines[i].strip().startswith(SEARCH_MARKER_PREFIX):
                break
            search_lines.append(lines[i])
            i += 1

        if not found_separator:
            logger.warning(
                "skipping SEARCH block at line %d: missing '=======' separator",
                header_start + 1,
            )
            continue  # do not advance i; outer loop handles it

        # Walk forward collecting replace lines until the closing marker.
        i += 1  # step past the separator line
        replace_lines: list[str] = []
        found_end = False
        while i < n:
            if lines[i].strip() == REPLACE_MARKER:
                found_end = True
                break
            if lines[i].strip().startswith(SEARCH_MARKER_PREFIX):
                break
            replace_lines.append(lines[i])
            i += 1

        if not found_end:
            logger.warning(
                "skipping SEARCH block at line %d: missing '>>>>>>> REPLACE' marker",
                header_start + 1,
            )
            continue

        # Step past the closing marker so the outer loop resumes after it.
        i += 1

        search_text = "\n".join(search_lines)
        replace_text = "\n".join(replace_lines)

        if not search_text:
            logger.warning(
                "skipping SEARCH block at line %d: empty search text",
                header_start + 1,
            )
            continue

        blocks.append(
            SearchReplaceBlock(
                file_path=file_path,
                search=search_text,
                replace=replace_text,
            )
        )

    return blocks


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


def apply_blocks(worktree: Path, blocks: list[SearchReplaceBlock]) -> ApplyResult:
    """Apply SEARCH/REPLACE blocks to ``worktree`` atomically.

    Each block's ``search`` text must match exactly once in its target file.
    On any zero-match or non-unique-match, every prior edit made by this
    call is rolled back and the function returns ``applied=False`` with a
    human-readable ``reason``. On success every distinct file referenced
    by the blocks has been updated and ``applied=True``.

    Multiple blocks may target the same file; later blocks see the
    contents produced by earlier blocks. Rollback nevertheless restores
    the **original** snapshot of each file (taken before any edits).

    Args:
        worktree: A writable clean copy of the base checkout. All paths
            are resolved relative to this directory.
        blocks: Non-empty list of edits to apply.

    Returns:
        :class:`ApplyResult` describing the outcome.

    Raises:
        PatchError: On unrecoverable IO failures during apply or rollback.
    """
    if not blocks:
        return ApplyResult(applied=False, reason="no blocks")

    try:
        worktree_root = worktree.resolve(strict=True)
    except FileNotFoundError as exc:
        raise PatchError(f"worktree does not exist: {worktree}") from exc

    # Resolve every target path up-front and reject anything that escapes
    # the worktree. Doing this before reading or writing anything keeps a
    # malicious or buggy block list from observing partial state.
    resolved_paths: list[Path] = []
    for block in blocks:
        resolved = _resolve_within_worktree(worktree_root, block.file_path)
        if resolved is None:
            return ApplyResult(
                applied=False,
                reason=(
                    f"path escape rejected: {block.file_path!r} resolves "
                    f"outside worktree"
                ),
            )
        resolved_paths.append(resolved)

    # Snapshot original bytes of every distinct target file. Using bytes
    # (not text) guarantees byte-identical rollback regardless of newline
    # or encoding quirks. Missing files snapshot as ``None`` so rollback
    # can delete files this call would have created (none today, but the
    # bookkeeping is cheap and future-proof).
    snapshots: dict[Path, Optional[bytes]] = {}
    for path in resolved_paths:
        if path in snapshots:
            continue
        try:
            snapshots[path] = path.read_bytes() if path.exists() else None
        except OSError as exc:
            raise PatchError(f"failed to snapshot {path}: {exc}") from exc

    # Apply blocks sequentially. On any failure roll back and report.
    for block, path in zip(blocks, resolved_paths):
        if not path.exists():
            _rollback(snapshots)
            return ApplyResult(
                applied=False,
                reason=f"target file not found: {block.file_path}",
            )

        try:
            current = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            try:
                _rollback(snapshots)
            except PatchError:
                logger.exception("rollback also failed after read error")
            raise PatchError(f"failed to read {path}: {exc}") from exc

        occurrences = current.count(block.search)
        if occurrences == 0:
            _rollback(snapshots)
            return ApplyResult(
                applied=False,
                reason=f"no match for block targeting {block.file_path}",
            )
        if occurrences > 1:
            _rollback(snapshots)
            return ApplyResult(
                applied=False,
                reason=(
                    f"non-unique match ({occurrences} occurrences) for block "
                    f"targeting {block.file_path}"
                ),
            )

        updated = current.replace(block.search, block.replace, 1)
        try:
            _atomic_write_text(path, updated)
        except OSError as exc:
            try:
                _rollback(snapshots)
            except PatchError:
                logger.exception("rollback also failed after write error")
            raise PatchError(f"failed to write {path}: {exc}") from exc

    return ApplyResult(applied=True, reason=None)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_within_worktree(worktree_root: Path, file_path: str) -> Optional[Path]:
    """Resolve ``file_path`` against ``worktree_root`` if it stays inside.

    Returns the resolved absolute path on success, or ``None`` if the path
    is absolute, escapes via ``..``, or cannot be safely resolved.
    """
    if not file_path:
        return None
    candidate = Path(file_path)
    if candidate.is_absolute():
        return None
    # ``Path.resolve`` collapses ``..`` segments so we can compare against
    # the worktree root. Use ``strict=False`` because the file may not
    # exist yet — existence is checked by the caller.
    try:
        resolved = (worktree_root / candidate).resolve(strict=False)
    except OSError:
        return None
    try:
        resolved.relative_to(worktree_root)
    except ValueError:
        return None
    return resolved


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically via tmp file + rename.

    The tmp file is created in the same directory as ``path`` so the
    final rename is on the same filesystem and therefore atomic on POSIX
    and best-effort atomic on Windows (``os.replace``).
    """
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".patch.", dir=str(directory))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        os.replace(tmp_path, path)
    except OSError:
        # Best-effort cleanup of the tmp file on failure.
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Byte-oriented counterpart to :func:`_atomic_write_text` for rollback."""
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".patch.", dir=str(directory))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp_path, path)
    except OSError:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise


def _rollback(snapshots: dict[Path, Optional[bytes]]) -> None:
    """Restore every snapshot to its original byte content.

    Files that did not exist before the apply are removed. Any IO failure
    during rollback is collected and re-raised as :class:`PatchError`
    after attempting to restore as much state as possible.
    """
    errors: list[str] = []
    for path, data in snapshots.items():
        try:
            if data is None:
                if path.exists():
                    path.unlink()
            else:
                _atomic_write_bytes(path, data)
        except OSError as exc:
            errors.append(f"{path}: {exc}")
    if errors:
        raise PatchError("rollback failed for: " + "; ".join(errors))
