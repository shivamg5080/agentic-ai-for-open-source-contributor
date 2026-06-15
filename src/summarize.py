"""Phase 4 — PR title + body synthesis.

This module turns the winning unified diff plus the originating issue into a
:class:`PRSummary` whose title is a single non-empty line and whose body
references the issue number — the postconditions documented in design.md
(Algorithmic Pseudocode/Phase 4) and Correctness Property 9.

The flow is:

1. Fetch (or load from cache) up to ``k`` recently-merged PR bodies for the
   target repo. Best-effort — empty list on any failure, so a stylistic miss
   never sinks the run.
2. Derive a small *style spec* from that corpus — title prefixes, common
   section headings, presence of ``Closes #N`` references, tone, typical
   length. Pure function; no LLM call.
3. Build a strict prompt that includes the style hint, 1-2 truncated body
   excerpts, the issue title/body (truncated), and the winning diff
   (truncated). Instruct the model to emit a deterministic ``TITLE: ... BODY:
   ...`` block so :func:`parse_title_body` can recover both halves
   unambiguously.
4. Validate the postconditions. Single-line title is enforced strictly
   (raises :class:`SummarizeError` on violation). The issue-number reference
   is enforced softly: if the model omits ``#<N>``, we append a
   ``Closes #<N>`` line rather than fail the run, since stylistic guidance
   sometimes drops issue refs.

All network I/O is gated by ``offline``; the recent-PR cache is written
atomically (tmp + rename) so a crashed run never leaves a half-written file.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

from src.config import Config
from src.issue import Issue
from src.llm import LLMError, complete


__all__ = [
    "PRSummary",
    "SummarizeError",
    "fetch_recent_merged_prs",
    "derive_style",
    "summarize",
    "parse_title_body",
]


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class SummarizeError(RuntimeError):
    """Raised when the LLM output cannot be parsed into a valid PR summary."""


@dataclass(frozen=True)
class PRSummary:
    """Final PR title + body returned to the orchestrator."""

    title: str
    body: str


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

_GITHUB_API = "https://api.github.com"
_REQUEST_TIMEOUT_S = 20

# Bound how many recent PRs we ever pull, regardless of caller request, so a
# typo in cfg cannot trigger a giant search-API page.
_MAX_RECENT_PRS = 10

# Truncation budgets for the prompt. The diff dominates, so we give it the
# largest budget; the issue body and PR excerpts are short stylistic context.
_DIFF_BUDGET_CHARS = 6000
_ISSUE_BUDGET_CHARS = 2000
_PR_EXCERPT_BUDGET_CHARS = 600
_MAX_PR_EXCERPTS = 2

# Conventional-commit prefix recognizer for title style derivation.
_CC_PREFIX_RE = re.compile(
    r"^(fix|feat|chore|docs|refactor|perf|test|build|ci|style|revert)(\([^)]+\))?!?:",
    re.IGNORECASE,
)

# Heading recognizer: lines starting with one or more `#` (markdown).
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)

# Closes/Fixes/Resolves #N reference recognizer (case-insensitive).
_CLOSES_REF_RE = re.compile(
    r"\b(close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+#\d+\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Recent PR fetching
# ---------------------------------------------------------------------------


def _auth_headers() -> dict:
    """GitHub REST headers, attaching ``GITHUB_TOKEN`` when present."""
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "agentless-go-contributor",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
    return headers


def _cache_file(cache_dir: Path, repo: str) -> Path:
    """On-disk cache path for a repo's recent-PR corpus."""
    return Path(cache_dir) / f"{repo.replace('/', '__')}-recent-prs.json"


def _atomic_write_json(target: Path, payload: list) -> None:
    """Atomically write ``payload`` (a JSON-serializable list) to ``target``."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(target.parent),
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    )
    tmp_path = Path(fd.name)
    try:
        with fd:
            json.dump(payload, fd, indent=2, ensure_ascii=False)
        os.replace(tmp_path, target)
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise


def fetch_recent_merged_prs(
    repo: str,
    k: int,
    *,
    offline: bool,
    cache_dir: Path = Path(".workdir") / "prs",
) -> list[str]:
    """Return up to ``k`` recently-merged PR bodies for ``repo``.

    Each entry is a single string of the form ``"# <title>\\n\\n<body>"`` so
    callers can splice it directly into a prompt as stylistic context.

    Online behavior:
      Issue a single Search-API call, persist the resulting list of strings to
      ``cache_dir / "<owner>__<name>-recent-prs.json"`` atomically, and return
      it.

    Offline behavior:
      Load the JSON list from the same path. If the file is missing or
      unreadable, return ``[]``. Stylistic guidance is best-effort and must
      never fail the whole run.

    Network failures online are also degraded to ``[]`` (logged via stderr) so
    a flaky network does not block a successful patch from getting summarized.
    """
    if not isinstance(repo, str) or "/" not in repo:
        return []
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        return []

    bounded_k = min(k, _MAX_RECENT_PRS)
    cache_path = _cache_file(Path(cache_dir), repo)

    if offline:
        return _load_cached_prs(cache_path, bounded_k)

    # Online: GitHub search for merged PRs in this repo, sorted by updated.
    url = (
        f"{_GITHUB_API}/search/issues"
        f"?q=repo:{repo}+is:pr+is:merged"
        f"&sort=updated&order=desc&per_page={bounded_k}"
    )
    try:
        resp = requests.get(url, headers=_auth_headers(), timeout=_REQUEST_TIMEOUT_S)
    except requests.RequestException as exc:
        _warn(f"recent-PR fetch failed (network): {exc}; falling back to cache/empty")
        return _load_cached_prs(cache_path, bounded_k)

    if resp.status_code >= 400:
        _warn(
            f"recent-PR fetch failed (HTTP {resp.status_code}); "
            f"falling back to cache/empty"
        )
        return _load_cached_prs(cache_path, bounded_k)

    try:
        data = resp.json()
    except ValueError:
        _warn("recent-PR response was not JSON; falling back to cache/empty")
        return _load_cached_prs(cache_path, bounded_k)

    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return _load_cached_prs(cache_path, bounded_k)

    bodies: list[str] = []
    for item in items[:bounded_k]:
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        body = item.get("body")
        title_str = title if isinstance(title, str) else ""
        body_str = body if isinstance(body, str) else ""
        if not title_str and not body_str:
            continue
        bodies.append(f"# {title_str}\n\n{body_str}".rstrip())

    # Persist for reproducible offline replays.
    try:
        _atomic_write_json(cache_path, bodies)
    except OSError as exc:
        _warn(f"failed to write recent-PR cache at {cache_path}: {exc}")

    return bodies


def _load_cached_prs(cache_path: Path, k: int) -> list[str]:
    """Read the cached PR list, returning at most ``k`` entries.

    Best-effort: any error returns ``[]``.
    """
    if not cache_path.is_file():
        return []
    try:
        raw = cache_path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    out: list[str] = []
    for entry in data:
        if isinstance(entry, str):
            out.append(entry)
        if len(out) >= k:
            break
    return out


def _warn(message: str) -> None:
    """Write a one-line warning to stderr; never raises."""
    try:
        import sys

        print(f"warning: {message}", file=sys.stderr)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Style derivation (pure)
# ---------------------------------------------------------------------------


def derive_style(recent_prs: list[str]) -> dict:
    """Extract a small, prompt-friendly style spec from recent PR bodies.

    Returns a dict with stable keys regardless of input — callers can rely on
    every field being present:

    ``prefix``
        The most common conventional-commit prefix among recent titles
        (``"fix"``, ``"feat"``, ...), or ``None`` if no clear plurality.
    ``headings``
        Up to 5 of the most common markdown section headings (e.g.
        ``"Summary"``, ``"Test Plan"``).
    ``tone``
        ``"neutral"`` for the empty corpus; ``"terse"`` if the median body is
        short (< 200 chars); ``"detailed"`` if long (> 800 chars); otherwise
        ``"balanced"``.
    ``has_closes_ref``
        True iff at least half of the bodies reference an issue with
        ``Closes/Fixes/Resolves #N``.
    ``avg_body_chars``
        Average body length across the corpus (0 for the empty corpus).

    Pure function: no I/O, no LLM call. Empty input yields a sensible default.
    """
    default = {
        "prefix": None,
        "headings": [],
        "tone": "neutral",
        "has_closes_ref": False,
        "avg_body_chars": 0,
    }
    if not recent_prs:
        return default

    titles: list[str] = []
    bodies: list[str] = []
    for entry in recent_prs:
        if not isinstance(entry, str):
            continue
        title, body = _split_title_body_block(entry)
        titles.append(title)
        bodies.append(body)

    # Most-common conventional-commit prefix among titles.
    prefix_counts: dict[str, int] = {}
    for t in titles:
        m = _CC_PREFIX_RE.match(t.strip())
        if m:
            prefix_counts[m.group(1).lower()] = prefix_counts.get(m.group(1).lower(), 0) + 1
    prefix: Optional[str] = None
    if prefix_counts:
        # Pick the highest count, but only if it has a clear plurality
        # (>= ceil(n/3)) — otherwise leave None to avoid inventing a style.
        sorted_prefixes = sorted(prefix_counts.items(), key=lambda kv: -kv[1])
        top_name, top_count = sorted_prefixes[0]
        if top_count * 3 >= max(1, len(titles)):
            prefix = top_name

    # Most-common headings across bodies (case-insensitive label match).
    heading_counts: dict[str, int] = {}
    canonical: dict[str, str] = {}  # lower -> first-seen original casing
    for b in bodies:
        for match in _HEADING_RE.finditer(b):
            label = match.group(2).strip()
            key = label.lower()
            heading_counts[key] = heading_counts.get(key, 0) + 1
            canonical.setdefault(key, label)
    headings = [
        canonical[key]
        for key, _count in sorted(heading_counts.items(), key=lambda kv: -kv[1])[:5]
    ]

    # Tone classification by median body length.
    lengths = sorted(len(b) for b in bodies)
    median = lengths[len(lengths) // 2] if lengths else 0
    if median == 0:
        tone = "neutral"
    elif median < 200:
        tone = "terse"
    elif median > 800:
        tone = "detailed"
    else:
        tone = "balanced"

    # Closes/Fixes/Resolves prevalence.
    closes_hits = sum(1 for b in bodies if _CLOSES_REF_RE.search(b))
    has_closes_ref = closes_hits * 2 >= max(1, len(bodies))  # at least half

    avg_body = (sum(len(b) for b in bodies) // len(bodies)) if bodies else 0

    return {
        "prefix": prefix,
        "headings": headings,
        "tone": tone,
        "has_closes_ref": has_closes_ref,
        "avg_body_chars": avg_body,
    }


def _split_title_body_block(entry: str) -> tuple[str, str]:
    """Split a ``"# <title>\\n\\n<body>"`` corpus entry into (title, body).

    Robust to entries that lack a leading heading: the first non-empty line
    becomes the title and the rest the body.
    """
    text = entry.lstrip("\ufeff").strip("\n")
    lines = text.splitlines()
    if not lines:
        return "", ""
    first = lines[0].strip()
    if first.startswith("#"):
        first = first.lstrip("#").strip()
    body = "\n".join(lines[1:]).strip("\n")
    return first, body


# ---------------------------------------------------------------------------
# Prompt building + parse
# ---------------------------------------------------------------------------


def _truncate(text: str, limit: int) -> str:
    """Truncate ``text`` to ``limit`` chars, marking elision for the LLM."""
    if not isinstance(text, str):
        return ""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n... [truncated]"


def _build_prompt(
    *,
    winning_diff: str,
    issue: Issue,
    recent_prs: list[str],
    style: dict,
) -> str:
    """Assemble the strict-format prompt sent to the model."""
    style_lines = [
        f"- title prefix: {style.get('prefix') or '(none)'}",
        f"- common section headings: {', '.join(style.get('headings') or []) or '(none)'}",
        f"- tone: {style.get('tone', 'neutral')}",
        f"- references issues with Closes/Fixes #N: {bool(style.get('has_closes_ref'))}",
        f"- typical body length (chars): {style.get('avg_body_chars', 0)}",
    ]

    excerpts: list[str] = []
    for pr in recent_prs[:_MAX_PR_EXCERPTS]:
        excerpts.append(_truncate(pr, _PR_EXCERPT_BUDGET_CHARS))
    excerpts_section = (
        "\n\n---\n\n".join(excerpts) if excerpts else "(no recent PR excerpts available)"
    )

    issue_block = (
        f"Issue #{issue.number}: {issue.title}\n\n"
        f"{_truncate(issue.body or '', _ISSUE_BUDGET_CHARS)}"
    )

    diff_block = _truncate(winning_diff or "", _DIFF_BUDGET_CHARS)

    return (
        "You are drafting a pull-request title and body for a Go repository.\n"
        "Match the conventions used by recent merged PRs in this repo.\n\n"
        "STYLE GUIDANCE:\n"
        + "\n".join(style_lines)
        + "\n\nRECENT PR EXCERPTS (style reference only — do not copy verbatim):\n"
        + excerpts_section
        + "\n\nISSUE BEING ADDRESSED:\n"
        + issue_block
        + "\n\nWINNING DIFF (the change you are summarizing):\n"
        + "```diff\n"
        + diff_block
        + "\n```\n\n"
        "OUTPUT FORMAT (strict — no preamble, no code fences around the whole "
        "block):\n"
        "TITLE: <one-line title, no newline characters>\n"
        "BODY:\n"
        f"<multi-line body that explains the change and references issue #{issue.number} "
        "via 'Closes #', 'Fixes #', or 'Resolves #'>\n\n"
        "Constraints:\n"
        "- The TITLE line must be a single non-empty line.\n"
        f"- The BODY must mention the issue number (#{issue.number}).\n"
        "- Keep the body focused on what changed and why; do not paste the "
        "diff back.\n"
    )


def parse_title_body(text: str) -> tuple[str, str]:
    """Recover (title, body) from the LLM response.

    Strict path: a ``TITLE:`` line followed by a ``BODY:`` line and the
    subsequent text. Surrounding whitespace and a single leading code fence
    are tolerated.

    Fallback path: when ``TITLE:`` is absent, the first non-empty line becomes
    the title and the remainder becomes the body. This covers models that
    return clean Markdown PR drafts despite the strict prompt.

    Raises:
        SummarizeError: when the title is empty or contains an embedded
        newline (the postcondition we cannot recover from).
    """
    if not isinstance(text, str):
        raise SummarizeError("LLM response must be a string")

    cleaned = text.strip()
    # Strip a single wrapping ``` fence if present.
    if cleaned.startswith("```"):
        # Drop the opening fence line.
        first_nl = cleaned.find("\n")
        if first_nl != -1:
            cleaned = cleaned[first_nl + 1 :]
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[: -len("```")].rstrip()

    title, body = _parse_strict(cleaned)
    if title is None:
        title, body = _parse_fallback(cleaned)

    if title is None or not title.strip():
        raise SummarizeError("could not extract a non-empty PR title from LLM response")
    title = title.strip()
    if "\n" in title or "\r" in title:
        raise SummarizeError(
            "PR title must be a single line; got embedded newline characters"
        )
    body = (body or "").strip("\n").rstrip()
    return title, body


def _parse_strict(text: str) -> tuple[Optional[str], str]:
    """Try the ``TITLE: ... BODY: ...`` format. Returns (None, '') if absent."""
    # Locate the TITLE: line. Note the post-colon whitespace class is
    # [ \t] (horizontal only) — ``\s*`` would devour the trailing newline and
    # capture the next line as the title.
    title_match = re.search(r"(?im)^[ \t]*TITLE[ \t]*:[ \t]*(.*?)[ \t]*$", text)
    if not title_match:
        return None, ""
    title = title_match.group(1).strip()

    # Body starts after the BODY: marker, if present.
    body_match = re.search(
        r"(?ims)^[ \t]*BODY[ \t]*:[ \t]*\n?(.*)\Z",
        text[title_match.end():],
    )
    if body_match:
        body = body_match.group(1)
    else:
        # No explicit BODY: marker — take the remainder after TITLE.
        body = text[title_match.end():]
    return title, body


def _parse_fallback(text: str) -> tuple[Optional[str], str]:
    """First non-empty line becomes the title; rest becomes the body."""
    lines = text.splitlines()
    title: Optional[str] = None
    body_start_idx = 0
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped:
            # Strip a leading markdown heading marker for the title.
            if stripped.startswith("#"):
                stripped = stripped.lstrip("#").strip()
            title = stripped
            body_start_idx = idx + 1
            break
    body = "\n".join(lines[body_start_idx:]).strip("\n")
    return title, body


# ---------------------------------------------------------------------------
# Top-level summarize()
# ---------------------------------------------------------------------------


def summarize(
    winning_diff: str,
    issue: Issue,
    recent_prs: list[str],
    cfg: Config,
) -> PRSummary:
    """Generate a :class:`PRSummary` for ``winning_diff``.

    Preconditions:
      ``winning_diff`` is a non-empty unified diff. ``recent_prs`` may be
      empty.

    Postconditions (Property 9):
      The returned ``title`` is a single non-empty line and the ``body``
      contains a textual reference to ``issue.number`` (``#<N>``).

    The body postcondition is enforced softly: if the LLM omits the issue
    reference we append a ``Closes #<N>`` line rather than fail the run, since
    style guidance occasionally drops issue refs and a usable summary is
    strictly better than no summary.
    """
    if not isinstance(winning_diff, str) or not winning_diff.strip():
        raise SummarizeError("winning_diff must be a non-empty unified diff")
    if not isinstance(issue, Issue):
        raise SummarizeError("issue must be an Issue instance")

    style = derive_style(recent_prs or [])
    prompt = _build_prompt(
        winning_diff=winning_diff,
        issue=issue,
        recent_prs=recent_prs or [],
        style=style,
    )

    try:
        responses = complete(prompt, cfg=cfg.llm, n=1)
    except LLMError as exc:
        raise SummarizeError(f"LLM call failed during summarize: {exc}") from exc
    if not responses:
        raise SummarizeError("LLM returned no responses for summarize prompt")

    title, body = parse_title_body(responses[0].text)

    # Soft enforcement of the issue-number reference.
    if f"#{issue.number}" not in body:
        suffix = f"\n\nCloses #{issue.number}"
        body = (body + suffix).strip("\n")

    return PRSummary(title=title, body=body)
