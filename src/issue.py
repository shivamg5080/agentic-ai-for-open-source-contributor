"""Issue ingestion: fetch a GitHub issue or load it from a local JSON cache.

This module is the entry point for Phase 0 of the pipeline. It produces a
validated :class:`Issue` (and optional :class:`MergedPRRef`) from one of two
sources:

* **Online** — a single GET against the GitHub REST issues endpoint, with the
  raw JSON persisted to a disk cache so subsequent offline runs are
  reproducible. A best-effort timeline scan attempts to discover the merged
  PR that closes the issue (used by the eval harness as ground truth).
* **Offline** — a previously cached JSON payload is loaded from disk.

The module deliberately depends only on :mod:`requests`. It treats every byte
returned by GitHub as untrusted text: nothing is interpreted, executed, or
formatted into shell commands. Validation is performed up front and raises
:class:`IssueError` with an actionable message on any structural problem so
the pipeline can fail fast.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import requests


# Regex for a GitHub `owner/name` slug. GitHub itself allows the chars below
# in repo and owner names; we keep the same character class for both halves.
_REPO_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")

# GitHub REST API base. Pinned here so tests can monkey-patch the module
# attribute if they ever need to.
_GITHUB_API = "https://api.github.com"

# Default network timeout (seconds). Kept conservative so a hung connection
# fails the run quickly with an actionable message.
_REQUEST_TIMEOUT_S = 20


class IssueError(ValueError):
    """Raised when an issue cannot be fetched, loaded, or validated.

    Inherits from :class:`ValueError` so callers that already handle bad
    user input uniformly continue to work.
    """


@dataclass(frozen=True)
class MergedPRRef:
    """Reference to the merged pull request that closed an issue.

    Used by the eval harness as ground truth. Every field is best-effort:
    when GitHub does not expose enough information to populate a field it
    is left as ``None`` rather than guessed.
    """

    number: int
    merge_commit: Optional[str] = None
    base_commit: Optional[str] = None


@dataclass(frozen=True)
class Issue:
    """Validated GitHub issue payload consumed by later phases."""

    repo: str
    number: int
    title: str
    body: str
    labels: list[str] = field(default_factory=list)
    base_commit: Optional[str] = None
    merged_pr: Optional[MergedPRRef] = None


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_repo(repo: str, allowlist: Optional[list[str]]) -> None:
    """Validate ``repo`` matches ``owner/name`` and (optionally) the allowlist.

    Raises :class:`IssueError` with an actionable message on failure.
    """
    if not isinstance(repo, str) or not repo:
        raise IssueError("repo must be a non-empty 'owner/name' string")
    if not _REPO_RE.match(repo):
        raise IssueError(
            f"repo {repo!r} is not a valid 'owner/name' slug "
            "(allowed chars: A-Z, a-z, 0-9, '.', '_', '-')"
        )
    if allowlist is not None and repo not in allowlist:
        raise IssueError(
            f"repo {repo!r} is not in the approved allowlist "
            f"({', '.join(allowlist) or '<empty>'})"
        )


def _validate_issue(issue: Issue) -> None:
    """Validate a fully-populated :class:`Issue`.

    Mirrors the Data Models / Issue rules in design.md. Raises
    :class:`IssueError` on any violation.
    """
    if not isinstance(issue.number, int) or isinstance(issue.number, bool) or issue.number <= 0:
        raise IssueError(f"issue number must be a positive integer, got {issue.number!r}")
    if not isinstance(issue.title, str) or not issue.title.strip():
        raise IssueError("issue title must be a non-empty string")
    if not isinstance(issue.body, str):
        raise IssueError("issue body must be a string (may be empty)")
    if not isinstance(issue.labels, list) or not all(isinstance(lbl, str) for lbl in issue.labels):
        raise IssueError("issue labels must be a list of strings")
    if issue.base_commit is not None and not isinstance(issue.base_commit, str):
        raise IssueError("issue base_commit must be a string or None")


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def _extract_labels(raw_labels: Any) -> list[str]:
    """Best-effort extraction of label name strings from GitHub's mixed format.

    GitHub returns labels as either dicts (``{"name": "..."}``) or plain
    strings. Anything else is silently dropped — labels are never load-bearing
    for correctness, only context for prompting.
    """
    if not isinstance(raw_labels, list):
        return []
    out: list[str] = []
    for entry in raw_labels:
        if isinstance(entry, str):
            out.append(entry)
        elif isinstance(entry, dict):
            name = entry.get("name")
            if isinstance(name, str) and name:
                out.append(name)
    return out


def _build_issue_from_dict(repo: str, data: dict, *, merged_pr: Optional[MergedPRRef] = None) -> Issue:
    """Construct an :class:`Issue` from a parsed GitHub-style JSON payload.

    Accepts both the raw GitHub REST shape and our cached/offline shape (which
    is just the raw shape persisted verbatim, optionally with ``base_commit``
    and ``merged_pr`` keys added).
    """
    if not isinstance(data, dict):
        raise IssueError("issue JSON must be a JSON object at the top level")

    number = data.get("number")
    title = data.get("title", "")
    body = data.get("body") or ""  # GitHub returns null for empty bodies
    labels = _extract_labels(data.get("labels"))

    # base_commit / merged_pr may be stored alongside the raw payload in
    # offline JSON; prefer explicit args, then JSON, then None.
    base_commit = data.get("base_commit")
    if merged_pr is None:
        raw_merged = data.get("merged_pr")
        if isinstance(raw_merged, dict) and isinstance(raw_merged.get("number"), int):
            merged_pr = MergedPRRef(
                number=raw_merged["number"],
                merge_commit=raw_merged.get("merge_commit"),
                base_commit=raw_merged.get("base_commit"),
            )

    return Issue(
        repo=repo,
        number=number if isinstance(number, int) else -1,  # surfaced by validation
        title=title if isinstance(title, str) else "",
        body=body if isinstance(body, str) else "",
        labels=labels,
        base_commit=base_commit if isinstance(base_commit, str) else None,
        merged_pr=merged_pr,
    )


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------


def _cache_path(cache_dir: Path, repo: str, number: int) -> Path:
    """Compute the on-disk cache path for ``(repo, number)``.

    The ``__`` separator avoids creating extra directory levels for the
    owner segment while remaining unambiguous.
    """
    return Path(cache_dir) / "issues" / f"{repo.replace('/', '__')}-{number}.json"


def _atomic_write_json(target: Path, payload: dict) -> None:
    """Write ``payload`` to ``target`` atomically (tmp file + rename)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    # delete=False so we can rename after closing on Windows.
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
            json.dump(payload, fd, indent=2, ensure_ascii=False, sort_keys=True)
        os.replace(tmp_path, target)
    except Exception:
        # Best-effort cleanup; swallow secondary errors so we report the original.
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------


def load_issue_json(path: Path) -> Issue:
    """Load an issue from a cached/offline JSON file.

    The file must contain a top-level JSON object that includes at minimum
    ``repo`` (or be paired with one stored in the file) and the standard
    GitHub issue fields (``number``, ``title``, ``body``, ``labels``).
    """
    path = Path(path)
    if not path.is_file():
        raise IssueError(
            f"offline issue JSON not found at {path}. "
            "Run online once to populate the cache, or pass --issue-json with an explicit file."
        )
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except OSError as exc:
        raise IssueError(f"failed to read issue JSON at {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise IssueError(f"issue JSON at {path} is not valid JSON: {exc.msg} (line {exc.lineno})") from exc

    if not isinstance(data, dict):
        raise IssueError(f"issue JSON at {path} must contain a JSON object at the top level")

    repo = data.get("repo")
    if not isinstance(repo, str) or not repo:
        raise IssueError(
            f"issue JSON at {path} must include a top-level string 'repo' field"
        )
    # No allowlist enforcement here: the caller (config layer) decides whether
    # to constrain repos. Slug shape is still validated.
    _validate_repo(repo, allowlist=None)

    issue = _build_issue_from_dict(repo, data)
    _validate_issue(issue)
    return issue


# ---------------------------------------------------------------------------
# Online fetch
# ---------------------------------------------------------------------------


def _auth_headers() -> dict:
    """Build request headers, attaching a token if ``GITHUB_TOKEN`` is set."""
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "agentless-go-contributor",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
    return headers


def _try_find_merged_pr(repo: str, number: int) -> Optional[MergedPRRef]:
    """Best-effort: find the PR (if any) that closed this issue.

    Walks the issue timeline looking for a ``closed`` event whose commit is
    populated, or a ``cross-referenced`` / ``referenced`` event linking to a
    merged PR. Any failure returns ``None`` rather than aborting the run —
    this metadata is only used by the eval harness.
    """
    url = f"{_GITHUB_API}/repos/{repo}/issues/{number}/timeline"
    headers = _auth_headers()
    # Mockingbird preview kept for compatibility with older self-hosted GHES.
    headers["Accept"] = (
        "application/vnd.github.mockingbird-preview+json, application/vnd.github+json"
    )
    try:
        resp = requests.get(url, headers=headers, timeout=_REQUEST_TIMEOUT_S)
    except requests.RequestException:
        return None
    if resp.status_code != 200:
        return None
    try:
        events = resp.json()
    except ValueError:
        return None
    if not isinstance(events, list):
        return None

    merge_commit: Optional[str] = None
    pr_number: Optional[int] = None
    for ev in events:
        if not isinstance(ev, dict):
            continue
        etype = ev.get("event")
        if etype == "closed":
            commit = ev.get("commit_id")
            if isinstance(commit, str) and commit:
                merge_commit = merge_commit or commit
        elif etype in ("cross-referenced", "referenced"):
            source = ev.get("source") if isinstance(ev.get("source"), dict) else None
            issue_obj = source.get("issue") if isinstance(source, dict) else None
            if isinstance(issue_obj, dict):
                pull = issue_obj.get("pull_request")
                num = issue_obj.get("number")
                if isinstance(pull, dict) and isinstance(num, int) and pr_number is None:
                    pr_number = num

    if pr_number is None and merge_commit is None:
        return None
    # We may know a merge_commit without the PR number, or vice-versa; that
    # is still useful ground truth for the eval harness.
    return MergedPRRef(number=pr_number if pr_number is not None else -1, merge_commit=merge_commit)


def fetch_issue(
    repo: str,
    number: int,
    *,
    offline: bool,
    cache_dir: Path,
    allowlist: Optional[list[str]] = None,
) -> Issue:
    """Fetch an issue from GitHub, or load it from disk in offline mode.

    Args:
        repo: ``owner/name`` slug. Validated against ``allowlist`` if provided.
        number: Positive issue number.
        offline: When True, only the on-disk cache is consulted.
        cache_dir: Root of the workspace cache. Issues are stored under
            ``cache_dir/issues/<owner>__<name>-<number>.json``.
        allowlist: Optional approved-repo allowlist. Repos outside this list
            are rejected before any network call.

    Raises:
        IssueError: on validation failure, missing offline cache, 404, or any
            network error. The message is actionable and tells the user how
            to recover (e.g. enable offline mode).
    """
    _validate_repo(repo, allowlist)
    if not isinstance(number, int) or isinstance(number, bool) or number <= 0:
        raise IssueError(f"issue number must be a positive integer, got {number!r}")

    cache_dir = Path(cache_dir)
    cache_file = _cache_path(cache_dir, repo, number)

    if offline:
        if not cache_file.is_file():
            raise IssueError(
                f"offline mode requested but no cached issue at {cache_file}. "
                "Run once with --offline=false (or unset offline in config.yaml) "
                "to populate the cache, or supply the JSON manually."
            )
        return load_issue_json(cache_file)

    # Online path.
    url = f"{_GITHUB_API}/repos/{repo}/issues/{number}"
    try:
        resp = requests.get(url, headers=_auth_headers(), timeout=_REQUEST_TIMEOUT_S)
    except requests.RequestException as exc:
        raise IssueError(
            f"network error fetching {url}: {exc}. "
            "Re-run with offline=true if you have a cached copy of this issue."
        ) from exc

    if resp.status_code == 404:
        raise IssueError(
            f"issue {repo}#{number} not found (HTTP 404). "
            "Verify the repo slug and issue number, or supply a local JSON via offline mode."
        )
    if resp.status_code == 403 and "rate limit" in resp.text.lower():
        raise IssueError(
            "GitHub API rate limit exceeded. Set the GITHUB_TOKEN environment "
            "variable, or re-run with offline=true using a cached issue."
        )
    if resp.status_code >= 400:
        raise IssueError(
            f"GitHub API returned HTTP {resp.status_code} for {url}: "
            f"{resp.text[:200]}"
        )

    try:
        data = resp.json()
    except ValueError as exc:
        raise IssueError(
            f"GitHub returned non-JSON response for {url}: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise IssueError(f"GitHub returned an unexpected JSON shape for {url}")

    # Refuse if GitHub says this number is actually a pull request — issues and
    # PRs share the number space and the user almost certainly meant the issue.
    if "pull_request" in data:
        raise IssueError(
            f"{repo}#{number} is a pull request, not an issue. "
            "Pass the issue number this PR closed instead."
        )

    # Best-effort timeline scan for ground-truth merged PR.
    merged_pr = _try_find_merged_pr(repo, number)

    # Persist the (lightly enriched) raw payload so offline runs are reproducible.
    payload_to_cache = dict(data)
    payload_to_cache["repo"] = repo
    if merged_pr is not None:
        payload_to_cache["merged_pr"] = {
            "number": merged_pr.number,
            "merge_commit": merged_pr.merge_commit,
            "base_commit": merged_pr.base_commit,
        }
    try:
        _atomic_write_json(cache_file, payload_to_cache)
    except OSError as exc:
        # Don't fail the whole run on a cache-write hiccup; surface as warning.
        # The issue is still returned for the in-memory pipeline.
        # (Stdlib logging would be nicer but llm.py owns the structured log
        # sink; a stderr print is sufficient at this layer.)
        import sys
        print(f"warning: failed to write issue cache at {cache_file}: {exc}", file=sys.stderr)

    issue = _build_issue_from_dict(repo, payload_to_cache, merged_pr=merged_pr)
    _validate_issue(issue)
    return issue


__all__ = [
    "Issue",
    "IssueError",
    "MergedPRRef",
    "fetch_issue",
    "load_issue_json",
]
