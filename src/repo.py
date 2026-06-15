"""Repository preparation and isolated worktrees.

Implements the contract from ``design.md`` (Components/repo.py):

* :func:`prepare_repo` clones (or reuses a cached clone) and checks out the
  pinned base commit in detached HEAD. The clone directory lives at
  ``workdir/repos/<owner>__<name>``.
* :func:`clean_worktree` materialises an isolated, fresh copy of the base
  checkout via ``git worktree add --detach`` so each candidate edits a clean
  tree without disturbing the cached clone.
* :func:`make_diff` returns the unified diff of uncommitted changes in a
  worktree (i.e. candidate edits) versus the base commit.
* :func:`cleanup_worktree` removes a worktree previously created by
  :func:`clean_worktree` (silent no-op if the path is not a worktree).

Security
--------
All git invocations use argument arrays (never ``shell=True``). Inputs are
validated against a strict allowlist before being passed to git, and clone /
worktree paths are confined to ``workdir`` to prevent path escape. The base
checkout is never mutated by this module: candidate edits land only in the
worktrees returned by :func:`clean_worktree`.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

# ``owner/name`` with a conservative GitHub-ish character set. Each side must
# start with an alphanumeric or underscore — this matches real GitHub naming
# rules and removes any chance the value parses as a CLI flag.
_REPO_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]*/[A-Za-z0-9_][A-Za-z0-9._-]*$")

# Either a hex SHA (any length 4-40) or a ref-like name with safe characters
# (letters, digits, dot, underscore, dash, forward slash). The leading char
# class explicitly excludes ``-`` so the value can never be confused with a
# command-line flag even though we always pass via argv arrays.
_BASE_COMMIT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
_HEX_SHA_RE = re.compile(r"^[0-9a-fA-F]{4,40}$")


class RepoError(RuntimeError):
    """Raised on git failures or invalid repo/commit inputs."""


@dataclass(frozen=True)
class RepoCheckout:
    repo: str
    path: Path
    base_commit: str


def prepare_repo(
    repo: str,
    base_commit: Optional[str],
    workdir: Path,
) -> RepoCheckout:
    """Clone ``repo`` into ``workdir/repos/<owner>__<name>`` if needed,
    fetch refs, and check out ``base_commit`` in detached HEAD.

    If ``base_commit`` is ``None`` the current ``HEAD`` of the cached clone is
    resolved with ``git rev-parse HEAD`` and used as the pinned base; the
    returned :class:`RepoCheckout` therefore always carries a concrete SHA so
    downstream phases get reproducible diffs.
    """
    _validate_repo(repo)
    if base_commit is not None:
        _validate_base_commit(base_commit)

    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    repos_root = (workdir / "repos").resolve()
    repos_root.mkdir(parents=True, exist_ok=True)

    safe_name = repo.replace("/", "__")
    clone_dir = (repos_root / safe_name).resolve()
    _ensure_within(clone_dir, repos_root)

    if not clone_dir.exists():
        url = f"https://github.com/{repo}.git"
        _run_git_checked(
            ["git", "clone", url, str(clone_dir)],
            cwd=repos_root,
        )
    elif not (clone_dir / ".git").exists():
        raise RepoError(
            f"clone path {clone_dir} exists but is not a git repository; "
            f"remove it or choose a different workdir."
        )

    # Refresh refs so the requested base_commit is reachable.
    _run_git_checked(
        ["git", "fetch", "--all", "--tags", "--prune"],
        cwd=clone_dir,
    )

    if base_commit is None:
        # Resolve to the cached clone's current HEAD so the rest of the run
        # has a concrete pin. This keeps reproducibility intact within a run
        # even when the caller did not supply an explicit commit.
        rc, head_sha, err = _run_git(["git", "rev-parse", "HEAD"], cwd=clone_dir)
        if rc != 0:
            raise RepoError(_fmt_git_error("git rev-parse HEAD", rc, head_sha, err))
        target = head_sha.strip()
        if not target:
            raise RepoError("git rev-parse HEAD returned empty output.")
    else:
        target = base_commit

    _run_git_checked(
        ["git", "checkout", "--detach", target],
        cwd=clone_dir,
    )

    rc, resolved, err = _run_git(["git", "rev-parse", "HEAD"], cwd=clone_dir)
    if rc != 0:
        raise RepoError(_fmt_git_error("git rev-parse HEAD", rc, resolved, err))
    resolved_sha = resolved.strip()

    # If a full 40-char SHA was requested, confirm it landed verbatim.
    if (
        _HEX_SHA_RE.match(target)
        and len(target) == 40
        and target.lower() != resolved_sha.lower()
    ):
        raise RepoError(
            f"checkout produced unexpected HEAD: requested {target}, got {resolved_sha}."
        )

    return RepoCheckout(repo=repo, path=clone_dir, base_commit=resolved_sha)


def clean_worktree(checkout: RepoCheckout) -> Path:
    """Create an isolated worktree pinned at ``checkout.base_commit``.

    Edits inside the returned path do **not** propagate back to
    ``checkout.path``: each call yields a fresh copy under
    ``workdir/worktrees/<8-hex>`` via ``git worktree add --detach``.
    """
    # workdir is the grandparent of the clone (workdir/repos/<name>).
    workdir = checkout.path.parent.parent
    worktrees_root = (workdir / "worktrees").resolve()
    worktrees_root.mkdir(parents=True, exist_ok=True)

    new_path = (worktrees_root / uuid.uuid4().hex[:8]).resolve()
    _ensure_within(new_path, worktrees_root)

    _run_git_checked(
        [
            "git",
            "worktree",
            "add",
            "--detach",
            str(new_path),
            checkout.base_commit,
        ],
        cwd=checkout.path,
    )
    return new_path


def make_diff(checkout_path: Path) -> str:
    """Return the unified diff of uncommitted changes in ``checkout_path``.

    The worktree returned by :func:`clean_worktree` is detached at the base
    commit, so ``git diff --no-color HEAD`` captures candidate edits as a
    standard patch.
    """
    rc, out, err = _run_git(
        ["git", "diff", "--no-color", "HEAD"],
        cwd=checkout_path,
    )
    if rc != 0:
        raise RepoError(_fmt_git_error("git diff --no-color HEAD", rc, out, err))
    return out


def cleanup_worktree(path: Path) -> None:
    """Remove a worktree previously created by :func:`clean_worktree`.

    Silently no-ops when ``path`` does not exist or is not a registered
    worktree, so callers can use it in best-effort cleanup paths.

    On Windows, ``git worktree remove`` occasionally fails with a permission
    error when the OS still holds open handles on freshly-written files. In
    that case we fall back to a brief retry loop and, as a last resort, a
    direct ``shutil.rmtree`` followed by ``git worktree prune`` against the
    main clone so we don't leave dangling metadata.
    """
    path = Path(path)
    if not path.exists():
        return

    main_repo = _find_main_repo(path)

    # Run from a directory *outside* the worktree so the remove can delete
    # ``path`` itself (on Windows, a process cannot delete its own cwd).
    # ``git -C <path>`` still gives git the worktree as its repo context.
    cwd = path.parent if path.parent.exists() else Path.cwd()
    rc, out, err = _run_git(
        ["git", "-C", str(path), "worktree", "remove", "--force", str(path)],
        cwd=cwd,
    )
    if rc == 0:
        return

    msg = (err + out).lower()
    benign = (
        "not a working tree",
        "is not a working tree",
        "not a valid path",
        "no such file or directory",
        "not a git repository",
    )
    if any(s in msg for s in benign):
        return

    # Fallback: hard-remove the directory with a short retry loop, then prune
    # the worktree metadata in the main clone so future git operations stay
    # consistent.
    last_err: Optional[BaseException] = None
    for delay in (0.0, 0.1, 0.3, 0.6):
        if delay:
            time.sleep(delay)
        if not path.exists():
            last_err = None
            break
        try:
            shutil.rmtree(path)
            last_err = None
            break
        except OSError as exc:
            last_err = exc
    if last_err is not None and path.exists():
        raise RepoError(
            f"could not remove worktree {path}: {last_err}"
        ) from last_err

    if main_repo is not None and (main_repo / ".git").exists():
        _run_git(["git", "worktree", "prune"], cwd=main_repo)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_repo(repo: str) -> None:
    if not isinstance(repo, str) or not _REPO_RE.match(repo):
        raise RepoError(
            f"repo must be of the form 'owner/name' with safe characters, got {repo!r}."
        )


def _validate_base_commit(base_commit: str) -> None:
    if not isinstance(base_commit, str) or not _BASE_COMMIT_RE.match(base_commit):
        raise RepoError(
            f"base_commit contains unexpected characters or is empty: {base_commit!r}."
        )


def _ensure_within(child: Path, parent: Path) -> None:
    """Reject paths that escape ``parent`` (after resolution)."""
    try:
        child.resolve().relative_to(parent.resolve())
    except ValueError as exc:
        raise RepoError(
            f"refusing to operate on {child}: path escapes {parent}."
        ) from exc


def _find_main_repo(worktree: Path) -> Optional[Path]:
    """Read the worktree's ``.git`` file to recover the main clone path.

    Linked worktrees store a single-line ``gitdir: <main>/.git/worktrees/<id>``
    pointer in the worktree's ``.git`` file. Returns ``None`` if the layout is
    not recognised, so callers can fall back to best-effort cleanup.
    """
    git_marker = worktree / ".git"
    if not git_marker.is_file():
        return None
    try:
        content = git_marker.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    prefix = "gitdir:"
    if not content.startswith(prefix):
        return None
    gitdir = Path(content[len(prefix):].strip())
    # Expected layout: <main>/.git/worktrees/<id>
    if (
        gitdir.parent.name == "worktrees"
        and gitdir.parent.parent.name == ".git"
    ):
        return gitdir.parent.parent.parent
    return None


def _run_git(args: Sequence[str], *, cwd: Path) -> Tuple[int, str, str]:
    """Run a git command via argv array. Returns ``(returncode, stdout, stderr)``."""
    if not args or args[0] != "git":
        raise RepoError(f"_run_git expects a git invocation, got {list(args)!r}.")
    proc = subprocess.run(  # noqa: S603 - argv array, no shell
        list(args),
        cwd=str(cwd),
        check=False,
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _run_git_checked(args: Sequence[str], *, cwd: Path) -> Tuple[str, str]:
    """Run a git command and raise :class:`RepoError` on non-zero exit."""
    rc, out, err = _run_git(args, cwd=cwd)
    if rc != 0:
        raise RepoError(_fmt_git_error(" ".join(args), rc, out, err))
    return out, err


def _fmt_git_error(cmd: str, rc: int, out: str, err: str) -> str:
    return (
        f"{cmd} failed (exit {rc}).\n"
        f"--- stdout ---\n{out.rstrip()}\n"
        f"--- stderr ---\n{err.rstrip()}"
    )
