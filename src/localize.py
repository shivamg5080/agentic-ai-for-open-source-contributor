"""Phase 1 — hierarchical localization.

This module is built up across two tasks:

* Task 4.1 defines the data models and :func:`build_skeleton`, a hermetic,
  regex-based outline of every ``.go`` file in a checkout. The skeleton
  feeds the LLM file-pick step without ever leaking full file bodies.
* Task 4.3 (added below) wires :func:`localize`, which uses the skeleton
  (or a ripgrep fallback when the skeleton is too large) to pick top-N
  suspicious files and narrow them to :class:`EditLocation` ranges.

Design intent
-------------
The skeleton is intentionally lo-fi: a directory tree plus a per-file list
of the package name and exported top-level declarations only. We avoid
shelling out to ``go doc`` or any Go tool so the step is fast, reproducible
across machines, and side-effect free against the cached clone.

Regex limitations
-----------------
The outline parser is a line-anchored regex pass — not a full Go parser.
That is sufficient for "is this file plausibly relevant to the issue?",
which is the only question the skeleton needs to answer, but it has known
blind spots:

* Multi-line function signatures are captured only up to the first line
  break (the design only requires a signature *line*, not the whole
  parameter list).
* String literals and comments containing the tokens ``func``/``type`` at
  column 0 will not occur in well-formed Go, so we accept the small risk
  of a false positive over the cost of full lexing.
* Block forms (``var ( ... )`` / ``const ( ... )``) are parsed by scanning
  inside the parens for indented exported identifiers; nested parens are
  not supported (Go disallows them at this position anyway).

Files that fail to parse (no ``package`` line) get an empty outline rather
than raising — the caller still wants to see them in the tree even if we
cannot summarize them.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

# llm is imported as a module (rather than ``from src.llm import complete_json``)
# so unit tests can monkey-patch ``src.llm.complete_json`` and have the
# replacement observed at call time. The import is at module load time because
# there is no circular dependency: src.llm does not import this module.
from src import llm as _llm

if TYPE_CHECKING:
    from src.config import Config
    from src.issue import Issue
    from src.repo import RepoCheckout


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public data models (frozen, hashable, immutable by convention).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FileOutline:
    """Outline of a single ``.go`` file: package + exported declarations.

    ``decls`` always holds *signature lines only* — never function bodies.
    Each entry is truncated to :data:`_DECL_MAX_CHARS` characters so a
    pathologically long signature cannot blow up the prompt budget.
    """

    path: str
    package: str
    decls: list[str]


@dataclass(frozen=True)
class RepoSkeleton:
    """A repo-wide outline used by the LLM file-pick step.

    Attributes:
        tree: Sorted relative paths of every directory and ``.go`` file
            under the repo root (directories carry a trailing ``/``).
        outlines: One :class:`FileOutline` per ``.go`` file, in the same
            order they were encountered while walking the tree.
        approx_tokens: Rough ``total_chars / 4`` estimate over the tree
            string and each outline's text. Non-negative; used only for
            the "fits in budget" decision in :func:`localize`.
    """

    tree: list[str]
    outlines: list[FileOutline]
    approx_tokens: int


@dataclass(frozen=True)
class EditLocation:
    """A narrowed-down place the LLM will be asked to edit.

    Populated by :func:`localize` (task 4.3); declared here so other
    modules can import the type from a single location.
    """

    file_path: str
    symbol: str | None
    start_line: int
    end_line: int
    context: str


@dataclass(frozen=True)
class LocalizationResult:
    """Phase 1 output handed to Phase 2 (repair).

    ``used_fallback`` records whether the ripgrep over-budget path was
    taken so eval / logs can report which branch ran.
    """

    ranked_files: list[str]
    locations: list[EditLocation]
    used_fallback: bool


# ---------------------------------------------------------------------------
# Constants / regexes
# ---------------------------------------------------------------------------

# Directory names we never descend into when building the skeleton. They
# are either VCS metadata, vendored third-party code, generated Go
# fixtures the toolchain itself ignores, or JS deps that shouldn't be in
# a Go repo but occasionally are.
_EXCLUDED_DIRS: frozenset[str] = frozenset({
    ".git",
    "vendor",
    "testdata",
    "node_modules",
})

# Cap each captured declaration line so a 4 KB single-line signature
# can't dominate the outline budget.
_DECL_MAX_CHARS: int = 200

# ``package foo`` on its own line. Go allows leading whitespace only via
# ``//go:build`` style comments, never on the package clause itself.
_RE_PACKAGE = re.compile(r"^package\s+([A-Za-z_]\w*)\s*$")

# ``func Name(...`` or ``func (recv T) Name(...``. We only need to know
# whether the name is exported (capital first letter); we capture the
# whole line for the outline.
_RE_FUNC = re.compile(
    r"^func\s+(?:\(\s*[^)]*\)\s+)?([A-Z]\w*)\s*\("
)

# ``type Name struct {``, ``type Name interface {``, ``type Name = ...``,
# ``type Name int``, etc. The trailing token list is intentionally loose;
# anything after the exported name is fine since we only need the name.
_RE_TYPE = re.compile(r"^type\s+([A-Z]\w*)\b")

# Singletons (no parens). The block form is handled separately below.
_RE_VAR_SINGLE = re.compile(r"^var\s+([A-Z]\w*)\b")
_RE_CONST_SINGLE = re.compile(r"^const\s+([A-Z]\w*)\b")

# Opens a block-form var/const declaration. Match anywhere on the line as
# long as the keyword starts the line, so trailing comments are tolerated.
_RE_VAR_BLOCK_OPEN = re.compile(r"^var\s*\(\s*(?://.*)?$")
_RE_CONST_BLOCK_OPEN = re.compile(r"^const\s*\(\s*(?://.*)?$")

# Closing paren of a block-form declaration (line-anchored).
_RE_BLOCK_CLOSE = re.compile(r"^\)\s*(?://.*)?$")

# Inside a block: leading whitespace, then an exported identifier. We
# capture the full line for the outline; the regex just gates inclusion.
_RE_BLOCK_EXPORTED = re.compile(r"^\s+([A-Z]\w*)\b")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_skeleton(checkout: "RepoCheckout") -> RepoSkeleton:
    """Walk ``checkout.path`` and return a lightweight repo skeleton.

    Preconditions:
        ``checkout.path`` exists and contains Go source files.

    Postconditions:
        * ``tree`` lists every directory and every ``.go`` file under the
          repo root (excluding the noise directories in
          :data:`_EXCLUDED_DIRS` and any hidden directory).
        * Each :class:`FileOutline` carries the file's package name plus
          its exported top-level declarations. No file bodies leak.
        * ``approx_tokens`` is a non-negative integer.

    Loop invariant:
        After processing the *i*-th encountered ``.go`` file, ``outlines``
        holds exactly *i* entries — one per file seen so far.
    """
    root = Path(checkout.path)
    if not root.is_dir():
        raise ValueError(f"checkout path does not exist or is not a dir: {root}")

    tree, go_files = _walk_repo(root)

    outlines: list[FileOutline] = []
    for rel_path in go_files:
        outlines.append(_outline_file(root, rel_path))
        # Loop invariant: outlines aligns 1:1 with the .go files processed
        # so far (in deterministic walk order).

    approx_tokens = _estimate_tokens(tree, outlines)
    return RepoSkeleton(tree=tree, outlines=outlines, approx_tokens=approx_tokens)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _walk_repo(root: Path) -> tuple[list[str], list[str]]:
    """Return ``(tree, go_files)`` — both lists hold POSIX-style relpaths.

    ``tree`` is sorted and contains every visited directory (with a
    trailing ``/``) plus every ``.go`` file. ``go_files`` contains only
    the ``.go`` file paths in the same sorted order, ready for outlining.
    """
    tree_entries: list[str] = []
    go_files: list[str] = []

    # ``os.walk`` lets us mutate ``dirs`` in-place to prune subtrees.
    for dirpath, dirs, files in os.walk(root):
        # Prune hidden and excluded directories before descending.
        dirs[:] = [
            d for d in dirs
            if not d.startswith(".") and d not in _EXCLUDED_DIRS
        ]

        rel_dir = os.path.relpath(dirpath, root).replace(os.sep, "/")
        if rel_dir == ".":
            tree_entries.append("./")
        else:
            tree_entries.append(rel_dir + "/")

        for fname in files:
            if not fname.endswith(".go"):
                continue
            rel_file = (
                fname if rel_dir == "."
                else f"{rel_dir}/{fname}"
            )
            tree_entries.append(rel_file)
            go_files.append(rel_file)

    tree_entries.sort()
    go_files.sort()
    return tree_entries, go_files


def _outline_file(root: Path, rel_path: str) -> FileOutline:
    """Parse a single ``.go`` file into a :class:`FileOutline`.

    Files that don't expose a ``package`` clause fall back to an empty
    outline (``package=""``, ``decls=[]``) and emit a debug log entry.
    Read errors are treated the same way so a single unreadable file
    doesn't sink the whole skeleton.
    """
    abs_path = root / rel_path
    try:
        text = abs_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.debug("skeleton: could not read %s: %s", rel_path, exc)
        return FileOutline(path=rel_path, package="", decls=[])

    package = ""
    decls: list[str] = []

    lines = text.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.rstrip()

        if not package:
            m = _RE_PACKAGE.match(stripped)
            if m:
                package = m.group(1)
                i += 1
                continue

        # Block forms: scan ahead and capture exported entries only.
        if _RE_VAR_BLOCK_OPEN.match(stripped) or _RE_CONST_BLOCK_OPEN.match(stripped):
            kind = "var" if stripped.startswith("var") else "const"
            i += 1
            while i < n and not _RE_BLOCK_CLOSE.match(lines[i].rstrip()):
                inner = lines[i]
                m = _RE_BLOCK_EXPORTED.match(inner)
                if m:
                    decls.append(_format_block_decl(kind, inner))
                i += 1
            # Skip the closing ')' if we found one.
            if i < n:
                i += 1
            continue

        # Single-line top-level forms.
        m = _RE_FUNC.match(stripped)
        if m:
            decls.append(_truncate(stripped))
            i += 1
            continue

        m = _RE_TYPE.match(stripped)
        if m:
            decls.append(_truncate(stripped))
            i += 1
            continue

        m = _RE_VAR_SINGLE.match(stripped) or _RE_CONST_SINGLE.match(stripped)
        if m:
            decls.append(_truncate(stripped))
            i += 1
            continue

        i += 1

    if not package:
        # No package clause means this isn't valid Go from our perspective —
        # drop the decls so callers see a clean "couldn't outline" signal
        # rather than half-parsed noise. Tree listing already records the
        # file's existence.
        logger.debug("skeleton: no package clause in %s", rel_path)
        return FileOutline(path=rel_path, package="", decls=[])

    return FileOutline(path=rel_path, package=package, decls=decls)


def _format_block_decl(kind: str, raw_line: str) -> str:
    """Normalize a block-form decl into a ``var Name ...`` style line.

    Inside ``var (`` / ``const (`` blocks the entries are indented and
    don't carry the keyword, so we re-prefix them for readability in the
    LLM prompt: ``    Foo = 1`` becomes ``var Foo = 1``.
    """
    return _truncate(f"{kind} {raw_line.strip()}")


def _truncate(s: str) -> str:
    """Cap a single decl line at :data:`_DECL_MAX_CHARS` characters."""
    if len(s) <= _DECL_MAX_CHARS:
        return s
    return s[: _DECL_MAX_CHARS - 1] + "…"


def _estimate_tokens(tree: Iterable[str], outlines: Iterable[FileOutline]) -> int:
    """Approximate token count: ``total_chars / 4``.

    Counts the tree listing plus, per outline, the path + package header
    plus each declaration line. The estimate is intentionally loose —
    it's only used to gate the ripgrep-fallback branch in :func:`localize`.
    """
    chars = 0
    for entry in tree:
        chars += len(entry) + 1  # +1 for the implicit newline
    for outline in outlines:
        chars += len(outline.path) + len(outline.package) + 4  # path + " : " + pkg
        for decl in outline.decls:
            chars += len(decl) + 1
    # Integer division yields a non-negative int for any non-negative chars.
    return max(0, chars // 4)


# ---------------------------------------------------------------------------
# Phase 1 — localize(): file-pick + edit-location narrowing
# ---------------------------------------------------------------------------
#
# Strategy (mirrors design.md/Algorithmic Pseudocode/Phase 1):
#
#   1. Build a hermetic skeleton of the checkout (already done by
#      :func:`build_skeleton`).
#   2. Decide which "candidate pool" of files the LLM may pick from:
#        * within-budget: every ``.go`` file in the skeleton;
#        * over-budget : ripgrep over symbols/error strings extracted from
#          the issue, then restrict the skeleton to those seeds. This is
#          the only branch that flips ``used_fallback=True``.
#   3. Ask the LLM to rank the top-N suspicious files from the candidate
#      pool (one strict-JSON call). Hallucinated paths are filtered out.
#   4. For each ranked file, re-extract its declarations *with line
#      numbers* (the skeleton outline only stores text), ask the LLM to
#      narrow them to specific symbols / line ranges, then attach
#      surrounding context from the source.
#
# Property 2 (Localization soundness) is enforced at construction time in
# :func:`_build_edit_location`: any narrowed range that falls outside
# ``[1, file_line_count]`` or whose target file does not exist / is not a
# ``.go`` file is dropped rather than passed downstream.
#
# All LLM calls flow through ``src.llm.complete_json`` so unit tests can
# monkey-patch a single seam without an API key.

# Threshold above which the skeleton is considered too large to send
# verbatim and we fall back to ripgrep-seeded restriction. The value is
# deliberately loose; it only gates a branch and is not a correctness
# guarantee. Roughly 30k tokens leaves comfortable headroom for the issue
# body, prompt boilerplate, and the LLM's own response on a 200k-context
# Claude model.
SKELETON_TOKEN_BUDGET: int = 30_000

# Number of source lines to include before/after the LLM's chosen range
# when filling :class:`EditLocation.context`. Five lines is enough for a
# repair model to see the surrounding declarations without ballooning the
# prompt budget.
_CONTEXT_PADDING_LINES: int = 5

# Hard ceiling on the size of the *context* slice attached to each
# :class:`EditLocation`. Even if the LLM picks an absurdly long range we
# refuse to ship more than this many lines downstream so a single edit
# location cannot dominate the repair prompt.
_CONTEXT_MAX_LINES: int = 200

# Maximum number of declarations included in the narrow-prompt for a
# single file. If a file exposes more, the first N are sent — the LLM is
# always free to pick line ranges that span outside the listed decls.
_NARROW_DECLS_LIMIT: int = 80

# Maximum number of locations the LLM may return per file. Caps prompt
# fan-out into Phase 2 even if the model gets exuberant.
_MAX_LOCATIONS_PER_FILE: int = 4

# Caps for the ripgrep fallback. We bound the number of distinct queries
# (so an issue body listing dozens of identifiers can't DoS rg) and the
# number of seed files surfaced (so the LLM file-pick prompt stays small).
_MAX_FALLBACK_QUERIES: int = 20
_MAX_FALLBACK_SEEDS: int = 80
_RIPGREP_TIMEOUT_S: float = 10.0

# Prompt templates live next to the package, not inside it.
_PROMPTS_DIR: Path = Path(__file__).resolve().parent.parent / "prompts"

# Suffix appended to every prompt to lock in a strict JSON output shape.
# Keeping this explicit (rather than embedding it in the placeholder
# prompt files) means the contract survives prompt rewrites in later
# tasks: ``complete_json`` parses + validates whatever the model emits.
_FILE_PICK_SUFFIX = (
    "\n\n---\n"
    "Output ONLY a single JSON object of the exact form:\n"
    '{{"files": ["relative/path/a.go", "relative/path/b.go", ...]}}\n'
    "Pick at most {top_n} paths. Each path MUST appear verbatim in the\n"
    "skeleton above (use forward slashes). No prose, no commentary, no\n"
    "code fences."
)

_NARROW_SUFFIX = (
    "\n\n---\n"
    "Output ONLY a single JSON object of the exact form:\n"
    '{"locations": [{"symbol": "Name or null", '
    '"start_line": <int>, "end_line": <int>}, ...]}\n'
    "Line numbers are 1-indexed and must satisfy "
    "1 <= start_line <= end_line. Each range should cover a declaration\n"
    "from the list above (a function body, a type, or a var/const block).\n"
    "Pick at most 4 locations. No prose, no commentary, no code fences."
)


# Symbol / error-string extraction regexes used by the ripgrep fallback.
# Tuned for Go-flavoured issues: CamelCase identifiers, dotted method
# references, and quoted strings (which often carry an error message).
_SYMBOL_RE = re.compile(r"\b[A-Z][A-Za-z0-9_]{2,}\b")
_DOTTED_SYMBOL_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\.[A-Z][A-Za-z0-9_]+\b")
_DQ_STRING_RE = re.compile(r'"([^"\\\n]{4,80})"')
_BT_STRING_RE = re.compile(r"`([^`\n]{4,80})`")

# Generic English / acronym tokens that match the symbol regex but carry
# no real localization signal — they would just dilute ripgrep results.
_NOISE_QUERIES: frozenset[str] = frozenset({
    "TODO", "FIXME", "HTTP", "HTTPS", "API", "URL", "URI",
    "JSON", "YAML", "XML", "CSV", "HTML", "CSS",
    "TODO:", "FIXME:", "NOTE:", "NOTE", "Bug", "Bugs",
})


def localize(
    checkout: "RepoCheckout",
    issue: "Issue",
    cfg: "Config",
) -> LocalizationResult:
    """Run Phase 1 hierarchical localization.

    Steps:
      1. Build a repo skeleton (always — it's cheap and we use it for
         outline rendering even on the fallback path).
      2. If the skeleton fits in the token budget, the LLM picks files
         from the *whole* skeleton; otherwise we ripgrep symbols and
         error strings from the issue and let the LLM pick from a
         restricted skeleton (``used_fallback=True``).
      3. For each ranked file, narrow to declarations / line ranges via
         a second LLM call, then attach surrounding context.

    Postconditions:
        * ``ranked_files`` length is ≤ ``cfg.top_n_files``; every entry
          is a ``.go`` path that exists in the candidate pool.
        * Every :class:`EditLocation` references an existing ``.go`` file
          with ``1 <= start_line <= end_line <= file_line_count`` and
          carries non-empty context (Property 2).
    """
    skeleton = build_skeleton(checkout)

    if skeleton.approx_tokens <= SKELETON_TOKEN_BUDGET:
        used_fallback = False
        candidate_pool = [o.path for o in skeleton.outlines]
        prompt_skeleton = _format_skeleton(skeleton)
    else:
        used_fallback = True
        candidate_pool, prompt_skeleton = _seed_via_ripgrep(checkout, issue, skeleton)

    ranked_files = _llm_pick_files(
        skeleton_text=prompt_skeleton,
        repo=checkout.repo,
        issue=issue,
        top_n=cfg.top_n_files,
        cfg=cfg,
        allowed_paths=candidate_pool,
    )

    locations: list[EditLocation] = []
    root = Path(checkout.path)
    for rel_path in ranked_files:
        # Loop invariant: rel_path is a member of ``ranked_files`` and
        # therefore drawn from the candidate pool — never hallucinated.
        decls_with_lines = _decls_with_lines(root, rel_path)
        narrowed = _llm_narrow_locations(
            file_path=rel_path,
            decls_with_lines=decls_with_lines,
            issue=issue,
            cfg=cfg,
        )
        for raw_loc in narrowed[:_MAX_LOCATIONS_PER_FILE]:
            loc = _build_edit_location(root, rel_path, raw_loc)
            if loc is not None:
                locations.append(loc)

    return LocalizationResult(
        ranked_files=ranked_files,
        locations=locations,
        used_fallback=used_fallback,
    )


# ---------------------------------------------------------------------------
# Skeleton formatting and restriction
# ---------------------------------------------------------------------------


def _format_skeleton(skeleton: RepoSkeleton) -> str:
    """Render a :class:`RepoSkeleton` as the text shown to the LLM.

    The format is intentionally line-oriented and verbatim-copyable so
    the LLM can echo paths back at us with zero ambiguity:

        TREE:
        ./
        api/
        api/foo.go

        OUTLINES:
        [api/foo.go] package foo
          func Bar(ctx Context) error
          type Config struct {

    Files whose outline is empty (no package clause / unreadable) are
    omitted from the OUTLINES section but still listed in TREE so the
    LLM knows they exist.
    """
    parts: list[str] = ["TREE:"]
    parts.extend(skeleton.tree)
    parts.append("")
    parts.append("OUTLINES:")
    for outline in skeleton.outlines:
        if not outline.package and not outline.decls:
            continue
        package = outline.package or "<unknown>"
        parts.append(f"[{outline.path}] package {package}")
        for decl in outline.decls:
            parts.append(f"  {decl}")
    return "\n".join(parts)


def _restrict_skeleton(
    skeleton: RepoSkeleton, allowed_paths: set[str]
) -> RepoSkeleton:
    """Return a new :class:`RepoSkeleton` whose outlines are restricted to
    ``allowed_paths``. Directory entries in the tree are kept so the LLM
    still has structural context; non-allowed file entries are dropped.
    ``approx_tokens`` is not recomputed — it's already a loose estimate.
    """
    new_outlines = [o for o in skeleton.outlines if o.path in allowed_paths]
    new_tree = [
        entry for entry in skeleton.tree
        if entry.endswith("/") or entry in allowed_paths
    ]
    return RepoSkeleton(
        tree=new_tree, outlines=new_outlines, approx_tokens=skeleton.approx_tokens
    )


# ---------------------------------------------------------------------------
# Ripgrep fallback (over-budget path)
# ---------------------------------------------------------------------------


def _seed_via_ripgrep(
    checkout: "RepoCheckout", issue: "Issue", skeleton: RepoSkeleton,
) -> tuple[list[str], str]:
    """Run the over-budget branch and return ``(candidate_pool, prompt_text)``.

    Extracts query strings from the issue, runs ripgrep (or a Python
    fallback when ``rg`` is missing), and restricts the skeleton to the
    matched files. If extraction or grep produce nothing we degrade to
    using the full skeleton — a slightly fat prompt is still better than
    no localization.
    """
    queries = _extract_symbols_and_errors(issue)
    seeds = _ripgrep_seeds(Path(checkout.path), queries) if queries else []

    if not seeds:
        # Empty seed set is uncommon but possible (e.g. a feature-request
        # issue with no concrete identifiers). Use the whole skeleton —
        # we already paid the build_skeleton cost.
        logger.debug(
            "localize: ripgrep fallback produced no seeds (queries=%d); "
            "using full skeleton.", len(queries),
        )
        candidate_pool = [o.path for o in skeleton.outlines]
        return candidate_pool, _format_skeleton(skeleton)

    seed_set = set(seeds)
    restricted = _restrict_skeleton(skeleton, seed_set)
    candidate_pool = [o.path for o in restricted.outlines]
    return candidate_pool, _format_skeleton(restricted)


def _extract_symbols_and_errors(issue: "Issue") -> list[str]:
    """Pull likely-relevant query strings out of the issue text.

    We collect, in order, exported-style identifiers, dotted method
    references, and quoted strings (single- or backtick-delimited, since
    Go errors are usually quoted). Common acronyms and English noise
    words are filtered out so the ripgrep step doesn't fire on every
    file in the repo.
    """
    text = f"{issue.title}\n{issue.body}"
    queries: list[str] = []
    seen: set[str] = set()

    def _add(q: str) -> None:
        q = q.strip()
        if not q or q in seen or q in _NOISE_QUERIES:
            return
        seen.add(q)
        queries.append(q)

    # Order matters only for determinism on tie: identifiers first, then
    # dotted refs, then strings. Dedup is by exact match.
    for m in _SYMBOL_RE.finditer(text):
        _add(m.group(0))
    for m in _DOTTED_SYMBOL_RE.finditer(text):
        _add(m.group(0))
    for m in _DQ_STRING_RE.finditer(text):
        _add(m.group(1))
    for m in _BT_STRING_RE.finditer(text):
        _add(m.group(1))

    return queries[:_MAX_FALLBACK_QUERIES]


def _ripgrep_seeds(root: Path, queries: list[str]) -> list[str]:
    """Return ``.go`` files (relative POSIX paths) matching any query.

    Uses ``rg -l -F`` (fixed-string, list filenames) when available; falls
    back to a pure-Python substring scan otherwise. Always uses argv
    arrays — never ``shell=True``.
    """
    if not queries:
        return []

    if shutil.which("rg") is not None:
        return _ripgrep_seeds_rg(root, queries)
    return _ripgrep_seeds_python(root, queries)


def _ripgrep_seeds_rg(root: Path, queries: list[str]) -> list[str]:
    """Run the real ripgrep binary once per query and union the results."""
    seeds: set[str] = set()
    for query in queries:
        try:
            proc = subprocess.run(
                [
                    "rg",
                    "-l",                   # filenames only
                    "-F",                   # treat query as fixed string, not regex
                    "--type", "go",
                    "--path-separator", "/",
                    "--", query, ".",
                ],
                cwd=str(root),
                capture_output=True,
                text=True,
                check=False,
                timeout=_RIPGREP_TIMEOUT_S,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            logger.debug("localize: ripgrep aborted (%s); stopping seed loop.", exc)
            break

        # rg exits 0 on match, 1 on no match, >1 on real errors.
        if proc.returncode not in (0, 1):
            logger.debug(
                "localize: ripgrep returned exit=%d for query=%r; skipping.",
                proc.returncode, query,
            )
            continue
        for line in proc.stdout.splitlines():
            line = line.strip()
            if line.startswith("./"):
                line = line[2:]
            if line.endswith(".go"):
                seeds.add(line)
        if len(seeds) >= _MAX_FALLBACK_SEEDS:
            break

    return sorted(seeds)[:_MAX_FALLBACK_SEEDS]


def _ripgrep_seeds_python(root: Path, queries: list[str]) -> list[str]:
    """Pure-Python fallback used when ``rg`` is not on PATH.

    Walks every ``.go`` file under ``root`` (skipping the same excluded
    directories as the skeleton walker) and records any file containing
    any query as a substring. O(files * queries) but fine on the small
    Go modules in our eval set.
    """
    seeds: set[str] = set()
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [
            d for d in dirs
            if not d.startswith(".") and d not in _EXCLUDED_DIRS
        ]
        for fname in files:
            if not fname.endswith(".go"):
                continue
            abs_path = Path(dirpath) / fname
            try:
                text = abs_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for query in queries:
                if query and query in text:
                    rel = abs_path.relative_to(root).as_posix()
                    seeds.add(rel)
                    break
            if len(seeds) >= _MAX_FALLBACK_SEEDS:
                return sorted(seeds)
    return sorted(seeds)


# ---------------------------------------------------------------------------
# LLM steps: file pick + narrow
# ---------------------------------------------------------------------------


def _llm_pick_files(
    *,
    skeleton_text: str,
    repo: str,
    issue: "Issue",
    top_n: int,
    cfg: "Config",
    allowed_paths: list[str],
) -> list[str]:
    """Ask the LLM to rank the top-N most suspicious files.

    The LLM is prompted with the skeleton text + issue title/body and
    must reply with ``{"files": [...]}``. We then defensively:

      * drop any path that isn't a string;
      * normalize backslashes to ``/``;
      * drop any path not in ``allowed_paths`` (no hallucinations);
      * dedupe while preserving order;
      * truncate to ``top_n``.
    """
    rendered = _render_prompt(
        "file_pick.md",
        repo=repo,
        issue_title=issue.title,
        issue_body=issue.body or "",
        skeleton=skeleton_text,
        top_n=str(top_n),
    )
    prompt = rendered + _FILE_PICK_SUFFIX.format(top_n=top_n)
    parsed = _llm.complete_json(prompt, cfg=cfg.llm, schema={"files": list})

    raw_files = parsed.get("files", [])
    allowed_set = {p.replace("\\", "/") for p in allowed_paths}
    seen: set[str] = set()
    out: list[str] = []
    for entry in raw_files:
        if not isinstance(entry, str):
            continue
        norm = entry.strip().replace("\\", "/")
        if norm.startswith("./"):
            norm = norm[2:]
        if not norm or not norm.endswith(".go"):
            continue
        if norm not in allowed_set or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
        if len(out) >= top_n:
            break
    return out


def _llm_narrow_locations(
    *,
    file_path: str,
    decls_with_lines: list[tuple[int, str]],
    issue: "Issue",
    cfg: "Config",
) -> list[dict]:
    """Ask the LLM to narrow a file to declarations / line ranges.

    Returns a list of raw ``{"symbol", "start_line", "end_line"}`` dicts;
    validation and conversion to :class:`EditLocation` happens in
    :func:`_build_edit_location`. Files with no detectable declarations
    short-circuit to ``[]`` rather than wasting an LLM call. LLM failures
    are logged and treated as "no narrowing for this file" so a single
    flaky call doesn't sink the whole phase.
    """
    if not decls_with_lines:
        return []

    decls_for_prompt = decls_with_lines[:_NARROW_DECLS_LIMIT]
    decls_str = "\n".join(
        f"LINE {line_no}: {decl_text}"
        for line_no, decl_text in decls_for_prompt
    )
    rendered = _render_prompt(
        "narrow.md",
        file_path=file_path,
        declarations=decls_str,
        issue_title=issue.title,
        issue_body=issue.body or "",
    )
    prompt = rendered + _NARROW_SUFFIX

    try:
        parsed = _llm.complete_json(
            prompt, cfg=cfg.llm, schema={"locations": list}
        )
    except _llm.LLMError as exc:
        logger.debug("localize: narrow failed for %s: %s", file_path, exc)
        return []

    locations = parsed.get("locations", [])
    if not isinstance(locations, list):
        return []
    return [loc for loc in locations if isinstance(loc, dict)]


def _render_prompt(template_name: str, **vars: object) -> str:
    """Load ``prompts/<template_name>`` and substitute ``{{key}}`` markers.

    The substitution is intentionally dumb (string replace, not Jinja) so
    the placeholder prompt files in ``prompts/`` continue to work as the
    project evolves. Missing templates raise FileNotFoundError, which is
    a programmer error rather than a runtime concern.
    """
    path = _PROMPTS_DIR / template_name
    text = path.read_text(encoding="utf-8")
    for key, value in vars.items():
        text = text.replace("{{" + key + "}}", str(value))
    return text


# ---------------------------------------------------------------------------
# Per-file declarations with line numbers
# ---------------------------------------------------------------------------


def _decls_with_lines(root: Path, rel_path: str) -> list[tuple[int, str]]:
    """Return ``(line_number, decl_text)`` pairs for top-level decls.

    Mirrors :func:`_outline_file` but tracks 1-indexed line numbers so
    the narrow prompt can talk about specific positions. Returns an
    empty list on read errors (a single unreadable file shouldn't sink
    the run; the file is still in ``ranked_files`` and will get an empty
    narrow result).
    """
    abs_path = root / rel_path
    try:
        text = abs_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.debug("localize: could not read %s for narrow: %s", rel_path, exc)
        return []

    lines = text.splitlines()
    out: list[tuple[int, str]] = []
    i = 0
    n = len(lines)
    while i < n:
        stripped = lines[i].rstrip()

        if _RE_VAR_BLOCK_OPEN.match(stripped) or _RE_CONST_BLOCK_OPEN.match(stripped):
            kind = "var" if stripped.startswith("var") else "const"
            i += 1
            while i < n and not _RE_BLOCK_CLOSE.match(lines[i].rstrip()):
                inner = lines[i]
                m = _RE_BLOCK_EXPORTED.match(inner)
                if m:
                    out.append((i + 1, _truncate(f"{kind} {inner.strip()}")))
                i += 1
            if i < n:
                i += 1  # skip the closing ')'
            continue

        if (
            _RE_FUNC.match(stripped)
            or _RE_TYPE.match(stripped)
            or _RE_VAR_SINGLE.match(stripped)
            or _RE_CONST_SINGLE.match(stripped)
        ):
            out.append((i + 1, _truncate(stripped)))
            i += 1
            continue

        i += 1

    return out


# ---------------------------------------------------------------------------
# EditLocation construction
# ---------------------------------------------------------------------------


def _build_edit_location(
    root: Path, rel_path: str, raw_loc: dict,
) -> EditLocation | None:
    """Convert a raw narrow result to a validated :class:`EditLocation`.

    Returns ``None`` (drop) if any of:
      * ``rel_path`` is not a ``.go`` file or doesn't exist on disk;
      * the file is empty (no lines to point at);
      * ``start_line`` / ``end_line`` aren't ints, or after clamping the
        result has ``start_line > end_line``.

    Property 2 (Localization soundness) follows directly: every returned
    ``EditLocation`` carries ``1 <= start_line <= end_line <= n_lines``,
    a valid ``.go`` file path, and non-empty context.
    """
    if not rel_path.endswith(".go"):
        return None

    abs_path = root / rel_path
    if not abs_path.is_file():
        return None

    start = raw_loc.get("start_line")
    end = raw_loc.get("end_line")
    # ``bool`` is a subclass of ``int`` — exclude it explicitly so a JSON
    # boolean doesn't slip through as a line number.
    if (
        not isinstance(start, int) or isinstance(start, bool)
        or not isinstance(end, int) or isinstance(end, bool)
    ):
        return None

    raw_symbol = raw_loc.get("symbol")
    if isinstance(raw_symbol, str):
        symbol: str | None = raw_symbol.strip() or None
    else:
        symbol = None

    try:
        text = abs_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    lines = text.splitlines()
    n_lines = len(lines)
    if n_lines == 0:
        return None

    # Clamp to [1, n_lines]. After clamping, reject any inversion.
    s = max(1, start)
    e = min(n_lines, end)
    if s > e:
        return None

    pad = _CONTEXT_PADDING_LINES
    ctx_start = max(1, s - pad)
    ctx_end = min(n_lines, e + pad)
    # Hard cap on context size — keep the prompt budget tame even on
    # pathologically wide ranges.
    if ctx_end - ctx_start + 1 > _CONTEXT_MAX_LINES:
        ctx_end = ctx_start + _CONTEXT_MAX_LINES - 1

    ctx_lines = lines[ctx_start - 1 : ctx_end]
    context = "\n".join(
        f"{ln_no:5d}: {line}"
        for ln_no, line in enumerate(ctx_lines, start=ctx_start)
    )
    if not context:
        return None

    return EditLocation(
        file_path=rel_path,
        symbol=symbol,
        start_line=s,
        end_line=e,
        context=context,
    )
