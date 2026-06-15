"""Runtime configuration loader for the Agentless Go Contributor.

Parses ``config.yaml`` plus optional CLI overrides into typed, frozen
dataclasses. The ``ANTHROPIC_API_KEY`` is sourced from the environment only —
never from the YAML file — and is exposed via :func:`require_api_key` rather
than being baked into the immutable :class:`Config`.

Override semantics
------------------
``load_config(path, overrides)`` accepts a flat ``dict`` of CLI overrides:

* Top-level keys (``repo``, ``issue_number``, ``base_commit``, ``top_n_files``,
  ``offline``, ``workdir``, ``outputs_dir``) replace their YAML counterparts.
* The ``llm`` key may be a partial dict that is shallow-merged into the YAML
  ``llm`` block.
* Dotted keys of the form ``"llm.<field>"`` (e.g. ``"llm.model"``) are also
  accepted and merged into the ``llm`` block.

Validation is fail-fast: any missing required field, wrong type, out-of-range
value, repo outside the allowlist, or stray API key in YAML raises
:class:`ConfigError` with an actionable message.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml


# Name of the environment variable that holds the Anthropic API key.
# Sourced from the environment only — never read from the YAML file.
ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"

# Keys that must never appear in YAML — secrets must come from the environment.
_FORBIDDEN_YAML_KEYS = (
    "anthropic_api_key",
    "anthropicApiKey",
    "api_key",
    "apiKey",
)

_REQUIRED_TOP_LEVEL = (
    "repo",
    "repo_allowlist",
    "issue_number",
    "top_n_files",
    "offline",
    "workdir",
    "outputs_dir",
    "llm",
)

_REQUIRED_LLM = (
    "model",
    "temperature",
    "sample_count",
    "max_retries",
    "request_timeout_s",
)


class ConfigError(ValueError):
    """Raised when configuration is missing, malformed, or unsafe."""


@dataclass(frozen=True)
class LLMConfig:
    model: str
    temperature: float
    sample_count: int
    max_retries: int
    request_timeout_s: int


@dataclass(frozen=True)
class Config:
    repo: str
    issue_number: int
    base_commit: Optional[str]
    top_n_files: int
    llm: LLMConfig
    offline: bool
    workdir: Path
    outputs_dir: Path


def load_config(path: Path, overrides: Optional[Mapping[str, Any]] = None) -> Config:
    """Load and validate configuration.

    Parameters
    ----------
    path:
        Path to the YAML configuration file.
    overrides:
        Optional mapping of CLI overrides. See module docstring for semantics.

    Raises
    ------
    ConfigError
        If the file is missing, malformed, contains a forbidden secret key,
        is missing a required field, has a value out of range, or selects a
        repo outside the allowlist.
    """
    raw = _read_yaml(path)
    _reject_secret_keys(raw)

    merged = _apply_overrides(raw, overrides or {})

    _require_keys(merged, _REQUIRED_TOP_LEVEL, where="config")

    repo = _require_str(merged, "repo")
    allowlist = _require_str_list(merged, "repo_allowlist")
    if repo not in allowlist:
        raise ConfigError(
            f"repo {repo!r} is not in repo_allowlist "
            f"({', '.join(allowlist) or 'empty'}); add it to config.yaml or "
            f"pass --repo with an approved repository."
        )

    issue_number = _require_int(merged, "issue_number")
    if issue_number <= 0:
        raise ConfigError(
            f"issue_number must be a positive integer, got {issue_number}."
        )

    top_n_files = _require_int(merged, "top_n_files")
    if top_n_files < 1:
        raise ConfigError(
            f"top_n_files must be >= 1, got {top_n_files}."
        )

    offline = _require_bool(merged, "offline")

    base_commit_raw = merged.get("base_commit", None)
    if base_commit_raw is not None and not isinstance(base_commit_raw, str):
        raise ConfigError(
            f"base_commit must be a string or null, got {type(base_commit_raw).__name__}."
        )
    base_commit: Optional[str] = base_commit_raw or None

    workdir = _require_path(merged, "workdir")
    outputs_dir = _require_path(merged, "outputs_dir")

    llm_raw = merged.get("llm")
    if not isinstance(llm_raw, Mapping):
        raise ConfigError("llm must be a mapping with model/temperature/...")
    _require_keys(llm_raw, _REQUIRED_LLM, where="llm")

    model = _require_str(llm_raw, "model", parent="llm")
    temperature = _require_number(llm_raw, "temperature", parent="llm")
    if temperature < 0:
        raise ConfigError(
            f"llm.temperature must be >= 0, got {temperature}."
        )
    sample_count = _require_int(llm_raw, "sample_count", parent="llm")
    if sample_count < 1:
        raise ConfigError(
            f"llm.sample_count must be >= 1, got {sample_count}."
        )
    max_retries = _require_int(llm_raw, "max_retries", parent="llm")
    if max_retries < 0:
        raise ConfigError(
            f"llm.max_retries must be >= 0, got {max_retries}."
        )
    request_timeout_s = _require_int(llm_raw, "request_timeout_s", parent="llm")
    if request_timeout_s <= 0:
        raise ConfigError(
            f"llm.request_timeout_s must be > 0, got {request_timeout_s}."
        )

    return Config(
        repo=repo,
        issue_number=issue_number,
        base_commit=base_commit,
        top_n_files=top_n_files,
        llm=LLMConfig(
            model=model,
            temperature=float(temperature),
            sample_count=sample_count,
            max_retries=max_retries,
            request_timeout_s=request_timeout_s,
        ),
        offline=offline,
        workdir=workdir,
        outputs_dir=outputs_dir,
    )


def get_api_key() -> Optional[str]:
    """Return the Anthropic API key from the environment, if set."""
    value = os.environ.get(ANTHROPIC_API_KEY_ENV)
    return value or None


def require_api_key() -> str:
    """Return the Anthropic API key or raise :class:`ConfigError` if absent."""
    value = get_api_key()
    if not value:
        raise ConfigError(
            f"{ANTHROPIC_API_KEY_ENV} is not set in the environment. "
            f"Export it before running the pipeline (offline mode does not "
            f"require it)."
        )
    return value


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise ConfigError(
            f"config file not found at {path}; pass --config or create config.yaml."
        )
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"failed to parse YAML at {path}: {exc}") from exc
    if data is None:
        raise ConfigError(f"config file at {path} is empty.")
    if not isinstance(data, dict):
        raise ConfigError(
            f"config file at {path} must define a mapping at the top level, "
            f"got {type(data).__name__}."
        )
    return data


def _reject_secret_keys(raw: Mapping[str, Any]) -> None:
    for key in _FORBIDDEN_YAML_KEYS:
        if key in raw:
            raise ConfigError(
                f"refusing to load: {key!r} found in YAML. API keys must come "
                f"from the {ANTHROPIC_API_KEY_ENV} environment variable, not "
                f"the config file."
            )


def _apply_overrides(raw: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict:
    merged: dict = dict(raw)
    # Carry the YAML llm block forward as a mutable copy so nested overrides
    # are shallow-merged on top of it.
    if "llm" in merged and isinstance(merged["llm"], Mapping):
        merged["llm"] = dict(merged["llm"])

    for key, value in overrides.items():
        if key == "llm":
            if not isinstance(value, Mapping):
                raise ConfigError(
                    f"override 'llm' must be a mapping, got {type(value).__name__}."
                )
            base = merged.get("llm")
            base_dict = dict(base) if isinstance(base, Mapping) else {}
            base_dict.update(value)
            merged["llm"] = base_dict
        elif "." in key:
            head, _, tail = key.partition(".")
            if head != "llm":
                raise ConfigError(
                    f"unsupported dotted override key {key!r}; only 'llm.<field>' is allowed."
                )
            base = merged.get("llm")
            base_dict = dict(base) if isinstance(base, Mapping) else {}
            base_dict[tail] = value
            merged["llm"] = base_dict
        else:
            merged[key] = value
    return merged


def _require_keys(mapping: Mapping[str, Any], keys: tuple, *, where: str) -> None:
    missing = [k for k in keys if k not in mapping]
    if missing:
        raise ConfigError(
            f"missing required {where} field(s): {', '.join(missing)}."
        )


def _qualified(name: str, parent: Optional[str]) -> str:
    return f"{parent}.{name}" if parent else name


def _require_str(mapping: Mapping[str, Any], name: str, *, parent: Optional[str] = None) -> str:
    value = mapping.get(name)
    if not isinstance(value, str) or not value:
        raise ConfigError(
            f"{_qualified(name, parent)} must be a non-empty string, got {value!r}."
        )
    return value


def _require_str_list(mapping: Mapping[str, Any], name: str, *, parent: Optional[str] = None) -> list:
    value = mapping.get(name)
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        raise ConfigError(
            f"{_qualified(name, parent)} must be a list of non-empty strings, got {value!r}."
        )
    return list(value)


def _require_int(mapping: Mapping[str, Any], name: str, *, parent: Optional[str] = None) -> int:
    value = mapping.get(name)
    # Reject bools explicitly — bool is a subclass of int in Python.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(
            f"{_qualified(name, parent)} must be an integer, got {value!r}."
        )
    return value


def _require_number(mapping: Mapping[str, Any], name: str, *, parent: Optional[str] = None) -> float:
    value = mapping.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(
            f"{_qualified(name, parent)} must be a number, got {value!r}."
        )
    return float(value)


def _require_bool(mapping: Mapping[str, Any], name: str, *, parent: Optional[str] = None) -> bool:
    value = mapping.get(name)
    if not isinstance(value, bool):
        raise ConfigError(
            f"{_qualified(name, parent)} must be a boolean, got {value!r}."
        )
    return value


def _require_path(mapping: Mapping[str, Any], name: str, *, parent: Optional[str] = None) -> Path:
    value = mapping.get(name)
    if isinstance(value, Path):
        return value
    if isinstance(value, str) and value:
        return Path(value)
    raise ConfigError(
        f"{_qualified(name, parent)} must be a non-empty path string, got {value!r}."
    )
