"""Anthropic Claude wrapper with retry, JSON parsing, and traceable logging.

This module is the single point through which every LLM call in the pipeline
flows. It centralizes:
  * Retry/backoff for transient errors (timeouts, 5xx, rate limit), bounded by
    ``cfg.max_retries``.
  * ``n``-sample fan-out (the Anthropic Messages API does not natively support
    ``n``; we loop and emit one ``LLMResponse`` per sample).
  * Strict-JSON parsing with a single repair re-prompt on malformed output.
  * Prompt + response logging to ``outputs/llm_calls.jsonl`` for traceability.

The public surface is intentionally small: ``LLMResponse``, ``LLMError``,
``complete``, ``complete_json``.

Imports of the ``anthropic`` SDK are deferred to call time so that unit tests
can monkeypatch ``_call_anthropic`` without forcing an API key into the env at
import time.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from src.config import LLMConfig


__all__ = ["LLMResponse", "LLMError", "complete", "complete_json"]


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class LLMError(RuntimeError):
    """Raised for unrecoverable LLM failures (auth, repeated transient errors,
    malformed JSON that cannot be repaired, schema mismatch).
    """


@dataclass(frozen=True)
class LLMResponse:
    """A single completion sample.

    ``text`` is the concatenated text from all text content blocks of the
    Anthropic message; ``raw`` is the dict-form of the underlying response,
    preserved for traceability and downstream debugging.
    """

    text: str
    raw: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


_DEFAULT_LOG_PATH = Path("outputs") / "llm_calls.jsonl"
_LOG_TRUNCATE_CHARS = 8000
_INITIAL_BACKOFF_S = 1.0
_BACKOFF_MULTIPLIER = 2.0


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _truncate(text: str, limit: int = _LOG_TRUNCATE_CHARS) -> tuple[str, bool]:
    """Return (possibly truncated text, was_truncated)."""
    if text is None:
        return "", False
    if len(text) <= limit:
        return text, False
    return text[:limit], True


def _log_call(
    *,
    prompt: str,
    response_text: Optional[str],
    raw: Optional[dict],
    error: Optional[str],
    attempt: int,
    sample_index: int,
    sample_count: int,
    model: str,
    temperature: Optional[float],
    log_path: Path = _DEFAULT_LOG_PATH,
) -> None:
    """Append a single JSON line describing this LLM call.

    Best-effort: never raises. Long bodies are truncated to ``_LOG_TRUNCATE_CHARS``
    with a ``..._truncated`` flag noted alongside.
    """
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_text, prompt_truncated = _truncate(prompt)
        resp_text, resp_truncated = _truncate(response_text or "")
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "temperature": temperature,
            "sample_index": sample_index,
            "sample_count": sample_count,
            "attempt": attempt,
            "prompt": prompt_text,
            "prompt_truncated": prompt_truncated,
            "response_text": resp_text,
            "response_truncated": resp_truncated,
            "error": error,
            "raw_keys": sorted(list(raw.keys())) if isinstance(raw, dict) else None,
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:  # pragma: no cover - logging must never break a run
        pass


# ---------------------------------------------------------------------------
# Retry helpers
# ---------------------------------------------------------------------------


def _is_transient(exc: BaseException) -> bool:
    """Heuristic classifier for retryable Anthropic / network errors.

    We intentionally check by class-name and HTTP status rather than importing
    SDK types directly so this module imports cleanly even when the SDK is
    absent at unit-test time.
    """
    name = type(exc).__name__
    transient_names = {
        "APITimeoutError",
        "APIConnectionError",
        "RateLimitError",
        "InternalServerError",
        "ServiceUnavailableError",
        "Timeout",
        "ConnectionError",
        "ReadTimeout",
        "ConnectTimeout",
    }
    if name in transient_names:
        return True
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and (status == 429 or 500 <= status < 600):
        return True
    return False


def _with_retries(fn: Callable[[], Any], *, max_retries: int) -> Any:
    """Invoke ``fn`` with exponential backoff on transient failures.

    Total attempts = ``max_retries + 1`` (initial attempt + retries). Non-transient
    exceptions are re-raised immediately, wrapped as :class:`LLMError`.
    """
    last_exc: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except LLMError:
            raise
        except BaseException as exc:  # noqa: BLE001 - we want broad transient handling
            last_exc = exc
            if not _is_transient(exc) or attempt == max_retries:
                if isinstance(exc, LLMError):
                    raise
                raise LLMError(f"LLM call failed: {type(exc).__name__}: {exc}") from exc
            sleep_s = _INITIAL_BACKOFF_S * (_BACKOFF_MULTIPLIER ** attempt)
            time.sleep(sleep_s)
    # Defensive: loop should always either return or raise above.
    raise LLMError(f"LLM call exhausted retries: {last_exc!r}")


# ---------------------------------------------------------------------------
# Anthropic call
# ---------------------------------------------------------------------------


def _require_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise LLMError(
            "ANTHROPIC_API_KEY is not set. Export it in your environment "
            "before invoking the pipeline (it must never be placed in config.yaml)."
        )
    return key


def _response_to_dict(response: Any) -> dict:
    """Best-effort conversion of an SDK response object to a plain dict."""
    for attr in ("model_dump", "to_dict", "dict"):
        fn = getattr(response, attr, None)
        if callable(fn):
            try:
                value = fn()
            except TypeError:
                continue
            if isinstance(value, dict):
                return value
    if isinstance(response, dict):
        return response
    try:
        return dict(response)
    except Exception:
        return {"repr": repr(response)}


def _extract_text(response: Any) -> str:
    """Concatenate text from all text-type content blocks."""
    content = getattr(response, "content", None)
    if content is None and isinstance(response, dict):
        content = response.get("content")
    if not content:
        return ""
    chunks: list[str] = []
    for block in content:
        block_type = getattr(block, "type", None)
        text = getattr(block, "text", None)
        if block_type is None and isinstance(block, dict):
            block_type = block.get("type")
            text = block.get("text")
        if block_type == "text" and isinstance(text, str):
            chunks.append(text)
    return "".join(chunks)


def _call_anthropic(
    prompt: str,
    *,
    cfg: "LLMConfig",
    temperature: float,
) -> LLMResponse:
    """Single Anthropic Messages API call. Defined at module level so unit
    tests can monkeypatch it without touching the SDK or environment.
    """
    # Lazy import keeps this module importable without the SDK installed.
    from anthropic import Anthropic  # type: ignore[import-not-found]

    api_key = _require_api_key()
    client = Anthropic(api_key=api_key, timeout=cfg.request_timeout_s)
    response = client.messages.create(
        model=cfg.model,
        max_tokens=4096,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return LLMResponse(text=_extract_text(response), raw=_response_to_dict(response))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def complete(
    prompt: str,
    *,
    cfg: "LLMConfig",
    temperature: Optional[float] = None,
    n: int = 1,
) -> list[LLMResponse]:
    """Generate ``n`` completions for ``prompt``.

    The Anthropic Messages API has no native ``n``; we loop and produce one
    ``LLMResponse`` per sample. Each call is wrapped in retry/backoff bounded
    by ``cfg.max_retries`` and logged to ``outputs/llm_calls.jsonl``.

    Raises ``LLMError`` on auth failure or after all retries are exhausted.
    """
    if n < 1:
        raise LLMError(f"complete(): n must be >= 1, got {n}")
    effective_temp = cfg.temperature if temperature is None else temperature

    responses: list[LLMResponse] = []
    for i in range(n):
        attempts = {"count": 0}

        def _do_call() -> LLMResponse:
            attempts["count"] += 1
            return _call_anthropic(prompt, cfg=cfg, temperature=effective_temp)

        try:
            resp = _with_retries(_do_call, max_retries=cfg.max_retries)
        except LLMError as exc:
            _log_call(
                prompt=prompt,
                response_text=None,
                raw=None,
                error=str(exc),
                attempt=attempts["count"],
                sample_index=i,
                sample_count=n,
                model=cfg.model,
                temperature=effective_temp,
            )
            raise

        _log_call(
            prompt=prompt,
            response_text=resp.text,
            raw=resp.raw,
            error=None,
            attempt=attempts["count"],
            sample_index=i,
            sample_count=n,
            model=cfg.model,
            temperature=effective_temp,
        )
        responses.append(resp)

    return responses


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _strip_fences(text: str) -> str:
    """Remove the first ```json ... ``` (or ``` ... ```) fence if present."""
    match = _JSON_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _find_balanced(text: str, open_ch: str, close_ch: str) -> Optional[str]:
    """Return the substring spanning the first balanced ``open_ch``/``close_ch``
    pair, accounting for string literals and escapes. None if no balanced pair.
    """
    start = text.find(open_ch)
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _extract_json_payload(text: str) -> Any:
    """Best-effort parse of an LLM response into a Python object.

    Strategy:
      1. Strip a leading ```json fence.
      2. Try ``json.loads`` directly.
      3. Locate the first balanced ``{...}`` or ``[...]`` block and parse that.

    Raises ``json.JSONDecodeError`` on failure.
    """
    cleaned = _strip_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Try object first, then array.
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        candidate = _find_balanced(cleaned, open_ch, close_ch)
        if candidate is not None:
            return json.loads(candidate)
    # Trigger the canonical decode error on the original cleaned text.
    return json.loads(cleaned)


def _validate_schema(obj: Any, schema: dict) -> None:
    """Minimal structural validation against ``schema``.

    ``schema`` is interpreted as a mapping of required top-level key -> type
    spec, where the spec is one of:
      * a Python ``type`` (e.g. ``str``, ``int``, ``list``, ``dict``)
      * a string alias: ``"str"``, ``"int"``, ``"float"``, ``"bool"``,
        ``"list"``, ``"dict"``, ``"any"``
      * any other value, in which case only key presence is checked.

    Raises :class:`LLMError` on any violation.
    """
    if not isinstance(schema, dict):
        return  # nothing to validate against
    if not isinstance(obj, dict):
        raise LLMError(
            f"schema validation: expected JSON object, got {type(obj).__name__}"
        )

    alias_map: dict[str, type] = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
    }

    for key, spec in schema.items():
        if key not in obj:
            raise LLMError(f"schema validation: missing required key '{key}'")
        expected: Optional[type]
        if isinstance(spec, type):
            expected = spec
        elif isinstance(spec, str):
            if spec == "any":
                expected = None
            else:
                expected = alias_map.get(spec)
        else:
            expected = None
        if expected is not None and not isinstance(obj[key], expected):
            raise LLMError(
                f"schema validation: key '{key}' expected {expected.__name__}, "
                f"got {type(obj[key]).__name__}"
            )


def _schema_hint(schema: dict) -> str:
    """Human-readable rendering of ``schema`` for the repair prompt."""
    if not isinstance(schema, dict):
        return str(schema)
    parts = []
    for key, spec in schema.items():
        if isinstance(spec, type):
            parts.append(f'"{key}": <{spec.__name__}>')
        else:
            parts.append(f'"{key}": <{spec}>')
    return "{ " + ", ".join(parts) + " }"


def complete_json(
    prompt: str,
    *,
    cfg: "LLMConfig",
    schema: dict,
) -> dict:
    """Generate a single JSON-shaped completion and return the parsed object.

    Performs at most one repair attempt: if the first response cannot be parsed
    as JSON (or fails schema validation), we re-prompt the model with the
    original prompt plus a short "your previous response was not valid JSON"
    instruction. After that, we give up with :class:`LLMError`.
    """
    responses = complete(prompt, cfg=cfg, n=1)
    first_text = responses[0].text

    parse_error: Optional[str] = None
    try:
        parsed = _extract_json_payload(first_text)
        if not isinstance(parsed, dict):
            raise LLMError(
                f"complete_json: expected JSON object at top level, "
                f"got {type(parsed).__name__}"
            )
        _validate_schema(parsed, schema)
        return parsed
    except json.JSONDecodeError as exc:
        parse_error = f"JSON parse error: {exc.msg} (line {exc.lineno} col {exc.colno})"
    except LLMError as exc:
        parse_error = str(exc)

    # One repair attempt.
    repair_prompt = (
        f"{prompt}\n\n"
        "Your previous response was not valid JSON or did not match the required "
        "schema. Return ONLY a single JSON object — no prose, no code fences — "
        f"matching this schema: {_schema_hint(schema)}.\n"
        f"Parsing/validation error from your previous response: {parse_error}"
    )
    repair_responses = complete(repair_prompt, cfg=cfg, n=1)
    repair_text = repair_responses[0].text
    try:
        parsed = _extract_json_payload(repair_text)
    except json.JSONDecodeError as exc:
        raise LLMError(
            f"complete_json: could not parse JSON after one repair attempt: "
            f"{exc.msg} (line {exc.lineno} col {exc.colno})"
        ) from exc
    if not isinstance(parsed, dict):
        raise LLMError(
            f"complete_json: repair attempt did not return a JSON object "
            f"(got {type(parsed).__name__})"
        )
    _validate_schema(parsed, schema)
    return parsed
