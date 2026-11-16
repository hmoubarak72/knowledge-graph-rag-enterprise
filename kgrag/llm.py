"""All LLM calls go through `LLMClient`: cached, budget-checked, schema-validated.

* Cache: SQLite keyed by SHA-256 of the full request. Re-runs cost $0.
* Budget: every call is estimated *before* it is sent; if ledger spend + estimate would exceed
  `KGRAG_BUDGET_USD` the call aborts with `BudgetExceededError`. Actual spend is recorded from
  `response.usage`.
* Validation: responses are parsed into a Pydantic model; on failure the model is re-asked with
  the validation error attached (bounded retries).
* Offline mode (`KGRAG_OFFLINE=1`): a deterministic `FakeBackend` supplies responses, used by tests.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from kgrag.config import Settings, get_settings
from kgrag.log import get_logger

log = get_logger(__name__)
T = TypeVar("T", bound=BaseModel)

# USD per 1M tokens (input, output). Source: Anthropic pricing table, 2026-11.
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-fable-5-2": (8.00, 40.00),
}
# Models that accept server-side refusal fallbacks ("default" routing by refusal category).
FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-2"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"
CHARS_PER_TOKEN = 3.5  # conservative pre-call estimate; real usage is recorded afterwards


class BudgetExceededError(RuntimeError):
    pass


class LLMValidationError(RuntimeError):
    pass


class LLMRefusalError(RuntimeError):
    pass


def price(model: str, input_tokens: int, output_tokens: int) -> float:
    pin, pout = PRICING.get(model, (4.00, 20.00))
    return (input_tokens * pin + output_tokens * pout) / 1_000_000


def estimate_cost(model: str, prompt_chars: int, expected_output_tokens: int) -> float:
    return price(model, int(prompt_chars / CHARS_PER_TOKEN), expected_output_tokens)


@dataclass
class CallStats:
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cached: bool
    latency_ms: float
    attempts: int


class Backend(Protocol):
    def complete(
        self, *, model: str, system: str, messages: list[dict[str, Any]], schema: dict[str, Any],
        max_tokens: int, effort: str,
    ) -> tuple[str, int, int]:
        """Return (text, input_tokens, output_tokens)."""


class AnthropicBackend:
    def __init__(self, api_key: str) -> None:
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=api_key or None)

    def complete(
        self, *, model: str, system: str, messages: list[dict[str, Any]], schema: dict[str, Any],
        max_tokens: int, effort: str,
    ) -> tuple[str, int, int]:
        kw: dict[str, Any] = {
            "model": model, "max_tokens": max_tokens, "system": system, "messages": messages,
            "output_config": {"format": {"type": "json_schema", "schema": schema}, "effort": effort},
        }
        resp: Any
        if model in FALLBACK_MODELS:
            # server-side refusal fallback ("default" routing by refusal category)
            resp = self.client.beta.messages.create(**kw, betas=[FALLBACK_BETA], fallbacks="default")
        else:
            resp = self.client.messages.create(**kw)
        if resp.stop_reason == "refusal":
            raise LLMRefusalError(f"model declined: {getattr(resp, 'stop_details', None)}")
        if resp.stop_reason == "max_tokens":
            raise LLMValidationError(f"output truncated at max_tokens={max_tokens}")
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return text, resp.usage.input_tokens, resp.usage.output_tokens


class FakeBackend:
    """Deterministic offline backend. `responder(model, system, user_text) -> dict|str`."""

    def __init__(self, responder: Callable[[str, str, str], Any]) -> None:
        self.responder = responder
        self.calls = 0

    def complete(
        self, *, model: str, system: str, messages: list[dict[str, Any]], schema: dict[str, Any],
        max_tokens: int, effort: str,
    ) -> tuple[str, int, int]:
        self.calls += 1
        user_text = messages[-1]["content"] if isinstance(messages[-1]["content"], str) else ""
        out = self.responder(model, system, user_text)
        text = out if isinstance(out, str) else json.dumps(out)
        return text, int((len(system) + len(user_text)) / CHARS_PER_TOKEN), int(len(text) / CHARS_PER_TOKEN)


def json_schema_for(model_cls: type[BaseModel]) -> dict[str, Any]:
    """Pydantic -> JSON schema in the subset structured outputs accepts."""
    try:
        from anthropic.lib._parse._transform import transform_schema

        return transform_schema(model_cls)
    except Exception:  # pragma: no cover - fallback if the private helper moves
        return model_cls.model_json_schema()


class _Store:
    """SQLite-backed response cache + spend ledger."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS cache(key TEXT PRIMARY KEY, text TEXT, model TEXT,
                    input_tokens INT, output_tokens INT, created REAL);
                CREATE TABLE IF NOT EXISTS ledger(ts REAL, model TEXT, purpose TEXT,
                    input_tokens INT, output_tokens INT, usd REAL);
                """
            )

    def get(self, key: str) -> tuple[str, int, int] | None:
        with self.lock:
            row = self.conn.execute(
                "SELECT text, input_tokens, output_tokens FROM cache WHERE key=?", (key,)
            ).fetchone()
        return (row[0], row[1], row[2]) if row else None

    def put(self, key: str, text: str, model: str, tin: int, tout: int) -> None:
        with self.lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO cache VALUES (?,?,?,?,?,?)", (key, text, model, tin, tout, time.time())
            )
            self.conn.commit()

    def spend(self, purpose: str | None = None) -> float:
        q, args = "SELECT COALESCE(SUM(usd),0) FROM ledger", ()
        if purpose:
            q, args = q + " WHERE purpose=?", (purpose,)  # type: ignore[assignment]
        with self.lock:
            return float(self.conn.execute(q, args).fetchone()[0])

    def record(self, model: str, purpose: str, tin: int, tout: int, usd: float) -> None:
        with self.lock:
            self.conn.execute("INSERT INTO ledger VALUES (?,?,?,?,?,?)", (time.time(), model, purpose, tin, tout, usd))
            self.conn.commit()


class LLMClient:
    def __init__(self, settings: Settings | None = None, backend: Backend | None = None,
                 store_path: Path | None = None) -> None:
        self.settings = settings or get_settings()
        self.store = _Store(store_path or self.settings.cache_dir / "llm.sqlite")
        # Benchmarks set this False so latency/cost reflect real calls (results are still written).
        self.read_cache = True
        if backend is not None:
            self.backend: Backend = backend
        elif self.settings.offline:
            from kgrag.offline import offline_responder

            self.backend = FakeBackend(offline_responder)
        else:
            self.backend = AnthropicBackend(self.settings.anthropic_api_key)

    # ------------------------------------------------------------------ budget
    def spent(self, purpose: str | None = None) -> float:
        return self.store.spend(purpose)

    def check_budget(self, estimated_usd: float, label: str = "call") -> None:
        spent = self.store.spend()
        cap = self.settings.budget_usd
        if spent + estimated_usd > cap:
            raise BudgetExceededError(
                f"{label}: estimated ${estimated_usd:.4f} + spent ${spent:.4f} exceeds cap ${cap:.2f} "
                f"(raise KGRAG_BUDGET_USD to continue)"
            )

    # ------------------------------------------------------------------ main entry point
    def structured(
        self,
        *,
        model: str,
        system: str,
        user: str,
        output: type[T],
        purpose: str,
        max_tokens: int = 8000,
        expected_output_tokens: int = 1500,
        effort: str = "medium",
        max_attempts: int = 3,
        post_validate: Callable[[T], None] | None = None,
    ) -> tuple[T, CallStats]:
        """Call the model, parse into `output`, retry with the validation error attached."""
        schema = json_schema_for(output)
        messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
        total_in = total_out = 0
        total_cost = 0.0
        t0 = time.perf_counter()
        all_cached = True
        last_err: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            key = hashlib.sha256(
                json.dumps({"m": model, "s": system, "msgs": messages, "schema": schema, "e": effort},
                           sort_keys=True).encode()
            ).hexdigest()
            hit = self.store.get(key) if self.read_cache else None
            if hit:
                text, tin, tout = hit
            else:
                all_cached = False
                est = estimate_cost(model, len(system) + sum(len(str(m["content"])) for m in messages),
                                    expected_output_tokens)
                self.check_budget(est, purpose)
                text, tin, tout = self.backend.complete(
                    model=model, system=system, messages=messages, schema=schema,
                    max_tokens=max_tokens, effort=effort,
                )
                cost = price(model, tin, tout)
                self.store.record(model, purpose, tin, tout, cost)
                total_cost += cost
                total_in += tin
                total_out += tout
            try:
                parsed = output.model_validate_json(text)
                if post_validate:
                    post_validate(parsed)
            except (ValidationError, ValueError) as err:
                last_err = err
                log.warning("llm.validation_failed", purpose=purpose, attempt=attempt, error=str(err)[:300])
                messages = messages + [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": (
                        "Your previous output failed validation with this error:\n"
                        f"{str(err)[:2000]}\n\nReturn the complete corrected JSON only."
                    )},
                ]
                continue
            if not hit:
                self.store.put(key, text, model, tin, tout)
            stats = CallStats(model, total_in, total_out, total_cost, all_cached,
                              (time.perf_counter() - t0) * 1000, attempt)
            return parsed, stats

        raise LLMValidationError(f"{purpose}: no valid output after {max_attempts} attempts: {last_err}")
