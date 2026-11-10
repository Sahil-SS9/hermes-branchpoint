"""LLM access for Branchpoint: one tiny interface, three backends.

* ``CtxLLM``       Hermes ``ctx.llm`` (the user's active model/auth; the plugin never sees a key).
* ``HermesCliLLM`` the ``hermes chat -Q -q`` command line, for running Branchpoint outside a session.
* ``FakeLLM``      scripted replies for tests.

Every backend charges a shared ``Meter``, which is how the per-run token budget gets a hard stop.
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .util import BranchpointError, hermes_child_env, run, short


class BudgetExceeded(BranchpointError):
    """The run (or this route's share of it) hit its hard token budget."""


class ModelOverrideRefused(BranchpointError):
    """The host refused the model this call asked for.

    Final for the run. Every model Branchpoint passes (an agent's, a reviewer's, the judge's, or the configured
    default) is a deliberate choice, so the call stops instead of retrying without the model: answering on a
    different model would make the run report, and judge, a model that never ran.
    """


@dataclass
class LLMReply:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float | None = None
    model: str = ""

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLM(Protocol):
    def complete(self, system: str, user: str, *, max_tokens: int = 3500, temperature: float | None = None,
                 model: str | None = None, purpose: str = "") -> LLMReply: ...


class Meter:
    """Thread-safe token meter with a hard cap; ``child`` carves a per-route share out of it."""

    def __init__(self, cap: int | None = None, parent: "Meter | None" = None, label: str = "run"):
        self.cap = cap
        self.parent = parent
        self.label = label
        self.used = 0
        self.usd = 0.0
        self.calls = 0
        self._lock = threading.Lock()

    def child(self, label: str, cap: int | None) -> "Meter":
        return Meter(cap, self, label)

    @property
    def exhausted(self) -> bool:
        if self.cap is not None and self.used >= self.cap:
            return True
        return bool(self.parent and self.parent.exhausted)

    def remaining(self) -> int | None:
        """Tokens left before the hard stop, or None when nothing is capped.

        A child meter is also bound by its parent's remaining budget: a route's share of the run budget is only
        real while the run itself still has that much left.
        """
        caps: list[int] = []
        if self.cap is not None:
            caps.append(max(0, int(self.cap) - int(self.used)))
        up = self.parent.remaining() if self.parent else None
        if up is not None:
            caps.append(up)
        return min(caps) if caps else None

    def check(self) -> None:
        if self.exhausted:
            who = self.label if (self.cap is not None and self.used >= self.cap) else (self.parent.label if self.parent else self.label)
            raise BudgetExceeded(f"token budget exhausted ({who}: {self.used}/{self.cap or '∞'})")

    def charge(self, tokens: int, usd: float | None = None) -> None:
        with self._lock:
            self.used += max(0, int(tokens))
            self.calls += 1
            if usd:
                self.usd += float(usd)
        if self.parent:
            self.parent.charge(tokens, usd)


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def redact_for_model(text: str) -> str:
    """Strip credentials from anything sent to a model, via the host's own scrubber when present."""
    try:
        from agent.redact import redact_sensitive_text  # public plugin API
        return redact_sensitive_text(text, force=True)
    except Exception:
        return text


def redaction_available() -> bool:
    try:
        from agent.redact import redact_sensitive_text  # noqa: F401
        return True
    except Exception:
        return False


@dataclass
class CtxLLMState:
    """Shared by every meter clone of one run's ``CtxLLM``: one refusal record per run, not per clone.

    A route or the council gets its own clone. Without one shared record a refusal would be learned three times
    and a clone could re-ask for a model the host has already denied. The record belongs to a *run*: the host
    re-reads its trust settings per call, so a permission granted between runs must be tried again instead of
    being refused from a stale record (see ``CtxLLM.for_run``).
    """

    refused: set[str] = field(default_factory=set)
    lock: Any = field(default_factory=threading.Lock, repr=False, compare=False)


#: Wording the host uses when it denies a plugin's model override (``PluginLlmTrustError``).
_MODEL_REFUSAL_WORDS = ("override", "not allowed", "may not", "forbidden", "not permitted", "denied", "trust")


def is_model_override_refusal(exc: BaseException) -> bool:
    """True only for 'this plugin may not choose a model'.

    A bare ``PermissionError`` can come from anywhere (an unwritable path, a provider policy), and treating it as
    a model refusal would silently retry on a different model and hide the real error. So the error has to be a
    permission error that is *about a model*: either the host's trust error by type, or model + a denial in its text.
    """
    if not isinstance(exc, PermissionError):
        return False
    if "trust" in type(exc).__name__.lower():
        return True
    msg = str(exc).lower()
    if "model" not in msg:
        return False
    return any(w in msg for w in _MODEL_REFUSAL_WORDS)


class CtxLLM:
    """Adapter over ``ctx.llm.complete``."""

    def __init__(self, ctx: Any, meter: Meter | None = None, timeout: float = 240.0, default_model: str | None = None,
                 state: CtxLLMState | None = None):
        self._ctx = ctx
        self.meter = meter or Meter()
        self.timeout = timeout
        self.default_model = default_model
        self.state = state or CtxLLMState()

    @property
    def refused_models(self) -> set[str]:
        """Models the host has already refused for this backend (empty until the first refusal)."""
        with self.state.lock:
            return set(self.state.refused)

    def with_meter(self, meter: Meter) -> "CtxLLM":
        """A clone that shares this run's refusal record (a route's share of the run budget)."""
        return CtxLLM(self._ctx, meter, self.timeout, self.default_model, self.state)

    def for_run(self, meter: Meter) -> "CtxLLM":
        """A clone with a clean refusal record: one new call == one new run.

        The host re-reads ``plugins.entries.branchpoint.llm.allow_model_override`` per call, so a model refused
        during one run may be granted before the next. Carrying the record across runs would refuse that run
        without ever asking the host. Inside a run the record is shared (``with_meter``), so a refusal is never
        retried and the run stays fail-closed.
        """
        return CtxLLM(self._ctx, meter, self.timeout, self.default_model, CtxLLMState())

    def _already_refused(self, model: str) -> bool:
        with self.state.lock:
            return model in self.state.refused

    def _remember_refusal(self, model: str) -> None:
        with self.state.lock:
            self.state.refused.add(model)

    def _refusal(self, model: str, detail: str) -> ModelOverrideRefused:
        return ModelOverrideRefused(
            f"Hermes refused the model override '{model}' that this run asked for ({detail}). Branchpoint stopped "
            "instead of answering on a different model: every model it is given is an explicit choice, so a "
            "swapped model would make this run's report name a model that never ran. Grant it with "
            "plugins.entries.branchpoint.llm.allow_model_override (or clear that model setting), then run again.")

    def complete(self, system: str, user: str, *, max_tokens: int = 3500, temperature: float | None = None,
                 model: str | None = None, purpose: str = "") -> LLMReply:
        self.meter.check()
        # An explicit ``model`` and the backend's ``default_model`` are both choices the run made; neither is a
        # hint the host may replace. If either was refused before, fail here without spending another round trip.
        requested = model or self.default_model
        if requested and self._already_refused(requested):
            raise self._refusal(requested, "already refused earlier in this run")
        kw: dict[str, Any] = {"max_tokens": max_tokens, "timeout": self.timeout}
        if temperature is not None:
            kw["temperature"] = temperature
        if requested:
            kw["model"] = requested
        if purpose:
            kw["purpose"] = f"branchpoint:{purpose}"
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            res = self._ctx.llm.complete(messages=messages, **kw)
        except PermissionError as exc:   # PluginLlmTrustError: a model override this plugin has not been granted
            if not requested or not is_model_override_refusal(exc):
                raise                    # a real permission problem, or nothing of ours to interpret: unchanged
            self._remember_refusal(requested)
            raise self._refusal(requested, short(exc, 120)) from exc
        usage = getattr(res, "usage", None)
        tin = int(getattr(usage, "input_tokens", 0) or 0)
        tout = int(getattr(usage, "output_tokens", 0) or 0)
        if not (tin or tout):
            total = int(getattr(usage, "total_tokens", 0) or 0)
            tin, tout = (total, 0) if total else (estimate_tokens(system + user), estimate_tokens(getattr(res, "text", "") or ""))
        self.meter.charge(tin + tout, getattr(usage, "cost_usd", None))
        return LLMReply(getattr(res, "text", "") or "", tin, tout, getattr(usage, "cost_usd", None), getattr(res, "model", "") or "")


def hermes_argv() -> list[str]:
    exe = shutil.which("hermes")
    return [exe] if exe else [sys.executable, "-m", "hermes_cli.main"]


_SESSION_LINE = re.compile(r"\n*session_id:\s*\S+\s*$")


class HermesCliLLM:
    """Backend that shells out to ``hermes chat -Q -q`` (no tools are requested; JSON-only prompts)."""

    def __init__(self, meter: Meter | None = None, model: str | None = None, timeout: float = 300.0):
        self.meter = meter or Meter()
        self.model = model
        self.timeout = timeout

    def with_meter(self, meter: Meter) -> "HermesCliLLM":
        return HermesCliLLM(meter, self.model, self.timeout)

    def complete(self, system: str, user: str, *, max_tokens: int = 3500, temperature: float | None = None,
                 model: str | None = None, purpose: str = "") -> LLMReply:
        self.meter.check()
        prompt = f"{system}\n\n---\n\n{user}\n\nReply with the requested output only. Do not call tools."
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
            fh.write(prompt)
            qf = fh.name
        try:
            argv = hermes_argv() + ["chat", "-Q", "--query-file", qf, "--max-turns", "1", "--ignore-rules", "--source", "tool", "-t", "safe"]
            m = model or self.model
            if m:
                argv += ["-m", m]
            res = run(argv, os.getcwd(), env=hermes_child_env(os.getcwd()), timeout=self.timeout)
        finally:
            try:
                os.unlink(qf)
            except OSError:
                pass
        if not res.ok:
            raise BranchpointError(f"hermes chat failed (rc={res.rc}): {(res.err or res.out)[-300:]}")
        text = _SESSION_LINE.sub("", res.out.strip()).strip()
        tin, tout = estimate_tokens(prompt), estimate_tokens(text)
        self.meter.charge(tin + tout)
        return LLMReply(text, tin, tout, None, m or "")


@dataclass
class FakeLLM:
    """Deterministic backend: ``script`` maps a purpose (or a callable) to reply text."""

    script: dict[str, Any] = field(default_factory=dict)
    meter: Meter = field(default_factory=Meter)
    calls: list[dict[str, str]] = field(default_factory=list)

    def with_meter(self, meter: Meter) -> "FakeLLM":
        clone = FakeLLM(self.script, meter, self.calls)
        return clone

    def complete(self, system: str, user: str, *, max_tokens: int = 3500, temperature: float | None = None,
                 model: str | None = None, purpose: str = "") -> LLMReply:
        self.meter.check()
        self.calls.append({"purpose": purpose, "system": system, "user": user, "model": model or ""})
        handler = self.script.get(purpose)
        if handler is None:
            raise BranchpointError(f"FakeLLM has no reply scripted for purpose {purpose!r}")
        raw = handler(system, user) if callable(handler) else handler
        if isinstance(raw, list):  # consecutive replies for repeated calls
            idx = sum(1 for c in self.calls if c["purpose"] == purpose) - 1
            raw = raw[min(idx, len(raw) - 1)]
        text = str(raw)
        reply = LLMReply(text, estimate_tokens(system + user), estimate_tokens(text), None, model or "fake")
        self.meter.charge(reply.tokens)
        return reply


LLMFactory = Callable[[Meter], LLM]
