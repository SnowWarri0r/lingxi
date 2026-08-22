"""Replay a case and score it.

The point of replaying rather than storing finished messages: three of the
four fixes made on 2026-08-19 lived in the assembly layer (a lexicon entry,
a prompt reorder, a new orchestrator field). A harness that froze the
assembled messages would have tested none of them.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from lingxi.conversation.engine import RESPONDER_PRESETS, ConversationEngine
from lingxi.evals.case import Case
from lingxi.evals.detectors import evaluate
from lingxi.facts.retriever import FactRetriever
from lingxi.facts.store import FactStore
from lingxi.memory.manager import MemoryManager
from lingxi.persona.loader import load_persona
from lingxi.persona.models import PersonaConfig


@dataclass
class CaseScore:
    id: str
    verdict: str                      # PASS | FAIL | BROKEN
    fail_rate: float = 0.0
    pass_rate: float = 0.0
    samples: int = 0
    premise_ok: bool = True
    premise_error: str = ""
    replies: list[str] = field(default_factory=list)


def _apply_overrides(persona: PersonaConfig, overrides: dict | None) -> PersonaConfig:
    """Return a persona copy with top-level fields replaced.

    This is the seam for running candidates in bulk (spec §10): same case,
    different persona wording, one score each.
    """
    if not overrides:
        return persona
    return persona.model_copy(update=dict(overrides))


async def build_turn(
    case: Case, *, overrides: dict | None = None, llm=None,
) -> tuple[str, list[dict], PersonaConfig]:
    """Assemble the case's turn through the real pipeline.

    `llm` is the provider the orchestrator runs on; tests inject a stub.
    """
    persona = _apply_overrides(load_persona(case.persona), overrides)
    channel, _, recipient_id = case.recipient.partition(":")

    with tempfile.TemporaryDirectory(prefix="lingxi-eval-") as tmp:
        tmp_path = Path(tmp)
        clock = lambda: case.clock          # noqa: E731 — one-liner by design
        store = FactStore(tmp_path / "facts.db", clock=clock)
        await store.init()
        for fact in case.resolved_facts():
            await store.write(fact)
        retriever = FactRetriever(store, clock=clock)

        engine = ConversationEngine(
            persona=persona,
            llm_provider=llm or await _main_llm(),
            memory_manager=MemoryManager(data_dir=str(tmp_path / "mem")),
            fact_retriever=retriever,
        )
        # Seed the interaction tracker. Without one, _prepare_turn_v2 finds no
        # record and the reminder announces 「这是你们第一次对话」 above however
        # many turns of history the case froze — a register shift on every case.
        engine.interaction_tracker = _seeded_tracker(case, tmp_path)

        await engine.memory.short_term.switch_recipient(case.recipient)
        for role, content, ts in case.resolved_history():
            turn = engine.memory.add_turn(role, content)
            turn.timestamp = ts

        # Weather is a live external variable: PromptBuilder._weather_line
        # reads a process-global cache in lingxi.temporal.weather that is
        # populated whenever the live lingxi-feishu bot happens to be
        # running on this box. Same case, same clock, same facts — but a
        # different assembled prompt depending on bot uptime, which is
        # exactly the non-determinism this harness exists to eliminate.
        # Sunrise/sunset stays live on purpose: it is pure offline
        # computation from the frozen clock and the persona's location,
        # deterministic, and worth exercising end to end.
        engine.prompt_builder._weather_line = lambda _now: None

        system, messages = await engine._prepare_turn_v2(
            case.input, None, channel, recipient_id, now=case.clock,
        )
        # Assembled and returned INSIDE the TemporaryDirectory block: the
        # store file (and the fact_retriever reading from it) must still
        # exist while _prepare_turn_v2 runs its facts/renderer path.
        return system, messages, persona


def _seeded_tracker(case: Case, tmp_path: Path):
    """An InteractionTracker holding exactly what the case says about them."""
    from datetime import timedelta

    from lingxi.temporal.tracker import InteractionRecord, InteractionTracker

    channel, _, recipient_id = case.recipient.partition(":")
    tracker = InteractionTracker(tmp_path / "tracker")
    tracker._loaded = True
    last = case.last_interaction() or case.clock
    tracker._records[f"{channel}:{recipient_id}"] = InteractionRecord(
        recipient_id=recipient_id,
        channel=channel,
        last_interaction=last,
        first_interaction=case.clock - timedelta(days=case.acquaintance.days),
        total_turns=case.acquaintance.turns,
        relationship_level=case.acquaintance.relationship_level,
    )
    return tracker


async def _main_llm():
    """The orchestrator's provider, built through the app's own auth path.

    Constructing ClaudeProvider() directly succeeds but carries no key, so
    the first real run would fail on the orchestrator call rather than at
    setup. Reuse the resolution app.create_engine uses.
    """
    from lingxi.app import _build_auth_manager
    from lingxi.auth.models import AuthMethod
    from lingxi.providers.registry import ProviderRegistry
    from lingxi.utils.config import load_config

    config = load_config("config/default.yaml")
    ProviderRegistry.register_defaults()
    return await ProviderRegistry.create_llm_with_auth(
        "claude", auth_manager=_build_auth_manager(config),
        auth_method=AuthMethod("oauth_pkce"), model="claude-sonnet-4-6",
    )


def _check_premise(case: Case, system: str, messages: list[dict]) -> str:
    """Empty string when the premise holds, else why it does not."""
    last = messages[-1]["content"]
    if not isinstance(last, str):
        last = " ".join(
            b.get("text", "") for b in last if b.get("type") == "text")
    prompt = system + "\n" + last
    for needle in case.premise.prompt_contains:
        if needle not in prompt:
            return f"prompt_contains 未命中：{needle!r}"
    for needle in case.premise.prompt_lacks:
        if needle in prompt:
            return f"prompt_lacks 被命中：{needle!r}"
    return ""


def _make_default_sampler(persona: PersonaConfig):
    """Sample the real responder the way production does.

    The parameters have to match or the score measures the harness rather
    than the agent. Production streams at the persona's own temperature/top_p
    with no length cap; this ran at temperature=0.9, no top_p, and
    max_tokens=300 — and 300 truncates from the end, which is exactly where a
    trailing question lands.
    """
    async def _sampler(system: str, messages: list[dict], n: int) -> list[str]:
        return await _default_sampler(system, messages, n, persona)
    return _sampler


def _visible_prose(raw: str) -> str:
    """What the user would actually have seen.

    The responder emits a ===META=== block of directives after the prose.
    Detectors matching against the raw text are reading something no human
    ever sees, which can both miss a real hit and invent one.
    """
    from lingxi.conversation.output_schema import META_DELIMITER
    cut = raw.find(META_DELIMITER)
    return (raw if cut == -1 else raw[:cut]).strip()


async def _default_sampler(
    system: str, messages: list[dict], n: int, persona: PersonaConfig,
) -> list[str]:
    """Sample the real responder n times concurrently."""
    import openai

    preset = RESPONDER_PRESETS["deepseek"]
    client = openai.AsyncOpenAI(
        api_key=os.environ[preset["key_env"]], base_url=preset["base_url"])
    model = os.environ.get(preset["model_env"]) or preset["default_model"]

    payload = [{"role": "system", "content": system}]
    for m in messages:
        content = m["content"]
        if not isinstance(content, str):
            content = " ".join(
                b.get("text", "") for b in content if b.get("type") == "text")
        payload.append({"role": m["role"], "content": content})

    async def _one() -> str:
        resp = await client.chat.completions.create(
            model=model, messages=payload,
            temperature=persona.sampling.temperature,
            top_p=persona.sampling.top_p,
            # Generous rather than absent: production streams uncapped, and
            # anything this long is already far past a real reply.
            max_tokens=1500,
            extra_body=preset["extra_body"],
        )
        return _visible_prose(resp.choices[0].message.content or "")

    return list(await asyncio.gather(*[_one() for _ in range(n)]))


async def score_case(
    case: Case, *, overrides: dict | None = None, sampler=None, llm=None,
) -> CaseScore:
    """Replay one case and score it. `sampler` is injectable for offline tests."""
    system, messages, persona = await build_turn(
        case, overrides=overrides, llm=llm)

    error = _check_premise(case, system, messages)
    if error:
        # BROKEN before sampling — a case whose premise no longer holds must
        # never report PASS (silently stale coverage), and must not spend
        # money sampling a prompt that no longer matches what the case froze.
        return CaseScore(id=case.id, verdict="BROKEN",
                         premise_ok=False, premise_error=error)

    replies = await (sampler or _make_default_sampler(persona))(
        system, messages, case.samples)
    fails = sum(1 for r in replies if evaluate(case.detect.fail, r, persona))
    passes = (
        sum(1 for r in replies if evaluate(case.detect.passing, r, persona))
        if case.detect.passing else 0
    )
    n = len(replies) or 1
    fail_rate = fails / n
    return CaseScore(
        id=case.id,
        # pass_rate is observation only — it answers "did the fix also
        # produce right behaviour", not "did it pass".
        verdict="PASS" if fail_rate <= case.budget.max_fail_rate else "FAIL",
        fail_rate=fail_rate, pass_rate=passes / n, samples=len(replies),
        replies=replies,
    )
