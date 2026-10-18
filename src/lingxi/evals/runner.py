"""Replay a case and score it.

The point of replaying rather than storing finished messages: three of the
four fixes made on 2026-08-19 lived in the assembly layer (a lexicon entry,
a prompt reorder, a new orchestrator field). A harness that froze the
assembled messages would have tested none of them.
"""

from __future__ import annotations

import asyncio
import os
import random
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from lingxi.conversation.engine import RESPONDER_PRESETS, ConversationEngine
from lingxi.evals.case import Case
from lingxi.evals.detectors import evaluate_async
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

        if case.mode == "proactive":
            system, messages = await _proactive_turn(case, engine, tmp_path)
        else:
            system, messages = await engine._prepare_turn_v2(
                case.input, None, channel, recipient_id, now=case.clock,
            )
        # Assembled and returned INSIDE the TemporaryDirectory block: the
        # store file (and the fact_retriever reading from it) must still
        # exist while assembly runs its facts/renderer path.
        return system, messages, persona


async def _proactive_turn(case: Case, engine, tmp_path: Path):
    """Assemble an opener through the real scheduler.

    Goes through ProactiveScheduler.build_proactive_prompt rather than
    rebuilding the blocks here: the assembly is where this path's mistakes
    live, and a harness that reconstructed it would be scoring its own copy.
    """
    from lingxi.temporal.proactive import ProactiveConfig, ProactiveScheduler

    channel, _, recipient_id = case.recipient.partition(":")
    from lingxi.channels.outbound import ChannelRegistry

    scheduler = ProactiveScheduler(
        config=ProactiveConfig(),
        tracker=engine.interaction_tracker,
        channel_registry=ChannelRegistry(),
        engine=engine,
        data_dir=str(tmp_path / "proactive"),
        fact_retriever=engine.fact_retriever,
    )
    scheduler._recent_proactive = {
        case.recipient: case.sent_proactive_entries()
    }
    record = engine.interaction_tracker._records[case.recipient]
    silence = case.clock - (case.last_interaction() or case.clock)
    return await scheduler.build_proactive_prompt(record, silence, case.clock)


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
            # Production streams uncapped. With thinking on, reasoning counts
            # against this cap, and at 1500 a long think left no room for the
            # reply: 6 of 20 samples of one case came back empty and were
            # scored as her failing, a 10% → 30% "regression" that was the
            # harness's own.
            max_tokens=8000,
            extra_body=preset["extra_body"],
        )
        return _visible_prose(resp.choices[0].message.content or "")

    return list(await asyncio.gather(*[_one() for _ in range(n)]))


_SCAN_SUFFIX = ("\n\n（这次已经替你查好了，不用再搜。下面是查到的原文，"
                "按上面的要求改写成 items：）\n")


def _held_facts(case: Case):
    from lingxi.memory.consolidator import HeldFact
    return [HeldFact(f"c{i}", f.content, f.ts)
            for i, f in enumerate(case.resolved_facts())
            if f.subject.startswith("user:")]


async def _stage_replies(case: Case, persona: PersonaConfig, llm) -> tuple[str, list[str]]:
    """Prompt and outputs for the paths that write her life and memory rather
    than talk to him. Each goes through the module's own prompt builder."""
    n = case.samples
    if case.mode == "moment":
        from lingxi.planner.executor import build_moment_prompt, moment_system
        events = sorted(
            (f for f in case.resolved_facts()
             if f.subject == "aria" and f.type.value == "event"
             and case.clock - f.ts <= __import__("datetime").timedelta(hours=2)),
            key=lambda f: f.ts)[-3:]
        prompt = build_moment_prompt(case.plan.content, case.plan.time_window,
                                     events, case.clock)
        system = moment_system(persona)

        async def one():
            r = await llm.complete(messages=[{"role": "user", "content": prompt}],
                                   system=system, max_tokens=200, temperature=0.8)
            return r.content.strip()
    elif case.mode == "scan":
        from lingxi.world.fetcher import build_fetch_prompt
        from lingxi.utils import lenient_json
        prompt = (build_fetch_prompt(persona, case.clock.date()) + _SCAN_SUFFIX
                  + "\n".join(f"- {r}" for r in case.scan_raw))

        async def one():
            r = await llm.complete(messages=[{"role": "user", "content": prompt}],
                                   max_tokens=2000, temperature=0.7)
            text = r.content
            try:
                items = lenient_json.loads(text[text.find("{"):text.rfind("}") + 1])["items"]
                return "\n".join(str(it.get("voice", "")) for it in items)
            except Exception:
                return ""
    else:  # memory
        from lingxi.memory.consolidator import Turn, consolidate, real_people
        held = _held_facts(case)
        turns = [Turn(r, c, ts) for r, c, ts in case.resolved_history()]
        prompt = "\n".join(t.content for t in turns)

        async def one():
            ops = await consolidate(llm, turns, held, agent=persona.name,
                                    people=real_people(persona))
            live = {h.id: h.content for h in held}
            for i, o in enumerate(ops):
                if o.op in ("update", "retire"):
                    live.pop(o.target, None)
                if o.op in ("add", "update"):
                    live[f"new{i}"] = o.content
            return "\n".join(live.values())
    sem = asyncio.Semaphore(4)

    async def bounded():
        async with sem:
            return await one()
    return prompt, list(await asyncio.gather(*[bounded() for _ in range(n)]))


async def _judge_all(case: Case, replies: list[str], persona, judge):
    async def both(r):
        f = await evaluate_async(case.detect.fail, r, persona, judge)
        p = (await evaluate_async(case.detect.passing, r, persona, judge)
             if case.detect.passing else False)
        return f, p
    sem = asyncio.Semaphore(4)

    async def bounded(r):
        async with sem:
            return await both(r)
    return await asyncio.gather(*[bounded(r) for r in replies])


async def score_case(
    case: Case, *, overrides: dict | None = None, sampler=None, llm=None,
    judge=None,
) -> CaseScore:
    """Replay one case and score it. `sampler` is injectable for offline tests."""
    if case.mode in ("moment", "scan", "memory"):
        persona = _apply_overrides(load_persona(case.persona), overrides)
        main = llm or await _main_llm()
        prompt, replies = await _stage_replies(case, persona, main)
        for needle in case.premise.prompt_contains:
            if needle not in prompt:
                return CaseScore(id=case.id, verdict="BROKEN", premise_ok=False,
                                 premise_error=f"prompt_contains 未命中：{needle!r}")
        judged = await _judge_all(case, replies, persona, judge or main)
        n = len(replies) or 1
        fail_rate = sum(f for f, _ in judged) / n
        return CaseScore(
            id=case.id,
            verdict="PASS" if fail_rate <= case.budget.max_fail_rate else "FAIL",
            fail_rate=fail_rate, pass_rate=sum(p for _, p in judged) / n,
            samples=len(replies), replies=replies)
    # The opener path picks one of six message styles at random per send, so a
    # single assembly measures one style and reports it as the path's
    # behaviour. Reassemble per sample instead, seeded so two runs of the same
    # case draw the same styles and stay comparable. Assembly here costs no
    # model call — the blocks come from SQLite.
    per_sample = case.mode == "proactive"

    async def _assemble(i: int):
        if per_sample:
            random.seed(f"{case.id}:{i}")
        return await build_turn(case, overrides=overrides, llm=llm)

    system, messages, persona = await _assemble(0)

    error = _check_premise(case, system, messages)
    if error:
        # BROKEN before sampling — a case whose premise no longer holds must
        # never report PASS (silently stale coverage), and must not spend
        # money sampling a prompt that no longer matches what the case froze.
        return CaseScore(id=case.id, verdict="BROKEN",
                         premise_ok=False, premise_error=error)

    sample = sampler or _make_default_sampler(persona)
    if per_sample:
        replies = list(await sample(system, messages, 1))
        for i in range(1, case.samples):
            sys_i, msgs_i, _ = await _assemble(i)
            replies += list(await sample(sys_i, msgs_i, 1))
    else:
        replies = await sample(system, messages, case.samples)
    needs_judge = "judge" in case.detect.fail or (
        case.detect.passing is not None and "judge" in case.detect.passing)
    if needs_judge and judge is None:
        judge = llm or await _main_llm()
    judged = await _judge_all(case, replies, persona, judge)
    fails = sum(f for f, _ in judged)
    passes = sum(p for _, p in judged)
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
