"""Main conversation engine: the core loop tying all subsystems together."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator

if TYPE_CHECKING:
    from lingxi.persona.biography_addenda import BiographyAddendaStore
    from lingxi.persona.biography_retriever import BiographyRetriever

from lingxi.conversation.context import ContextAssembler
from lingxi.conversation.output_schema import TurnOutput, parse_turn_output
from lingxi.facts.models import FactType
from lingxi.facts.retriever import FactQuery
from lingxi.conversation.prompt_assembly import pick_prefill
from lingxi.fewshot.models import AnnotationTurn, FewShotSample
from lingxi.fewshot.retriever import FewShotRetriever
from lingxi.fewshot.seeds_loader import load_seeds
from lingxi.fewshot.store import AnnotationStore, FewShotStore
from lingxi.memory.manager import MemoryManager
from lingxi.persona.models import PersonaConfig
from lingxi.persona.prompt_builder import PromptBuilder
from lingxi.providers.base import LLMProvider
from lingxi.providers.embedding import EmbeddingProvider
from lingxi.temporal.tracker import InteractionTracker
from lingxi.temporal.relationship import RelationshipEvaluator


@dataclass
class StreamEvent:
    """A typed event from the streaming response."""

    type: str  # "chunk", "mood", "memory_write", "plan_update", "sticker", "done"
    content: str = ""

_DIRECTIVE_TAGS = (
    "mood_update",
    "memory_write",
    "plan_update",
    "emotion",
    "expression",
    "action",
    "inner",
)


# Heavy-topic markers — when ANY of these appear in the user message, we
# suppress biography injection and instruct compress to be ≤25 chars and
# absent of grandstanding ("我爸那会儿…"). The list is intentionally narrow
# to avoid false positives on light teasing / hypotheticals.
_HEAVY_TOPIC_MARKERS = (
    # death / loss
    "走了", "走的那年", "走的时候", "离开了", "去世", "过世", "不在了", "没了",
    "丧亲", "葬礼", "白事", "遗体", "尸体", "前年走", "去年走",
    # serious illness
    "癌", "肿瘤", "化疗", "放疗", "晚期",
    "脑梗", "脑干", "脑出血", "中风", "心梗", "心衰",
    "重病", "病危", "ICU", "插管", "植物人",
    # mental health crises
    "想死", "想不开", "活不下去", "自杀", "抑郁症",
    # relationship / job ruptures
    "离婚", "分手了", "出轨",
    "被裁", "失业了", "被辞退", "被开除",
)


def _looks_like_heavy_topic(user_input: str) -> bool:
    """Cheap substring check for heavy-emotion markers in user message.

    Designed to be conservative: prefers false-negatives (still grandstands
    sometimes) over false-positives (suppresses biography on jokes/teasing).
    """
    if not user_input:
        return False
    text = user_input
    return any(marker in text for marker in _HEAVY_TOPIC_MARKERS)


# Heavy markers IN BIOGRAPHY EVENT CONTENT — events containing these
# should only surface when the user query itself is on a matching topic.
# Without this filter, embedding similarity at threshold=0.25 could pull
# a "想过结束" biography fragment into a turn where the user is just
# venting / confronting / chatting, and Aria weaves it into her reply
# inappropriately (production: user said "你在胡说八道些什么", Aria
# referenced suicidal ideation memories from biography).
_HEAVY_BIO_CONTENT_MARKERS: tuple[str, ...] = (
    "想过结束", "想结束", "自杀", "活不下去", "崩溃过", "撑不下",
    "葬礼", "去世", "离世", "病重", "癌", "重病",
    "丧", "失去过",
)


def _bio_event_is_heavy(event) -> bool:
    """True if a biography event's content carries heavy markers.

    Used to filter retrieved bio events: heavy events should only
    surface when the user query is itself heavy. Otherwise the model
    treats them as fresh material to share, which feels wrong on
    light/confrontational turns.
    """
    text = (getattr(event, "content", "") or "")
    return any(m in text for m in _HEAVY_BIO_CONTENT_MARKERS)


class ConversationEngine:
    """Orchestrates persona-aware conversation with memory and planning."""

    def __init__(
        self,
        persona: PersonaConfig,
        llm_provider: LLMProvider,
        memory_manager: MemoryManager,
        context_assembler: ContextAssembler | None = None,
        interaction_tracker: InteractionTracker | None = None,
        relationship_evaluator: RelationshipEvaluator | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        fewshot_store: FewShotStore | None = None,
        annotation_store: AnnotationStore | None = None,
        fewshot_retriever: FewShotRetriever | None = None,
        fact_retriever=None,
        life_writer=None,
        inference_writer=None,
        world_writer=None,
        user_statement_writer=None,
        core_memory_writer=None,
        plan_executor=None,
        sticker_store=None,
    ):
        self.persona = persona
        self.llm = llm_provider
        self.memory = memory_manager
        self.fact_retriever = fact_retriever
        self.life_writer = life_writer
        self.inference_writer = inference_writer
        self.world_writer = world_writer
        self.user_statement_writer = user_statement_writer
        self.core_memory_writer = core_memory_writer
        self.plan_executor = plan_executor
        self.sticker_store = sticker_store
        if embedding_provider is not None:
            self.memory.set_embedding_provider(embedding_provider)
        self.prompt_builder = PromptBuilder(persona)
        self.context_assembler = context_assembler or ContextAssembler()
        self.interaction_tracker = interaction_tracker
        self.relationship_evaluator = (
            relationship_evaluator or RelationshipEvaluator(persona, llm_provider)
        )

        self._relationship_level: int = 1
        self._current_recipient_key: str | None = None
        self._last_response_text: str = ""
        # Sticker the agent chose this turn, keyed by recipient_key (1/turn
        # cap). Per-recipient (not a singleton field) so concurrent turns for
        # different chats can't overwrite each other's choice. Reset at turn
        # start in _prepare_turn_v2; read at turn end to emit (wired by the
        # streaming path in a follow-on task).
        self._pending_stickers: dict[str, str | None] = {}
        # Candidates surfaced by search_stickers this turn, per recipient:
        # {recipient_key: {sticker_id: file_path}}. send_sticker can only send
        # an id the agent actually searched (it must read the candidate's
        # emotion/when_to_use first), and judges fit itself.
        self._sticker_candidates: dict[str, dict[str, str]] = {}

        # Per-recipient locks: serialize reactive turns for the SAME recipient
        # so two messages from user A can't interleave through the engine's
        # singleton mutable state (current_recipient_key / mood / emotion /
        # short_term active buffer). Cross-recipient turns can still proceed
        # in parallel — a lock is created on first use per recipient_key.
        import asyncio as _asyncio
        self._turn_locks: dict[str, _asyncio.Lock] = {}
        self._turn_locks_guard: _asyncio.Lock = _asyncio.Lock()

        # In-flight memory_write tasks. fire-and-forget add_fact tasks land
        # here so end_session() can await them before exit; otherwise quick
        # shutdowns lose pending Chroma writes.
        self._pending_memory_tasks: set[_asyncio.Task] = set()

        self.fewshot_store = fewshot_store
        self.annotation_store = annotation_store
        self.fewshot_retriever = fewshot_retriever
        self._recent_inner_thoughts: dict[str, str] = {}

        # Biography retriever is set up after construction via bootstrap_biography()
        # because embedding the events requires an embedder to be available first.
        self.biography_retriever: "BiographyRetriever | None" = None
        # CC-style LLM selector — pick events by judgment, not cosine similarity.
        # Embedding similarity at threshold 0.25 was too loose and leaked
        # heavy events into wrong turns. The selector uses Haiku to read a
        # full manifest and pick 0-2 GENUINELY relevant events.
        from lingxi.persona.biography_selector import BiographySelector
        self.biography_selector: "BiographySelector | None" = None
        self.biography_addenda_store: "BiographyAddendaStore | None" = None
        self._last_biography_hit: bool = False

        # Two-call (think → compress) — separate provider for the compress step
        # if the persona enables it. Built lazily on first use.
        self._compress_llm: LLMProvider | None = None
        self._responder_llm: LLMProvider | None = None
        self._last_biography_hits: list = []
        self._last_user_input: str = ""
        self._last_fewshots: list = []
        self._last_is_heavy_topic: bool = False

        # Initialize
        self.memory.set_llm_provider(llm_provider)

    async def _dispatch_memory_tool(self, name: str, args: dict, recipient_key: str) -> str:
        """Execute one MemGPT memory tool, scoped by recipient_key. Returns a
        string for the tool_result. Errors are returned (not raised) so the
        agent can recover."""
        from datetime import datetime
        from lingxi.brain.memory_tools import CORE_BLOCK_MAX_CHARS
        from lingxi.facts.models import Fact, FactType, Source
        from lingxi.facts.retriever import FactQuery

        def _subject_for(scope: str) -> str:
            if scope == "self":
                return "aria"
            if scope == "world":
                return "world"
            return f"user:{recipient_key}"

        try:
            if name == "archival_memory_search":
                scope = args.get("scope", "user")
                subject = _subject_for(scope)
                facts = await self.fact_retriever.fetch(FactQuery(
                    subject=subject, semantic=args.get("query"), limit=5))
                if not facts:
                    return "（没找到相关记忆）"
                return "\n".join(
                    f"- [{f.ts.strftime('%m-%d')}] {f.content}" for f in facts)

            if name == "archival_memory_insert":
                scope = args.get("scope", "user")
                subject = _subject_for(scope)
                writer = self.inference_writer if scope == "self" else self.user_statement_writer
                if writer is None:
                    return "（写入未启用）"
                await writer.write(
                    subject=subject, content=args["content"], type=FactType.PATTERN,
                    source=writer.ALLOWED_SOURCE, ts=datetime.now(),
                    importance=args.get("importance"))
                return "inserted"

            if name in ("core_memory_append", "core_memory_replace"):
                if self.core_memory_writer is None:
                    return "（核心记忆未启用）"
                block = args.get("block", "human")
                subject = "aria" if block == "persona" else f"user:{recipient_key}"
                current = await self.fact_retriever.get_core_block(subject)
                cur_text = current.content if current else ""
                if name == "core_memory_append":
                    new_text = (cur_text + "\n" + args["content"]).strip()
                else:
                    if args["old"] not in cur_text:
                        return "substring not found in core block"
                    new_text = cur_text.replace(args["old"], args["new"])
                if len(new_text) > CORE_BLOCK_MAX_CHARS:
                    return "core memory full, use core_memory_replace to condense"
                await self.core_memory_writer.write(
                    subject=subject, content=new_text, type=FactType.CORE,
                    source=Source.LLM_INFERRED, ts=datetime.now(),
                    supersedes=current.id if current else None)
                return "ok"

            if name == "search_stickers":
                if self.sticker_store is None:
                    return "（表情库未启用）"
                query = args.get("query", "")
                hits = await self.sticker_store.search(query, k=6)
                if not hits:
                    return "（没找到相关表情,这轮就别发了）"
                self._sticker_candidates[recipient_key] = {
                    h.id: h.file_path for h in hits}
                lines = [
                    f"[{h.id}] {h.caption}（{h.emotion}）— {h.when_to_use}"
                    for h in hits]
                return (
                    "候选表情(挑一张真的贴当下气氛的,用 send_sticker 发它的 id;"
                    "都不合适就别发):\n" + "\n".join(lines))

            if name == "send_sticker":
                if self._pending_stickers.get(recipient_key) is not None:
                    return "本轮已经发过一张表情了"
                cands = self._sticker_candidates.get(recipient_key, {})
                sid = args.get("sticker_id", "")
                if sid not in cands:
                    return "这个 id 不在候选里,先用 search_stickers 看看有哪些,再发里面的 id"
                self._pending_stickers[recipient_key] = cands[sid]
                return "好,这张会发出去"

            if name == "conversation_search":
                turns = await self.memory.short_term.snapshot_for_recipient(recipient_key)
                q = args.get("query", "")
                hits = [t for t in turns if q in (t.content or "")][:8]
                if not hits:
                    return "（最近对话里没找到）"
                return "\n".join(
                    f"- [{t.timestamp.strftime('%m-%d %H:%M')}] "
                    f"{'对方' if t.role == 'user' else '我'}: {t.content[:80]}"
                    for t in hits)

            return f"unknown tool: {name}"
        except Exception as e:
            return f"tool error: {e}"

    async def _generate_with_tools(
        self,
        system_prompt: str,
        messages: list[dict],
        *,
        recipient_key: str,
        prefill: str = "",
        purpose: str = "chat_full",
    ) -> str:
        """Agentic generation loop: the model may call memory tools mid-turn.
        Returns the final user-facing text. Caps iterations to avoid runaway."""
        from lingxi.brain.memory_tools import MEMORY_TOOLS
        MAX_TOOL_ITERS = 5
        msgs = list(messages)
        iters = 0
        while True:
            forced = iters >= MAX_TOOL_ITERS
            tool_choice = {"type": "none"} if forced else {"type": "auto"}
            result = await self.llm.complete(
                messages=msgs,
                system=system_prompt,
                temperature=self.persona.sampling.temperature,
                top_p=self.persona.sampling.top_p,
                prefill=prefill if iters == 0 else "",
                tools=MEMORY_TOOLS,
                tool_choice=tool_choice,
                _debug_purpose=purpose,
            )
            if forced or result.finish_reason != "tool_use" or not result.tool_calls:
                return result.content
            msgs.append({"role": "assistant", "content": result.raw_content_blocks})
            tool_results = []
            for call in result.tool_calls:
                out = await self._dispatch_memory_tool(
                    call["name"], call.get("input", {}), recipient_key)
                tool_results.append({
                    "type": "tool_result", "tool_use_id": call["id"], "content": out})
            msgs.append({"role": "user", "content": tool_results})
            iters += 1

    @staticmethod
    def _build_user_message(user_input: str, images: list[dict] | None) -> dict:
        """Build the current-turn user message. With images, attach them as
        Anthropic multimodal blocks; the text block falls back to a non-empty
        caption so image-only messages never produce empty content (which the
        API rejects with 400 'user messages must have non-empty content')."""
        if not images:
            return {"role": "user", "content": user_input}
        blocks: list[dict] = []
        for img in images:
            blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": img.get("media_type", "image/png"),
                    "data": img["data"],
                },
            })
        blocks.append({"type": "text", "text": user_input or "（请看这张图）"})
        return {"role": "user", "content": blocks}

    async def _prepare_turn_v2(
        self,
        user_input: str,
        images: list[dict] | None,
        channel: str | None,
        recipient_id: str | None,
    ) -> tuple[str, list[dict]]:
        """New chat-prep path: orchestrator + renderer.

        Replaces _prepare_turn when self.fact_retriever is wired. Old
        path remains as fallback so we can A/B-compare.
        """
        from lingxi.brain.orchestrator import StateDigest, decide
        from lingxi.brain.renderer import render_dynamic_blocks
        from lingxi.persona.prompt_builder import build_persona_block

        recipient_key = f"{channel}:{recipient_id}" if recipient_id else "_anon"

        # Per-recipient state setup: restore this recipient's short-term
        # buffer + relationship level, and record the interaction. (Emotion
        # state was stripped — pure GA, the agent's state IS its memory stream.)
        self._current_recipient_key = recipient_key
        # Fresh turn: clear any sticker the previous turn selected for this
        # recipient.
        self._pending_stickers[recipient_key] = None
        self._sticker_candidates[recipient_key] = {}
        if recipient_key:
            await self.memory.short_term.switch_recipient(recipient_key)

        # Capture the previous interaction time BEFORE record_interaction
        # overwrites it to now — the time-awareness reminder uses it for the
        # "how long since we last talked" delta.
        last_interaction_time: datetime | None = None
        if self.interaction_tracker and channel and recipient_id:
            rec = self.interaction_tracker.get_record(channel, recipient_id)
            if rec:
                self._relationship_level = rec.relationship_level
                last_interaction_time = rec.last_interaction
            self.interaction_tracker.record_interaction(channel, recipient_id)

        # Text persisted to short-term for the user turn (with image marker).
        memory_text = user_input
        if images:
            memory_text = f"[发送了{len(images)}张图片] {user_input}".strip()

        # No facts layer wired (tests / minimal embeds): degrade to a plain
        # persona prompt + short-term dialog history, skipping the
        # orchestrator/renderer entirely.
        if self.fact_retriever is None:
            memory_context = await self.memory.assemble_context(
                query=user_input, recipient_key=recipient_key,
            )
            messages = self.context_assembler.assemble_messages(memory_context)
            self.memory.add_turn("user", memory_text)
            messages.append(self._build_user_message(user_input, images))
            return build_persona_block(self.persona), messages

        # 1. Build state digest purely from the facts memory stream.
        # "current activity" = latest aria.event fact; recent lived = next few.
        # No emotion/mood scalar — pure GA, state is the memory stream.
        aria_events = await self.fact_retriever.fetch(
            FactQuery(subject="aria", type=FactType.EVENT, limit=4)
        )
        digest = StateDigest(
            activity=aria_events[0].content[:60] if aria_events else "",
            mood="",
            last_lived=[f.content[:50] for f in aria_events[:3]],
        )

        # 2. Build catalog — filter user buckets to ONLY the current
        # recipient. Without this, Sonnet sees ALL users' patterns/jokes
        # and may query the wrong user's data (which then either pollutes
        # 【身边的事】 or leaks one user's private texture into another's chat).
        catalog_raw = await self.fact_retriever.catalog()
        cur_user_prefix = f"user:{recipient_key}."
        catalog = {
            k: v for k, v in catalog_raw.items()
            if not k.startswith("user:") or k.startswith(cur_user_prefix)
        }

        # 2.5. Pull recent dialog so orchestrator can capture topic-arc
        # (a 3-line user message often makes no sense without the last
        # 5-15 turns of context). Also pass previous thread_summary so
        # orchestrator keeps continuity across long sessions.
        memory_context = await self.memory.assemble_context(
            query=user_input, recipient_key=recipient_key,
        )
        messages = self.context_assembler.assemble_messages(memory_context)
        # Persist the user turn AFTER assembling history (so it isn't
        # duplicated this turn, but is available next turn).
        self.memory.add_turn("user", memory_text)
        prev_summary = self._thread_summaries.get(recipient_key, "") if hasattr(self, "_thread_summaries") else ""

        # 3. Orchestrator decides
        decision = await decide(
            self.llm, user_input, digest, catalog,
            history=messages,
            prev_thread_summary=prev_summary,
        )
        print(
            f"[brain] orch decision: register={decision.register} "
            f"engage={decision.engage_level:.1f} "
            f"queries={len(decision.fact_queries)} "
            f"anchor={decision.topic_anchor[:30]!r}",
            flush=True,
        )

        if decision.plan_conflict and self.plan_executor is not None:
            self.plan_executor.request_replan()

        # Persist thread_summary for next turn
        if decision.thread_summary:
            if not hasattr(self, "_thread_summaries"):
                self._thread_summaries = {}
            self._thread_summaries[recipient_key] = decision.thread_summary

        # 4. Render
        persona_block = build_persona_block(self.persona)
        # Migration writes user facts with subject=user:<full_channel:id>
        # (e.g. "user:feishu:oc_xxx"), so renderer needs the full
        # "feishu:oc_xxx" to build the matching subject. Don't strip the
        # channel prefix.
        dynamic_block = await render_dynamic_blocks(
            self.fact_retriever, decision, recipient_key=recipient_key,
        )
        system_prompt = persona_block + "\n\n" + dynamic_block

        # Voice anchors: retrieve real-corpus speech samples whose context
        # matches this turn, append as a "here's how you talk" block. This is
        # the anti-翻译腔 lever — the pure-GA refactor had cut it, leaving the
        # fewshot store loaded but never queried. Threshold-gated so off-topic
        # turns get nothing (e.g. an emo-only corpus won't surface on a happy
        # turn). corrected_speech is real human text; we anchor cadence only.
        if self.fewshot_retriever is not None:
            try:
                query_text = self._last_inner_thought_for(recipient_key) or user_input
                anchors = await self.fewshot_retriever.retrieve(
                    query_text=query_text, recipient_key=recipient_key,
                    k=4, threshold=0.5)
                block = self._render_fewshots_as_text(anchors)
                if block:
                    system_prompt = f"{system_prompt}\n\n{block}"
                    print(f"[fewshot] {len(anchors)} voice anchors injected "
                          f"(q={query_text[:20]!r})", flush=True)
            except Exception as e:
                print(f"[fewshot] retrieve failed (non-fatal): {e}")

        # Append the current user turn, prepending the per-turn focus reminder
        # (current real time + the utterance Aria just made). The pure-GA
        # refactor dropped this from the reactive path, which is why Aria lost
        # track of the time — restore it here.
        user_msg = self._build_user_message(user_input, images)
        focus = self._build_focus_reminder(last_interaction_time)
        if focus:
            if isinstance(user_msg["content"], str):
                user_msg["content"] = f"{focus}\n\n{user_msg['content']}"
            else:
                user_msg["content"].insert(0, {"type": "text", "text": focus})
        messages.append(user_msg)

        return system_prompt, messages

    def _render_fewshots_as_text(self, samples: list) -> str:
        """Render retrieved real-corpus lines as a voice-cadence reference block.

        We anchor on RHYTHM/口气 (碎句/省主语/语气词), not content — a text
        block ("here's how you talk") suits that better than user/assistant
        pairs, and avoids the model reading a thread-title context as a real
        user turn."""
        lines = [s.corrected_speech for s in samples if s.corrected_speech.strip()]
        if not lines:
            return ""
        body = "\n".join(f"- {ln}" for ln in lines)
        return (
            "## 你平时说话的语感（真实示例 —— 学这个**节奏/口气/碎句感**，"
            "别照搬内容，别当台词）\n"
            f"{body}"
        )

    def _build_focus_reminder(self, last_interaction_time: datetime | None) -> str | None:
        """Build the per-turn `<system-reminder>`: current real time + the
        thing Aria just said (so short user replies are read in context).

        Restores the time-awareness that the pure-GA prompt refactor dropped
        from the reactive path — without it Aria has no idea what time it is.
        """
        from lingxi.conversation.turn_focus import detect_last_assistant_turn

        laq: str | None = None
        las: str | None = None
        try:
            info = detect_last_assistant_turn(self.memory.short_term.get_history())
            if info is not None:
                text, is_question = info
                if is_question:
                    laq = text
                else:
                    las = text
        except Exception:
            pass
        return self.prompt_builder.build_turn_focus_reminder(
            current_time=datetime.now(),
            last_interaction_time=last_interaction_time,
            last_assistant_question=laq,
            last_assistant_statement=las,
        )

    def _last_inner_thought_for(self, recipient_key: str | None) -> str | None:
        """Cheapest signal for the retriever: the previous turn's inner_thought.

        Kept in-memory on the engine — no persistence needed since it's only
        used to build the next prompt.
        """
        if recipient_key is None:
            return None
        return self._recent_inner_thoughts.get(recipient_key)

    def _persona_voice_hint(self) -> str:
        """Derive a one-line voice descriptor from the persona YAML.

        Tone + top personality traits only. We deliberately do NOT inject
        verbal_habits here — those tend to be writerly self-descriptions
        like "喜欢用天文学隐喻" that, when handed to a small/cheap compress
        model as an instruction, get treated as "always do this," which
        produces pure AI-tone in casual IM chat. verbal_habits still
        influence the main think-call via prompt_builder.
        """
        p = self.persona
        parts: list[str] = []
        tone = p.speaking_style.tone.strip()
        if tone and tone != "neutral":
            parts.append(tone)
        top_traits = sorted(
            p.personality.traits, key=lambda t: t.intensity, reverse=True
        )[:2]
        if top_traits:
            parts.append("/".join(t.trait for t in top_traits))
        return "，".join(parts)

    # === Two-call (think → compress) helpers ==========================

    def _get_compress_llm(self) -> LLMProvider:
        """Lazy-build a separate provider for the compress call.

        Reuses the main LLM's auth (OAuth token / API key) but with a
        smaller/cheaper model (default Haiku) for low-latency compression.
        """
        if self._compress_llm is not None:
            return self._compress_llm
        cfg = self.persona.compression
        # Reuse same provider class with a different model
        from lingxi.providers.claude import ClaudeProvider
        if isinstance(self.llm, ClaudeProvider):
            self._compress_llm = ClaudeProvider(
                api_key=self.llm._api_key,
                model=cfg.compress_model,
            )
        else:
            # Non-Claude main provider — fall back to using main provider
            # (compression-as-a-task still works, just no latency win)
            self._compress_llm = self.llm
        return self._compress_llm

    def _responder_is_external(self) -> bool:
        """True when the chat reply is generated by a non-main responder
        (e.g. doubao) — a single coherent pass with no chat-time tools."""
        return getattr(self.persona, "responder", None) is not None \
            and self.persona.responder.provider not in ("", "main")

    def _get_responder_llm(self) -> LLMProvider:
        """Lazy-build the chat responder. provider="doubao" routes the
        user-facing voice to a Chinese-native model via the ARK
        OpenAI-compatible endpoint (reuses ARK_API_KEY). Anything else falls
        back to the main LLM. Degrades to main if ARK key / model missing."""
        if self._responder_llm is not None:
            return self._responder_llm
        rc = getattr(self.persona, "responder", None)
        if rc is not None and rc.provider == "doubao":
            import os
            from lingxi.providers.openai_provider import OpenAIProvider
            ark_key = os.environ.get("ARK_API_KEY", "")
            # The ARK endpoint id is infra config, not committed in the persona
            # yaml — read it from env (DOUBAO_RESPONDER_MODEL) with the yaml as
            # an optional override for non-secret setups.
            model = rc.model or os.environ.get("DOUBAO_RESPONDER_MODEL", "")
            if ark_key and model:
                self._responder_llm = OpenAIProvider(
                    api_key=ark_key,
                    model=model,
                    base_url="https://ark.cn-beijing.volces.com/api/v3",
                    # doubao-seed models reason for ~15s before emitting content,
                    # which kills perceived streaming (card sits blank, then the
                    # whole reply dumps in <1s). Disable thinking → first token in
                    # ~0.5s and the reply streams char-by-char. Harmless on
                    # non-reasoning endpoints (ARK ignores it).
                    extra_body={"thinking": {"type": "disabled"}},
                )
                return self._responder_llm
            print("[engine] responder=doubao but ARK_API_KEY/model missing "
                  "— falling back to main LLM")
        self._responder_llm = self.llm
        return self._responder_llm

    @staticmethod
    def _to_openai_messages(messages: list[dict]) -> list[dict]:
        """Convert Anthropic-format messages to OpenAI/ARK format.

        History turns are plain strings (unchanged). The current user turn may
        carry Anthropic multimodal blocks (image + text); doubao is multimodal
        too, it just wants the OpenAI shape: image blocks become image_url with
        a data: URI. This is what lets image turns ride the doubao responder
        instead of being split off to Claude."""
        out: list[dict] = []
        for m in messages:
            content = m.get("content")
            if not isinstance(content, list):
                out.append(m)
                continue
            parts: list[dict] = []
            for block in content:
                btype = block.get("type")
                if btype == "text":
                    parts.append({"type": "text", "text": block.get("text", "")})
                elif btype == "image":
                    src = block.get("source", {})
                    mt = src.get("media_type", "image/png")
                    data = src.get("data", "")
                    parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mt};base64,{data}"},
                    })
                # tool_use / tool_result don't occur on the no-tool single-pass path
            out.append({"role": m.get("role", "user"), "content": parts})
        return out

    async def _generate_single_pass(
        self, system_prompt: str, messages: list[dict], *, purpose: str = "chat",
    ) -> str:
        """One coherent pass on the responder model: full context in, prose +
        ===META=== out. NO function-calling — memory recall is front-loaded by
        the orchestrator, and memory writes / stickers ride the META block.
        Buffers the stream (META + clean_speech are global) and returns the
        full raw text. Retries once on empty generation (transient hiccup)."""
        llm = self._get_responder_llm()
        oai_messages = self._to_openai_messages(messages)
        full = ""
        for _attempt in range(2):
            full = ""
            try:
                async for chunk in llm.complete_stream(
                    messages=oai_messages,
                    system=system_prompt,
                    temperature=self.persona.sampling.temperature,
                    top_p=self.persona.sampling.top_p,
                    _debug_purpose=purpose,
                ):
                    if chunk.content:
                        full += chunk.content
            except Exception as e:
                print(f"[engine] single-pass generation failed: {e}")
                full = ""
            if full.strip():
                break
            print("[engine] empty generation — retrying once", flush=True)
        return full

    async def _resolve_sticker(self, query: str, recipient_key: str) -> None:
        """The persona put a sticker intent (a mood/emotion) in META — search
        the store and stage the best match for the turn-end emit. Takes the top
        FTS hit (not a random pick) so the sticker tracks the stated mood. No-op
        on no store / no match / one already staged this turn."""
        if not query or self.sticker_store is None:
            return
        if self._pending_stickers.get(recipient_key):
            return
        try:
            emb = self.memory.embedding_provider
            if emb is not None and await self.sticker_store.has_vectors():
                qv = await emb.embed(query)
                hits = await self.sticker_store.search_semantic(qv, k=6)
            else:
                hits = await self.sticker_store.search(query, k=6)
        except Exception as e:
            print(f"[sticker] search failed for {query!r}: {e}", flush=True)
            return
        if not hits:
            print(f"[sticker] no match for {query!r}", flush=True)
            return
        self._pending_stickers[recipient_key] = hits[0].file_path
        print(f"[sticker] {query!r} → {hits[0].file_path}", flush=True)

    async def _run_think(self, system_prompt: str, messages: list[dict]) -> str:
        """Call the main LLM in think mode. Returns raw text (inner_thought + meta).

        Defensive guard against an empty messages list — the Anthropic API
        rejects with `messages: at least one message is required`. We
        observed one transient case in production around concurrent
        relational extraction; root cause unclear, but if it happens
        again at least the error log says where instead of just bubbling
        a 400 from inside complete().
        """
        if not messages:
            raise RuntimeError(
                "_run_think received empty messages list — likely a "
                "race in context assembly or a recipient with no buffered "
                "history yet. system prompt was "
                f"{len(system_prompt)} chars."
            )
        result = await self.llm.complete(
            messages=messages,
            system=system_prompt,
            temperature=self.persona.sampling.temperature,
            top_p=self.persona.sampling.top_p,
            max_tokens=1500,
        )
        return result.content

    def _build_compress_input(self, inner_thought: str, user_input: str) -> tuple[str, list[dict]]:
        """Build the (system, messages) pair for the compress call."""
        from lingxi.conversation.prompts.compress import (
            build_compress_prompt,
            render_fewshots_for_compress,
        )

        # Pure GA: no fewshot voice anchors in the compress pass either.
        seeds: list = []

        style = self.persona.style
        max_chars = style.speech_max_chars

        # Pull Aria's most recent assistant turn from short_term as anchor
        # for compress. Without this, short user replies like '给我吃'
        # have no context (compress doesn't see history) and the model
        # misreads them. detect_last_assistant_turn skips trailing user
        # messages so we get the right anchor even after retries.
        previous_assistant_msg = ""
        try:
            from lingxi.conversation.turn_focus import detect_last_assistant_turn
            history = self.memory.short_term.get_history()
            turn_info = detect_last_assistant_turn(history)
            if turn_info is not None:
                previous_assistant_msg = turn_info[0]
        except Exception:
            pass

        prompt_text = build_compress_prompt(
            persona_name=self.persona.name,
            voice_hint=self._persona_voice_hint(),
            inner_thought=inner_thought,
            user_message=user_input,
            fewshots_block=render_fewshots_for_compress(seeds),
            max_chars=max_chars,
            previous_assistant_msg=previous_assistant_msg,
        )
        return prompt_text, [{"role": "user", "content": prompt_text}]

    async def _run_compress(self, inner_thought: str, user_input: str) -> str:
        """Call compress LLM and return the full speech (non-streaming)."""
        cfg = self.persona.compression
        llm = self._get_compress_llm()
        prompt_text, messages = self._build_compress_input(inner_thought, user_input)
        # Compress uses single user message; the persona prompt is in the
        # user-message text itself (intentional: compress should NOT see the
        # full persona/memory context, just the thought to translate).
        result = await llm.complete(
            messages=messages,
            max_tokens=cfg.compress_max_tokens,
            temperature=cfg.compress_temperature,
        )
        from lingxi.conversation.response_cleaner import clean_speech
        return clean_speech(result.content.strip())

    async def _run_compress_stream(self, inner_thought: str, user_input: str):
        """Stream compressed speech as text chunks."""
        cfg = self.persona.compression
        llm = self._get_compress_llm()
        _, messages = self._build_compress_input(inner_thought, user_input)
        async for chunk in llm.complete_stream(
            messages=messages,
            max_tokens=cfg.compress_max_tokens,
            temperature=cfg.compress_temperature,
        ):
            if chunk.content:
                yield chunk.content

    # === end two-call helpers ==========================================

    def _persist_state(self, channel: str | None, recipient_id: str | None) -> None:
        """Save current emotion state back to the InteractionRecord.

        Also schedules async persistence of short-term buffer and
        marks any agenda items that Aria appears to have delivered.
        """
        if not self.interaction_tracker or not channel or not recipient_id:
            return
        rec = self.interaction_tracker.get_record(channel, recipient_id)
        if rec is None:
            return

        # Persist short-term buffer async
        try:
            import asyncio as _asyncio
            loop = _asyncio.get_running_loop()
            loop.create_task(self.memory.short_term.persist_current())
        except RuntimeError:
            pass

    async def chat(
        self,
        user_input: str,
        images: list[dict] | None = None,
        channel: str | None = None,
        recipient_id: str | None = None,
    ) -> str:
        """Process a user message. Returns the speech text (for text channels).

        For full output use `chat_full()` which returns TurnOutput.
        """
        output = await self.chat_full(user_input, images, channel, recipient_id)
        return output.speech

    async def _lock_for_recipient(self, channel: str | None, recipient_id: str | None):
        """Get/create the per-recipient turn lock.

        Two reactive turns for the SAME recipient must serialize through
        engine singleton state (`_current_recipient_key`, short_term
        `_buffer`). Different recipients get
        different locks and proceed in parallel — but parallel reactive
        chats from different users still share singleton state, so callers
        should be aware: in practice the IM channel runs one chat at a
        time; this lock primarily protects against re-entry from the same
        user within an in-flight turn (e.g., user double-sends).
        """
        import asyncio as _asyncio
        key = f"{channel or '_'}:{recipient_id or '_'}"
        async with self._turn_locks_guard:
            lock = self._turn_locks.get(key)
            if lock is None:
                lock = _asyncio.Lock()
                self._turn_locks[key] = lock
        return lock

    async def chat_full(
        self,
        user_input: str,
        images: list[dict] | None = None,
        channel: str | None = None,
        recipient_id: str | None = None,
    ) -> TurnOutput:
        """Process a user message. Returns the complete TurnOutput."""
        lock = await self._lock_for_recipient(channel, recipient_id)
        async with lock:
            return await self._chat_full_locked(user_input, images, channel, recipient_id)

    async def _chat_full_locked(
        self,
        user_input: str,
        images: list[dict] | None,
        channel: str | None,
        recipient_id: str | None,
    ) -> TurnOutput:
        system_prompt, messages = await self._prepare_turn_v2(
            user_input, images, channel, recipient_id
        )

        rkey = self._current_recipient_key or "_anon"
        if self.persona.compression.enabled:
            # Two-call: think (with memory tools) then compress (Haiku, no tools)
            think_raw = await self._generate_with_tools(
                system_prompt, messages, recipient_key=rkey, purpose="think")
            output = self._process_response(think_raw)
            inner_thought = output.inner_thought or output.speech
            speech = await self._run_compress(inner_thought, user_input)
            output.speech = speech
            output.inner_thought = inner_thought
        elif self._responder_is_external():
            # Single coherent pass on the external responder (doubao), no tools.
            # Multimodal turns ride doubao too (blocks converted to OpenAI form).
            raw = await self._generate_single_pass(
                system_prompt, messages, purpose="chat_full_single_pass")
            output = self._process_response(raw)
        else:
            think_raw = await self._generate_with_tools(
                system_prompt, messages, recipient_key=rkey,
                prefill=pick_prefill(self.persona.style), purpose="chat_full")
            output = self._process_response(think_raw)
        output.turn_id = str(uuid.uuid4())

        # Persist AnnotationTurn so the user can annotate later
        if self.annotation_store is not None and channel and recipient_id:
            try:
                await self.annotation_store.record(AnnotationTurn(
                    turn_id=output.turn_id,
                    recipient_key=f"{channel}:{recipient_id}",
                    user_message=user_input,
                    inner_thought=output.inner_thought,
                    speech=output.speech,
                ))
            except Exception:
                # Don't let storage errors break the chat
                pass

        self._last_response_text = output.speech
        self.memory.add_turn("assistant", output.speech)
        self._persist_state(channel, recipient_id)
        return output

    async def chat_stream(
        self,
        user_input: str,
        images: list[dict] | None = None,
        channel: str | None = None,
        recipient_id: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream the persona's response as raw text chunks."""
        lock = await self._lock_for_recipient(channel, recipient_id)
        async with lock:
            async for chunk in self._chat_stream_locked(
                user_input, images, channel, recipient_id
            ):
                yield chunk

    async def _chat_stream_locked(
        self,
        user_input: str,
        images: list[dict] | None,
        channel: str | None,
        recipient_id: str | None,
    ) -> AsyncIterator[str]:
        system_prompt, messages = await self._prepare_turn_v2(
            user_input, images, channel, recipient_id
        )

        prefill = pick_prefill(self.persona.style)

        full_response = ""
        async for chunk in self.llm.complete_stream(
            messages=messages,
            system=system_prompt,
            temperature=self.persona.sampling.temperature,
            top_p=self.persona.sampling.top_p,
            prefill=prefill,
            _debug_purpose="chat_stream",
        ):
            if chunk.content:
                full_response += chunk.content
                yield chunk.content

        output = self._process_response(full_response)
        output.turn_id = str(uuid.uuid4())

        # Persist AnnotationTurn so the user can annotate later
        if self.annotation_store is not None and channel and recipient_id:
            try:
                await self.annotation_store.record(AnnotationTurn(
                    turn_id=output.turn_id,
                    recipient_key=f"{channel}:{recipient_id}",
                    user_message=user_input,
                    inner_thought=output.inner_thought,
                    speech=output.speech,
                ))
            except Exception:
                # Don't let storage errors break the chat
                pass

        self._last_response_text = output.speech
        self.memory.add_turn("assistant", output.speech)
        self._persist_state(channel, recipient_id)

    async def chat_stream_events(
        self,
        user_input: str,
        images: list[dict] | None = None,
        channel: str | None = None,
        recipient_id: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream the persona's response as typed events.

        Speech streams as `chunk` events until the META delimiter.
        After the delimiter, the metadata JSON is buffered and parsed at end.

        Yields:
            StreamEvent("chunk", text)         - visible speech fragment (streams)
            StreamEvent("expression", str)     - from meta.expression (at end)
            StreamEvent("action", str)         - from meta.action (at end)
            StreamEvent("mood", str)           - from meta.mood (at end)
            StreamEvent("memory_write", text)  - per item (at end)
            StreamEvent("plan_update", text)   - per item (at end)
            StreamEvent("sticker", path)       - sticker file to send (at end)
            StreamEvent("done", speech)        - final clean speech
        """
        lock = await self._lock_for_recipient(channel, recipient_id)
        async with lock:
            async for event in self._chat_stream_events_locked(
                user_input, images, channel, recipient_id
            ):
                yield event

    async def _chat_stream_events_locked(
        self,
        user_input: str,
        images: list[dict] | None,
        channel: str | None,
        recipient_id: str | None,
    ) -> AsyncIterator[StreamEvent]:
        system_prompt, messages = await self._prepare_turn_v2(
            user_input, images, channel, recipient_id
        )

        if self.persona.compression.enabled:
            # Two-call streaming: think (non-stream) then compress (stream).
            # NOTE: _run_think does NOT pass tools, so memory tools AND
            # send_sticker cannot fire on this branch. Aria runs with
            # compression disabled (the else branch below, which uses the
            # agentic tool loop), so stickers work on the live path. If you
            # ever enable compression on a sticker-using persona, wire tools
            # into _run_think or stickers/memory-edits will silently no-op here.
            try:
                think_raw = await self._run_think(system_prompt, messages)
                output_pre = self._process_response(think_raw)
                inner_thought = output_pre.inner_thought or output_pre.speech
            except Exception as e:
                # Think call dead → emit graceful fallback rather than
                # leaving the card stuck. User can b-correct.
                print(f"[engine] think call failed: {e}")
                from lingxi.conversation.output_schema import TurnOutput
                output = TurnOutput()
                output.speech = "嗯 网络抽了一下 你再发一次"
                output.turn_id = str(uuid.uuid4())
                if self.annotation_store is not None and channel and recipient_id:
                    try:
                        await self.annotation_store.record(AnnotationTurn(
                            turn_id=output.turn_id,
                            recipient_key=f"{channel}:{recipient_id}",
                            user_message=user_input,
                            inner_thought="",
                            speech=output.speech,
                        ))
                    except Exception:
                        pass
                yield StreamEvent("turn_id", output.turn_id)
                yield StreamEvent("done", output.speech)
                return

            # Surface a thinking preview so the UI can show "Aria 正在……"
            preview = inner_thought.strip().replace("\n", " ")[:30]
            if preview:
                yield StreamEvent("thinking", preview)

            # Accumulate the entire compressed reply WITHOUT emitting
            # per-chunk events. clean_speech() is global (paragraph collapse
            # + line-level narration detection), so partial cleaning during
            # stream would still flash em-dashes / `\n\n` to the user before
            # the final replace. Trade ~0.8s of typing animation for a
            # single render of the cleaned result. The "thinking" preview
            # above already covers "something's happening" UX.
            full_speech = ""
            compress_error: Exception | None = None
            try:
                async for chunk_text in self._run_compress_stream(inner_thought, user_input):
                    if chunk_text:
                        full_speech += chunk_text
            except Exception as e:
                compress_error = e
                print(f"[engine] compress stream failed, falling back: {e}")

            # Fallback 1: try non-streaming compress (same retry policy in provider)
            if not full_speech.strip() and compress_error is not None:
                try:
                    full_speech = await self._run_compress(inner_thought, user_input)
                except Exception as e:
                    print(f"[engine] compress non-stream also failed: {e}")
                    full_speech = ""

            # Fallback 2: still empty → emit a graceful filler so the card
            # doesn't sit on "💭 思考中..." forever. Persist anyway so user
            # can b-correct the failure case.
            from lingxi.conversation.response_cleaner import clean_speech
            cleaned = clean_speech(full_speech)
            if not cleaned.strip():
                # No "走神"/"再说一遍" — IM-anti-pattern (see memory). Plain hiccup.
                cleaned = "诶 我这边卡了一下"

            output = output_pre
            output.speech = cleaned
            output.inner_thought = inner_thought
        elif self._responder_is_external():
            # Single coherent pass on the external responder (e.g. doubao),
            # STREAMED: emit prose as chunk events so the card updates live.
            # Cut at the META delimiter so the metadata JSON never flashes;
            # memory writes ride that block (parsed from the full raw at end).
            # Image turns ride doubao too (blocks converted to OpenAI form).
            from lingxi.conversation.output_schema import META_DELIMITER
            llm = self._get_responder_llm()
            oai_messages = self._to_openai_messages(messages)
            raw = ""
            sent = 0  # chars of prose already emitted as chunks
            try:
                async for chunk in llm.complete_stream(
                    messages=oai_messages,
                    system=system_prompt,
                    temperature=self.persona.sampling.temperature,
                    top_p=self.persona.sampling.top_p,
                    _debug_purpose="chat_single_pass",
                ):
                    if not chunk.content:
                        continue
                    raw += chunk.content
                    cut = raw.find(META_DELIMITER)
                    prose = raw if cut == -1 else raw[:cut]
                    # Hold back a tail that could be a partial delimiter so we
                    # never stream a half-written "===MET" into the bubble.
                    emit_upto = (len(prose) if cut != -1
                                 else max(0, len(prose) - (len(META_DELIMITER) - 1)))
                    if emit_upto > sent:
                        yield StreamEvent("chunk", prose[sent:emit_upto])
                        sent = emit_upto
            except Exception as e:
                print(f"[engine] single-pass stream failed: {e}")
            output = self._process_response(raw)
            if not output.speech.strip():
                output.speech = "诶 我这边卡了一下"
            # No chat-time tools on this path, so a sticker the persona wanted
            # rides the META block: resolve the intent → stage a sticker file,
            # which the turn-end emit below sends.
            if output.sticker:
                await self._resolve_sticker(
                    output.sticker, self._current_recipient_key or "_anon")
        else:
            # Tool-use precludes token streaming (the model may call memory
            # tools before replying), so run the agentic loop to completion
            # then emit — same buffered shape as the compression branch above.
            rkey = self._current_recipient_key or "_anon"
            full_response = ""
            for _attempt in range(2):  # retry once — empty gen is usually a transient hiccup
                try:
                    full_response = await self._generate_with_tools(
                        system_prompt, messages, recipient_key=rkey,
                        prefill=pick_prefill(self.persona.style),
                        purpose="chat_stream_split")
                except Exception as e:
                    print(f"[engine] tool-loop generation failed: {e}")
                    full_response = ""
                if full_response.strip():
                    break
                print("[engine] empty generation — retrying once", flush=True)
            output = self._process_response(full_response)
            if not output.speech.strip():
                # Last resort. Do NOT fabricate inattention ("走神") or ask the
                # user to repeat — in IM the message is right there in history.
                # Own a plain technical hiccup instead.
                output.speech = "诶 我这边卡了一下"
        output.turn_id = str(uuid.uuid4())

        # Persist AnnotationTurn so the user can annotate later
        if self.annotation_store is not None and channel and recipient_id:
            try:
                await self.annotation_store.record(AnnotationTurn(
                    turn_id=output.turn_id,
                    recipient_key=f"{channel}:{recipient_id}",
                    user_message=user_input,
                    inner_thought=output.inner_thought,
                    speech=output.speech,
                ))
            except Exception:
                # Don't let storage errors break the chat
                pass

        self._last_response_text = output.speech
        self.memory.add_turn("assistant", output.speech)
        self._persist_state(channel, recipient_id)

        # Surface turn_id so annotation UIs can reference this turn
        if output.turn_id:
            yield StreamEvent("turn_id", output.turn_id)

        # Emit structured events from parsed metadata
        if output.expression:
            yield StreamEvent("expression", output.expression)
        if output.action:
            yield StreamEvent("action", output.action)
        if output.mood_label:
            yield StreamEvent("mood", output.mood_label)
        for mw in output.memory_writes:
            yield StreamEvent("memory_write", mw)
        for pu in output.plan_updates:
            yield StreamEvent("plan_update", pu)

        # Emit the sticker the agent chose this turn (if any), before `done` so
        # the channel sends it right after the speech bubble. Pop so a later
        # turn can't re-emit a stale path.
        rkey = self._current_recipient_key or "_anon"
        sticker_path = self._pending_stickers.pop(rkey, None)
        if sticker_path:
            yield StreamEvent("sticker", sticker_path)

        yield StreamEvent("done", output.speech)

    def _process_response(self, raw: str) -> TurnOutput:
        """Parse LLM raw output into TurnOutput and apply state changes.

        Returns the full TurnOutput (all modalities). Channels adapt it
        to their needs via adapters.
        """
        import asyncio

        output = parse_turn_output(raw)

        # Cache inner_thought for next turn's retrieval query
        if self._current_recipient_key and output.inner_thought:
            self._recent_inner_thoughts[self._current_recipient_key] = output.inner_thought

        # Apply memory writes → facts.db (subject=user:<recipient_key>).
        # These are durable facts the persona chose to remember about the
        # interlocutor; they surface back via the orchestrator/renderer's
        # 【你和他】 block. Fire-and-forget, tracked so end_session() can flush.
        if self.user_statement_writer is not None and self._current_recipient_key:
            from lingxi.facts.models import Source
            subject = f"user:{self._current_recipient_key}"
            for content in output.memory_writes:
                try:
                    loop = asyncio.get_running_loop()
                    task = loop.create_task(
                        self.user_statement_writer.write(
                            subject=subject,
                            content=content,
                            type=FactType.PATTERN,
                            source=Source.USER_STATED,
                            ts=datetime.now(),
                        )
                    )
                    self._pending_memory_tasks.add(task)
                    task.add_done_callback(self._pending_memory_tasks.discard)
                except RuntimeError:
                    pass

        self._last_output = output
        return output

    async def flush_pending_memory_writes(self) -> int:
        """Await all in-flight memory_write tasks. Call before consolidation
        or shutdown so add_fact() embeddings + Chroma writes don't get
        silently dropped on quick exit.
        """
        import asyncio as _asyncio
        pending = list(self._pending_memory_tasks)
        if not pending:
            return 0
        done, _unfinished = await _asyncio.wait(pending, timeout=10.0)
        for t in done:
            try:
                t.result()
            except Exception as e:
                print(f"[engine] memory_write task failed: {e}")
        return len(done)

    async def end_session(
        self,
        channel: str | None = None,
        recipient_id: str | None = None,
    ) -> dict:
        """End the current conversation session and consolidate memory."""
        # Flush in-flight memory writes BEFORE consolidating so the
        # consolidator sees the latest facts.
        await self.flush_pending_memory_writes()

        recipient_key = (
            f"{channel}:{recipient_id}" if channel and recipient_id else "_global"
        )
        result = await self.memory.consolidate_session(recipient_key=recipient_key)

        # Relationship evaluation (post-consolidation, fresh facts available)
        if self.interaction_tracker and channel and recipient_id:
            self.interaction_tracker.record_session_end(channel, recipient_id)
            rec = self.interaction_tracker.get_record(channel, recipient_id)
            if rec and self.relationship_evaluator:
                old_level = rec.relationship_level
                try:
                    new_level = await self.relationship_evaluator.evaluate(rec, self.memory)
                except Exception as e:
                    print(f"[relationship] eval failed (non-fatal): {e}")
                    new_level = old_level
                if new_level != old_level:
                    self.interaction_tracker.update_relationship_level(
                        channel, recipient_id, new_level
                    )
                    self._relationship_level = new_level
                    result["relationship_level_changed"] = True
                    result["old_relationship_level"] = old_level
                    result["new_relationship_level"] = new_level
                    print(
                        f"[relationship] {channel}:{recipient_id} "
                        f"level {old_level} → {new_level}"
                    )

        await self.memory.save()
        if self.interaction_tracker:
            await self.interaction_tracker.save()
        return result

    async def load_state(self) -> None:
        """Load persisted state (short-term memory, interactions) from disk."""
        await self.memory.load()
        if self.interaction_tracker:
            await self.interaction_tracker.load()

    async def bootstrap_biography(self) -> int:
        """Initialize the biography retriever.

        Merges seeded life_events from persona YAML with any addenda
        accumulated at runtime (self-added via reflection loop).
        Returns total number of events embedded.
        """
        from lingxi.persona.biography_addenda import BiographyAddendaStore
        from lingxi.persona.biography_retriever import BiographyRetriever

        embedder = self.memory.embedding_provider or (
            self.fewshot_retriever.embedder if self.fewshot_retriever else None
        )
        if embedder is None:
            return 0

        seeded = list(self.persona.biography.life_events)

        # Load runtime addenda from disk
        addenda_dir = Path(self.memory.data_dir) / "inner_life"
        self.biography_addenda_store = BiographyAddendaStore(data_dir=addenda_dir)
        addenda_entries = await self.biography_addenda_store.load()
        added_events = [entry.event for entry in addenda_entries]

        all_events = seeded + added_events
        if not all_events:
            return 0

        self.biography_retriever = BiographyRetriever(events=all_events, embedder=embedder)
        await self.biography_retriever.bootstrap()

        # Also build the LLM selector (CC-style). Uses Haiku for the
        # selection call (fast + cheap). Selector is the primary path
        # for biography retrieval; the retriever is kept around as a
        # fallback / for offline tooling but engine doesn't read from it.
        from lingxi.persona.biography_selector import BiographySelector
        selector_llm = self._get_compress_llm()  # Haiku
        self.biography_selector = BiographySelector(
            events=all_events,
            llm=selector_llm,
        )

        return len(all_events)

    async def add_biography_event(
        self,
        event,
        recipient_key: str | None = None,
        source: str = "reflection",
    ) -> None:
        """Append a self-accumulated LifeEvent: persist to addenda + index in retriever."""
        from lingxi.persona.biography_addenda import BiographyAddendaEntry

        if self.biography_retriever is None or self.biography_addenda_store is None:
            return

        await self.biography_addenda_store.append(
            BiographyAddendaEntry(
                event=event,
                source=source,
                recipient_key=recipient_key,
            )
        )
        await self.biography_retriever.append(event)
        # Keep the selector's event list in sync — engine reads from it
        if self.biography_selector is not None:
            self.biography_selector.append(event)

    async def bootstrap_fewshot_seeds(
        self,
        seeds_path: str | Path = "config/fewshot/seeds.yaml",
    ) -> int:
        """Populate the fewshot pool from seeds.yaml if not already present.

        Returns the number of samples added (0 if all already existed).
        """
        if self.fewshot_store is None:
            return 0

        p = Path(seeds_path)
        if not p.is_absolute():
            # Resolve relative to CWD
            p = Path.cwd() / p

        samples = load_seeds(p)

        # Deduplicate by id — Chroma will raise on duplicate ids
        added = 0
        for s in samples:
            try:
                embedding = await self._embed_for_fewshot(s)
                await self.fewshot_store.add(s, embedding=embedding)
                added += 1
            except Exception:
                # Already exists or chroma error; silently skip
                continue
        return added

    async def _embed_for_fewshot(self, sample: FewShotSample) -> list[float]:
        """Embed the inner_thought (or fall back to context_summary) via the LLM provider."""
        text = sample.inner_thought or sample.context_summary
        try:
            return await self.llm.embed(text)
        except NotImplementedError:
            # Provider doesn't support embeddings — use the MemoryManager's one
            if self.memory.embedding_provider is not None:
                return await self.memory.embedding_provider.embed(text)
            raise
