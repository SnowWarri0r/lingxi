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
from lingxi.conversation.prompt_assembly import (
    build_style_preamble,
    pick_prefill,
)
from lingxi.fewshot.models import AnnotationTurn, FewShotSample
from lingxi.fewshot.retriever import FewShotRetriever
from lingxi.fewshot.seeds_loader import load_seeds
from lingxi.fewshot.store import AnnotationStore, FewShotStore
from lingxi.memory.manager import MemoryManager
from lingxi.persona.models import EmotionState, PersonaConfig
from lingxi.persona.prompt_builder import PromptBuilder
from lingxi.planning.executor import ActionExecutor
from lingxi.planning.planner import Planner
from lingxi.planning.scheduler import Scheduler
from lingxi.providers.base import LLMProvider
from lingxi.providers.embedding import EmbeddingProvider
from lingxi.temporal.tracker import InteractionTracker
from lingxi.temporal.relationship import RelationshipEvaluator

# Inner life is optional — keep the engine usable without it
try:
    from lingxi.inner_life.store import InnerLifeStore
    from lingxi.inner_life.agenda import AgendaEngine
    from lingxi.inner_life.subjective import SubjectiveLayer
except ImportError:
    InnerLifeStore = None  # type: ignore
    AgendaEngine = None  # type: ignore
    SubjectiveLayer = None  # type: ignore


@dataclass
class StreamEvent:
    """A typed event from the streaming response."""

    type: str  # "chunk", "mood", "memory_write", "plan_update", "done"
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


class ConversationEngine:
    """Orchestrates persona-aware conversation with memory and planning."""

    def __init__(
        self,
        persona: PersonaConfig,
        llm_provider: LLMProvider,
        memory_manager: MemoryManager,
        planner: Planner | None = None,
        context_assembler: ContextAssembler | None = None,
        interaction_tracker: InteractionTracker | None = None,
        relationship_evaluator: RelationshipEvaluator | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        inner_life_store=None,
        agenda_engine=None,
        subjective_layer=None,
        fewshot_store: FewShotStore | None = None,
        annotation_store: AnnotationStore | None = None,
        fewshot_retriever: FewShotRetriever | None = None,
    ):
        self.persona = persona
        self.llm = llm_provider
        self.memory = memory_manager
        self.inner_life_store = inner_life_store
        self.agenda_engine = agenda_engine
        self.subjective_layer = subjective_layer
        if embedding_provider is not None:
            self.memory.set_embedding_provider(embedding_provider)
        self.prompt_builder = PromptBuilder(persona)
        self.planner = planner or Planner(llm_provider, persona)
        self.context_assembler = context_assembler or ContextAssembler()
        self.scheduler = Scheduler()
        self.executor = ActionExecutor(memory_manager)
        self.interaction_tracker = interaction_tracker
        self.relationship_evaluator = (
            relationship_evaluator or RelationshipEvaluator(persona, llm_provider)
        )

        self._current_mood: str = persona.emotional_baseline.default_mood
        self._emotion_state: EmotionState = EmotionState.from_baseline(persona.emotional_baseline)
        self._relationship_level: int = 1
        self._current_recipient_key: str | None = None
        self._last_response_text: str = ""

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
        self.biography_addenda_store: "BiographyAddendaStore | None" = None
        self._last_biography_hit: bool = False

        # Two-call (think → compress) — separate provider for the compress step
        # if the persona enables it. Built lazily on first use.
        self._compress_llm: LLMProvider | None = None
        self._last_biography_hits: list = []
        self._last_user_input: str = ""
        self._last_fewshots: list = []
        self._last_is_heavy_topic: bool = False

        # Initialize
        self.memory.set_llm_provider(llm_provider)
        self.planner.initialize_goals()

    async def _prepare_turn(
        self,
        user_input: str,
        images: list[dict] | None = None,
        channel: str | None = None,
        recipient_id: str | None = None,
    ) -> tuple[str, list[dict]]:
        """Common setup for a conversation turn. Returns (system_prompt, messages).

        Args:
            user_input: The user's text message.
            images: Optional list of image dicts, each:
                {"media_type": "image/png", "data": "<base64>"}
            channel: The channel name ("feishu", "web", "cli").
            recipient_id: The recipient identifier in that channel.
        """
        # Compute last-interaction time BEFORE recording this interaction
        last_interaction_time: datetime | None = None
        recipient_key = f"{channel}:{recipient_id}" if channel and recipient_id else None

        # Stash for use in _process_response (sync method, can't get from args)
        self._current_recipient_key = recipient_key

        # Switch short-term memory context to this recipient (restore buffer)
        if recipient_key:
            await self.memory.short_term.switch_recipient(recipient_key)

        if self.interaction_tracker and channel and recipient_id:
            rec = self.interaction_tracker.get_record(channel, recipient_id)
            if rec:
                last_interaction_time = rec.last_interaction
                self._relationship_level = rec.relationship_level
                if rec.emotion_dimensions:
                    self._emotion_state.dimensions = dict(rec.emotion_dimensions)
                if rec.emotion_last_decay:
                    self._emotion_state.last_decay_at = rec.emotion_last_decay
                if rec.emotion_narrative:
                    self._current_mood = rec.emotion_narrative
            self.interaction_tracker.record_interaction(channel, recipient_id)

        # Track input in short-term memory (with image indicator)
        memory_text = user_input
        if images:
            memory_text = f"[发送了{len(images)}张图片] {user_input}".strip()
        self.memory.add_turn("user", memory_text)

        # Mid-term layer: opportunistically compress aged turns (older than
        # the verbatim window) into one-line summaries before assembling
        # the context. Cheap because it only runs when there are pending
        # turns that haven't been summarized yet.
        try:
            await self.memory.compress_aged_turns(
                threshold_minutes=self.context_assembler.budget.verbatim_window_minutes
            )
        except Exception as e:
            print(f"[mid-term] compress failed: {e}")

        memory_context = await self.memory.assemble_context(
            user_input or "(图片消息)",
            recipient_key=recipient_key,
        )

        proactive_nudge = ""
        if self.planner:
            action_info = await self.planner.check_proactive_action(memory_context)
            if action_info and action_info.get("should_act"):
                proactive_nudge = (
                    f"\n[内心想法：你想要{action_info.get('content', '')}，"
                    f"因为{action_info.get('reason', '')}。如果合适，自然地融入你的回复中。]"
                )

        triggered = self.scheduler.check_event_triggers(user_input or "")
        if triggered:
            for t in triggered:
                proactive_nudge += f"\n[事件触发：{t}]"

        # Decay emotions toward baseline before rendering
        self._emotion_state.decay_toward_baseline(
            self.persona.emotional_baseline.baseline_dimensions or {"平静": 0.5, "好奇": 0.3}
        )
        self._emotion_state.narrative_label = self._current_mood

        # Load inner life state / subjective view / agenda (if available)
        inner_state = None
        subjective_view = None
        pending_agenda: list = []
        if self.inner_life_store is not None:
            try:
                inner_state = await self.inner_life_store.load_state()
            except Exception:
                pass
            if recipient_key and self.subjective_layer is not None:
                try:
                    subjective_view = await self.subjective_layer.get(recipient_key)
                except Exception:
                    pass
            if recipient_key and self.agenda_engine is not None:
                try:
                    pending_agenda = await self.agenda_engine.top_pending(recipient_key, limit=5)
                except Exception:
                    pass

        # Retrieve biographical events relevant to the current turn.
        # Query: user input (most directly topical); later we could also
        # fold in the previous inner_thought for continuity.
        biography_hits: list = []
        # Heavy-topic gate: if the user is sharing a serious life event
        # (death/illness/loss/breakup/firing), do NOT surface biography
        # at all. Even with a "don't grandstand" rule, having those
        # memories in the system prompt biases the model to share them.
        # Better not to put them in front of the model on those turns.
        is_heavy_topic = _looks_like_heavy_topic(user_input)
        if self.biography_retriever is not None and user_input.strip() and not is_heavy_topic:
            try:
                biography_hits = await self.biography_retriever.retrieve(
                    query=user_input, k=2, threshold=0.25,
                )
            except Exception as e:
                print(f"[biography] retrieve failed: {e}", flush=True)
                biography_hits = []
            if biography_hits:
                summary = "; ".join(f"{e.age}岁·{e.content[:18]}..." for e in biography_hits)
                print(f"[biography] hit {len(biography_hits)} for {user_input[:20]!r}: {summary}", flush=True)
            else:
                print(f"[biography] no hit for {user_input[:20]!r}", flush=True)
        elif is_heavy_topic:
            print(f"[biography] suppressed (heavy topic) for {user_input[:30]!r}", flush=True)
        self._last_biography_hit = bool(biography_hits)
        self._last_is_heavy_topic = is_heavy_topic

        # Stash for compress step (it needs user_input + biography hits)
        self._last_biography_hits = list(biography_hits)
        self._last_user_input = user_input

        prompt_mode = "think" if self.persona.compression.enabled else "single"
        system_prompt = self.prompt_builder.build_system_prompt(
            memory_context=memory_context,
            active_plans=self.planner.active_plans if self.planner else None,
            current_mood=self._current_mood,
            relationship_level=self._relationship_level,
            current_time=datetime.now(),
            last_interaction_time=last_interaction_time,
            emotion_state=self._emotion_state,
            inner_state=inner_state,
            subjective_view=subjective_view,
            pending_agenda=pending_agenda,
            biography_hits=biography_hits,
            mode=prompt_mode,
        )
        if proactive_nudge:
            system_prompt += f"\n\n## 当前内心活动{proactive_nudge}"

        messages = self.context_assembler.assemble_messages(memory_context)

        # If there are images, inject them into the last user message as multimodal blocks
        if images:
            # Rebuild last user message as content block array
            text_content = user_input or "（请看这张图）"
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
            blocks.append({"type": "text", "text": text_content})

            # Replace or append the last user message
            if messages and messages[-1]["role"] == "user":
                messages[-1] = {"role": "user", "content": blocks}
            else:
                messages.append({"role": "user", "content": blocks})

        # --- Few-shot retrieval (rendered as TEXT inside system prompt now,
        # not as user/assistant message pairs interleaved with real history).
        # The pair-injection approach caused the model to confabulate prior
        # user messages from fewshot context. Text-block treatment avoids
        # that while still anchoring voice. ---
        baseline_seeds = self._phase0_seed_fewshots()[:3]
        retrieved: list[FewShotSample] = []
        if self.fewshot_retriever is not None:
            try:
                query_text = self._last_inner_thought_for(recipient_key) or user_input
                retrieved = await self.fewshot_retriever.retrieve(
                    query_text=query_text,
                    recipient_key=recipient_key,
                    k=3,
                )
            except Exception:
                retrieved = []
        seed_fewshots = baseline_seeds + retrieved
        self._last_fewshots = seed_fewshots

        # Render fewshots as a text block and append to system prompt
        if seed_fewshots:
            voice_examples_block = self._render_fewshots_as_text(seed_fewshots[:5])
            if voice_examples_block:
                system_prompt = f"{system_prompt}\n\n{voice_examples_block}"

        # Attach style preamble to the last user message
        effective_style = self.persona.style
        if self._last_biography_hit:
            effective_style = effective_style.model_copy(
                update={"speech_max_chars": max(effective_style.speech_max_chars, 60)},
            )
        style_preamble = build_style_preamble(
            effective_style,
            voice_hint=self._persona_voice_hint(),
            biography_hit=self._last_biography_hit,
        )
        self._apply_style_preamble(messages, style_preamble)

        # Final message list = real chat history only (NO fewshot pair injection)
        return system_prompt, messages

    @staticmethod
    def _render_fewshots_as_text(samples: list[FewShotSample]) -> str:
        """Render fewshots as a TEXT block in the system prompt.

        Format: '场景：X / 你那时说："Y"'. The samples are voice anchors,
        explicitly NOT prior conversation turns the model should treat
        as if the user said them.
        """
        if not samples:
            return ""
        lines = [
            "## 你的说话样本（仅作语感参考——这些**不是**对方刚说的话，"
            "**不是**真实历史，只是几个'当年类似场景下你会怎么说'的示例）"
        ]
        for s in samples:
            ctx = (s.context_summary or "").strip()[:60]
            speech = (s.corrected_speech or "").strip()[:100]
            if not speech:
                continue
            if ctx:
                lines.append(f'- 场景："{ctx}" → 你那时说："{speech}"')
            else:
                lines.append(f'- 你那时说："{speech}"')
        return "\n".join(lines)

    def _phase0_seed_fewshots(self) -> list[FewShotSample]:
        """Hardcoded baseline seeds before Task 16's retriever lands.

        Covers three common AI-tone failure modes: over-eager agreement,
        cliché punchlines, and help-desk sign-offs.
        """
        return [
            FewShotSample(
                id="p0-1",
                inner_thought="",
                corrected_speech="哦？啥机械？",
                context_summary="用户提到他朋友的朋友也在做工业机械",
                tags=["好奇", "追问"],
                source="seed",
            ),
            FewShotSample(
                id="p0-2",
                inner_thought="",
                corrected_speech="也是，折腾完还要复盘 累。",
                context_summary="用户说刚开完一个长会",
                tags=["共鸣", "吐槽"],
                source="seed",
            ),
            FewShotSample(
                id="p0-3",
                inner_thought="",
                corrected_speech="嗯 早点睡。",
                context_summary="用户说困了要睡了",
                tags=["日常", "短"],
                source="seed",
            ),
        ]

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

    async def _run_think(self, system_prompt: str, messages: list[dict]) -> str:
        """Call the main LLM in think mode. Returns raw text (inner_thought + meta)."""
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
        from lingxi.conversation.prompt_assembly import DEFAULT_BLACKLIST

        # Use whatever _prepare_turn already retrieved (3 seeds + up to 3
        # context-relevant samples). Falling back to seeds-only when there's
        # no stash means cold-start still has voice anchors.
        seeds = list(self._last_fewshots) if self._last_fewshots else self._phase0_seed_fewshots()[:3]
        seeds = seeds[:5]

        style = self.persona.style
        blacklist_phrases = list(DEFAULT_BLACKLIST) + list(style.blacklist_phrases)
        max_chars = style.speech_max_chars
        if self._last_biography_hit:
            max_chars = max(max_chars, 60)
        # Heavy topic: hard-cap to 25 chars regardless of biography. Short +
        # present beats grandstanding empathy speech.
        if self._last_is_heavy_topic:
            max_chars = min(max_chars, 25)

        prompt_text = build_compress_prompt(
            persona_name=self.persona.name,
            voice_hint=self._persona_voice_hint(),
            inner_thought=inner_thought,
            user_message=user_input,
            fewshots_block=render_fewshots_for_compress(seeds),
            max_chars=max_chars,
            blacklist="、".join(blacklist_phrases),
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

    def _apply_style_preamble(self, messages: list[dict], preamble: str) -> None:
        """Prepend the preamble to the last user message's text.

        Handles both string content and multimodal block lists.
        """
        if not messages:
            return
        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                msg["content"] = preamble + content
                return
            if isinstance(content, list):
                # Find last text block; prepend preamble to it
                for block in reversed(content):
                    if isinstance(block, dict) and block.get("type") == "text":
                        block["text"] = preamble + block.get("text", "")
                        return
                # No text block? Add one
                content.append({"type": "text", "text": preamble})
                return

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
        rec.emotion_dimensions = dict(self._emotion_state.dimensions)
        rec.emotion_last_decay = self._emotion_state.last_decay_at
        rec.emotion_narrative = self._current_mood

        # Persist short-term buffer async
        try:
            import asyncio as _asyncio
            loop = _asyncio.get_running_loop()
            loop.create_task(self.memory.short_term.persist_current())

            # Mark agenda items as delivered if response seems to have mentioned them
            if self.agenda_engine and self._last_response_text and self._current_recipient_key:
                loop.create_task(
                    self._mark_agenda_delivered_if_mentioned(
                        self._current_recipient_key, self._last_response_text
                    )
                )

            # Reactive life: chat just happened → drop social_need + stamp
            if self.inner_life_store is not None:
                loop.create_task(self._touch_inner_life_chatted())
        except RuntimeError:
            pass

    async def _touch_inner_life_chatted(self) -> None:
        """Lightweight hook: inform the life simulator that a chat just occurred.

        Reduces social_need (she just talked to someone) and stamps last_chat_at
        so drift_dynamics can keep social_need depressed for the next ~hour.
        """
        if self.inner_life_store is None:
            return
        try:
            state = await self.inner_life_store.load_state()
            from datetime import datetime as _dt
            state.last_chat_at = _dt.now()
            state.social_need = max(0.1, state.social_need - 0.15)
            await self.inner_life_store.save_state(state)
        except Exception:
            # Never let life-state errors break the chat path
            pass

    async def _mark_agenda_delivered_if_mentioned(
        self, recipient_key: str, response_text: str
    ) -> None:
        """Heuristic: if Aria's response contains part of an agenda item, mark it delivered."""
        if self.agenda_engine is None:
            return
        try:
            pending = await self.agenda_engine.top_pending(recipient_key, limit=10)
        except Exception:
            return
        delivered_ids: list[str] = []
        response_lower = response_text.lower()
        for item in pending:
            # Crude overlap: if 40%+ of agenda content words appear in response
            content_words = set(item.content.lower().split())
            if not content_words:
                continue
            overlap = sum(1 for w in content_words if w in response_lower)
            if overlap / len(content_words) >= 0.4:
                delivered_ids.append(item.id)
        if delivered_ids:
            await self.agenda_engine.mark_delivered(recipient_key, delivered_ids)

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
        engine singleton state (`_current_recipient_key`, `_current_mood`,
        `_emotion_state`, short_term `_buffer`). Different recipients get
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
        system_prompt, messages = await self._prepare_turn(
            user_input, images, channel, recipient_id
        )

        if self.persona.compression.enabled:
            # Two-call: think (Sonnet) then compress (Haiku)
            think_raw = await self._run_think(system_prompt, messages)
            output = self._process_response(think_raw)
            inner_thought = output.inner_thought or output.speech
            speech = await self._run_compress(inner_thought, user_input)
            output.speech = speech
            output.inner_thought = inner_thought
        else:
            prefill = pick_prefill(self.persona.style)
            result = await self.llm.complete(
                messages=messages,
                system=system_prompt,
                temperature=self.persona.sampling.temperature,
                top_p=self.persona.sampling.top_p,
                prefill=prefill,
            )
            output = self._process_response(result.content)
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
        system_prompt, messages = await self._prepare_turn(
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
        from lingxi.conversation.output_schema import META_DELIMITER

        system_prompt, messages = await self._prepare_turn(
            user_input, images, channel, recipient_id
        )

        if self.persona.compression.enabled:
            # Two-call streaming: think (non-stream) then compress (stream)
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
                cleaned = "嗯 我刚刚走神了一下，你再说一遍？"

            output = output_pre
            output.speech = cleaned
            output.inner_thought = inner_thought
        else:
            prefill = pick_prefill(self.persona.style)

            full_response = ""
            delimiter_seen = False
            tail = ""
            tail_size = len(META_DELIMITER) - 1

            async for chunk in self.llm.complete_stream(
                messages=messages,
                system=system_prompt,
                temperature=self.persona.sampling.temperature,
                top_p=self.persona.sampling.top_p,
                prefill=prefill,
            ):
                if not chunk.content:
                    continue
                full_response += chunk.content

                if delimiter_seen:
                    continue

                combined = tail + chunk.content
                idx = combined.find(META_DELIMITER)
                if idx == -1:
                    if len(combined) > tail_size:
                        emit = combined[:-tail_size] if tail_size > 0 else combined
                        if emit:
                            yield StreamEvent("chunk", emit)
                        tail = combined[-tail_size:] if tail_size > 0 else ""
                    else:
                        tail = combined
                else:
                    before = combined[:idx]
                    if before:
                        yield StreamEvent("chunk", before)
                    delimiter_seen = True
                    tail = ""

            if not delimiter_seen and tail:
                yield StreamEvent("chunk", tail)

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

        # Apply memory writes (scoped to current recipient). Track tasks so
        # `end_session()` (or process shutdown) can flush before exit —
        # without this, embedding/Chroma writes that haven't completed get
        # silently dropped on quick shutdowns or back-to-back consolidate.
        for content in output.memory_writes:
            try:
                loop = asyncio.get_running_loop()
                task = loop.create_task(
                    self.memory.add_fact(
                        content,
                        importance=0.7,
                        recipient_key=self._current_recipient_key,
                    )
                )
                self._pending_memory_tasks.add(task)
                task.add_done_callback(self._pending_memory_tasks.discard)
            except RuntimeError:
                pass

        # Apply mood label update
        if output.mood_label:
            self._current_mood = output.mood_label

        # Apply emotion dimension deltas
        if output.emotion_deltas:
            self._emotion_state.apply_deltas(
                output.emotion_deltas,
                volatility=self.persona.emotional_baseline.mood_volatility,
            )

        # Apply plan updates
        for directive in output.plan_updates:
            if self.planner:
                self.planner.update_from_directive(directive)

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
                new_level = await self.relationship_evaluator.evaluate(rec, self.memory)
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
        if self.planner:
            await self.planner.save_to_disk(
                str(self.memory.data_dir / "plans.json")
            )
        if self.interaction_tracker:
            await self.interaction_tracker.save()
        return result

    async def load_state(self) -> None:
        """Load persisted state (memory, plans, interactions) from disk."""
        await self.memory.load()
        if self.planner:
            await self.planner.load_from_disk(
                str(self.memory.data_dir / "plans.json")
            )
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
