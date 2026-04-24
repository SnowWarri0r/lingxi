"""Main conversation engine: the core loop tying all subsystems together."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator

from lingxi.conversation.adapters import TextAdapter
from lingxi.conversation.context import ContextAssembler
from lingxi.conversation.output_schema import TurnOutput, parse_turn_output
from lingxi.conversation.prompt_assembly import (
    build_style_preamble,
    pick_prefill,
    render_fewshots_as_messages,
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
from lingxi.providers.base import LLMProvider, StreamChunk
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

        self.fewshot_store = fewshot_store
        self.annotation_store = annotation_store
        self.fewshot_retriever = fewshot_retriever
        self._recent_inner_thoughts: dict[str, str] = {}

        # Biography retriever is set up after construction via bootstrap_biography()
        # because embedding the events requires an embedder to be available first.
        self.biography_retriever: "BiographyRetriever | None" = None
        self._last_biography_hit: bool = False

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
        if self.biography_retriever is not None and user_input.strip():
            try:
                biography_hits = await self.biography_retriever.retrieve(
                    query=user_input, k=3, threshold=0.55,
                )
            except Exception:
                biography_hits = []
        self._last_biography_hit = bool(biography_hits)

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

        # --- Dynamic few-shot (Task 16) ---
        # Spec §6.3: always anchor with seeds as baseline, then suffix with
        # retrieved user_correction/positive samples in the "most recent"
        # position for strongest LLM imitation.
        baseline_seeds = self._phase0_seed_fewshots()[:3]  # anchor
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
        # Seeds first (baseline), retrieved last (recency = strongest signal)
        seed_fewshots = baseline_seeds + retrieved
        few_shot_msgs = render_fewshots_as_messages(seed_fewshots)

        # Attach style preamble to the last user message
        # When biography hit: relax the length cap so Aria can actually
        # share a personal anecdote ("我也有过xxx...") instead of being
        # forced back into one-liner mode.
        effective_style = self.persona.style
        if self._last_biography_hit:
            effective_style = effective_style.model_copy(
                update={"speech_max_chars": max(120, effective_style.speech_max_chars)},
            )
        style_preamble = build_style_preamble(
            effective_style,
            voice_hint=self._persona_voice_hint(),
            biography_hit=self._last_biography_hit,
        )
        self._apply_style_preamble(messages, style_preamble)

        # Final message list = few-shot pairs + history (which already includes the user turn)
        final_messages = few_shot_msgs + messages
        return system_prompt, final_messages

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

        Pulls tone + top personality traits (+ 1 verbal habit if short)
        so the style preamble can keep Aria's voice from being flattened
        into generic WeChat register.
        """
        p = self.persona
        parts: list[str] = []
        tone = p.speaking_style.tone.strip()
        if tone and tone != "neutral":
            parts.append(tone)
        # Top 2 traits (above baseline intensity)
        top_traits = sorted(
            p.personality.traits, key=lambda t: t.intensity, reverse=True
        )[:2]
        if top_traits:
            parts.append("/".join(t.trait for t in top_traits))
        # One verbal habit for flavor (optional)
        if p.speaking_style.verbal_habits:
            parts.append(p.speaking_style.verbal_habits[0])
        return "，".join(parts)

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
        except RuntimeError:
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

    async def chat_full(
        self,
        user_input: str,
        images: list[dict] | None = None,
        channel: str | None = None,
        recipient_id: str | None = None,
    ) -> TurnOutput:
        """Process a user message. Returns the complete TurnOutput."""
        system_prompt, messages = await self._prepare_turn(
            user_input, images, channel, recipient_id
        )

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
        from lingxi.conversation.output_schema import META_DELIMITER

        system_prompt, messages = await self._prepare_turn(
            user_input, images, channel, recipient_id
        )

        prefill = pick_prefill(self.persona.style)

        full_response = ""
        delimiter_seen = False
        # Carry a small tail between chunks so we can detect the delimiter
        # even when it's split across chunk boundaries.
        tail = ""
        tail_size = len(META_DELIMITER) - 1  # enough to match if delimiter straddles

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
                # After delimiter, everything is metadata — don't stream
                continue

            combined = tail + chunk.content
            idx = combined.find(META_DELIMITER)
            if idx == -1:
                # Delimiter not yet seen; safely emit everything except the
                # trailing `tail_size` chars (they might be the start of delimiter)
                if len(combined) > tail_size:
                    emit = combined[:-tail_size] if tail_size > 0 else combined
                    if emit:
                        yield StreamEvent("chunk", emit)
                    tail = combined[-tail_size:] if tail_size > 0 else ""
                else:
                    tail = combined
            else:
                # Delimiter found — emit everything before it and stop streaming
                before = combined[:idx]
                if before:
                    yield StreamEvent("chunk", before)
                delimiter_seen = True
                tail = ""

        # If we never saw the delimiter, flush remaining tail as speech
        if not delimiter_seen and tail:
            yield StreamEvent("chunk", tail)

        # Parse full response, apply state changes
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

        # Apply memory writes (scoped to current recipient)
        for content in output.memory_writes:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(
                    self.memory.add_fact(
                        content,
                        importance=0.7,
                        recipient_key=self._current_recipient_key,
                    )
                )
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

    async def end_session(
        self,
        channel: str | None = None,
        recipient_id: str | None = None,
    ) -> dict:
        """End the current conversation session and consolidate memory."""
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
        """Initialize the biography retriever by embedding all life_events.

        Returns the number of events embedded (0 if biography empty or
        no embedder available).
        """
        events = self.persona.biography.life_events
        if not events:
            return 0
        embedder = self.memory.embedding_provider or (
            self.fewshot_retriever.embedder if self.fewshot_retriever else None
        )
        if embedder is None:
            return 0
        from lingxi.persona.biography_retriever import BiographyRetriever

        self.biography_retriever = BiographyRetriever(events=list(events), embedder=embedder)
        await self.biography_retriever.bootstrap()
        return len(events)

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
