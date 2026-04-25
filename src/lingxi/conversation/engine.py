"""Main conversation engine: the core loop tying all subsystems together."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncIterator

from lingxi.conversation.adapters import TextAdapter
from lingxi.conversation.context import ContextAssembler
from lingxi.conversation.output_schema import TurnOutput, parse_turn_output
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

        return system_prompt, messages

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

        result = await self.llm.complete(
            messages=messages,
            system=system_prompt,
        )

        output = self._process_response(result.content)
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

        full_response = ""
        async for chunk in self.llm.complete_stream(
            messages=messages,
            system=system_prompt,
        ):
            if chunk.content:
                full_response += chunk.content
                yield chunk.content

        output = self._process_response(full_response)
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

        full_response = ""
        delimiter_seen = False
        # Carry a small tail between chunks so we can detect the delimiter
        # even when it's split across chunk boundaries.
        tail = ""
        tail_size = len(META_DELIMITER) - 1  # enough to match if delimiter straddles

        async for chunk in self.llm.complete_stream(
            messages=messages,
            system=system_prompt,
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
        self._last_response_text = output.speech
        self.memory.add_turn("assistant", output.speech)
        self._persist_state(channel, recipient_id)

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
