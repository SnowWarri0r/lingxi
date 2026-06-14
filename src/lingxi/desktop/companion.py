"""PetCompanion: turns coding-activity transitions into in-character pet lines.

Runs as a background task in the bot. Polls the ActivitySensor; on meaningful
transitions it generates a short line in the persona's voice (doubao) and
publishes it through the snapshot the /pet/state endpoint reads.

Throttled hard — a desktop companion that pipes up every poll is a nightmare.
Only fires on transitions, with a global cooldown + anti-repeat. The persona
engine + doubao voice live here in the bot; the pet window just shows the line.
"""

from __future__ import annotations

import asyncio
import time

from lingxi.desktop.activity_sensor import ActivitySignal, detect_activity


class PetCompanion:
    def __init__(
        self,
        engine,
        *,
        poll_secs: float = 12.0,
        min_gap_secs: float = 180.0,
    ) -> None:
        self.engine = engine
        self.poll_secs = poll_secs
        self.min_gap_secs = min_gap_secs

        self._sig = ActivitySignal("no_session", "", 0.0, "")
        self._prev: ActivitySignal | None = None
        self._active_since: float | None = None
        self._last_long_nudge = 0.0
        self._last_comment_ts = 0.0

        self._speech = ""
        self._speech_seq = 0
        self._recent_lines: list[str] = []

    # ---- read side (called from the /pet/state endpoint thread) ----------
    def snapshot(self) -> dict:
        return {
            "activity": self._sig.kind,
            "activity_detail": self._sig.detail,
            "speech": self._speech,
            "speech_seq": self._speech_seq,
        }

    # ---- run loop --------------------------------------------------------
    async def run(self) -> None:
        print("[pet-companion] started", flush=True)
        while True:
            try:
                await self._tick()
            except Exception as e:
                print(f"[pet-companion] tick failed: {e}", flush=True)
            await asyncio.sleep(self.poll_secs)

    async def _tick(self) -> None:
        now = time.time()
        sig = detect_activity(now=now)
        prev = self._prev
        self._sig = sig

        if sig.is_active and self._active_since is None:
            self._active_since = now

        situation = self._situation(prev, sig, now)

        self._prev = sig
        if not sig.is_active:
            self._active_since = None

        if situation and (now - self._last_comment_ts) >= self.min_gap_secs:
            line = await self._generate(situation)
            if line and line not in self._recent_lines:
                self._speech = line
                self._speech_seq += 1
                self._last_comment_ts = now
                self._recent_lines = (self._recent_lines + [line])[-8:]
                print(f"[pet-companion] {sig.kind} → {line}", flush=True)

    def _situation(
        self, prev: ActivitySignal | None, sig: ActivitySignal, now: float
    ) -> str | None:
        """Map a transition to a one-line description of what the user is doing,
        or None if nothing worth piping up about."""
        if prev is None:
            return None

        # 1. Finished a heads-down stretch → paused / back to a human turn.
        if prev.is_active and sig.kind == "awaiting_user":
            streak = (now - self._active_since) if self._active_since else 0.0
            if streak >= 45:
                what = prev.detail or "写代码"
                return f"主人刚才埋头忙了好一阵（{what}），这会儿停下来了"

        # 2. Long continuous grind → keep him company (own slower cooldown).
        if (
            sig.is_active
            and self._active_since
            and (now - self._active_since) >= 300
            and (now - self._last_long_nudge) >= 600
        ):
            self._last_long_nudge = now
            return f"主人已经连着忙了好一会儿了（{sig.detail or '一直在敲代码'}）"

        # 3. Came back after stepping away.
        if prev.kind == "idle" and sig.is_active:
            return "主人离开了一会儿，刚回到电脑前又开始忙了"

        return None

    async def _generate(self, situation: str) -> str:
        from lingxi.conversation.output_schema import META_DELIMITER
        from lingxi.conversation.response_cleaner import clean_speech
        from lingxi.persona.prompt_builder import build_persona_block

        system = build_persona_block(self.engine.persona)
        user = (
            f"【你在主人的电脑桌面上陪着他，不是在 IM 里。你此刻看到的情形："
            f"{situation}。】\n就这个情形，你会冒出来对他说的一句话是什么？"
            f"短，就一句，贴着这个情形，别问一串问题。"
        )
        try:
            llm = self.engine._get_responder_llm()
            msgs: list[dict] = [{"role": "user", "content": user}]
            if self.engine._responder_is_external():
                msgs = self.engine._to_openai_messages(msgs)
            result = await llm.complete(
                messages=msgs, system=system, max_tokens=120, temperature=0.95
            )
        except Exception as e:
            print(f"[pet-companion] gen failed: {e}", flush=True)
            return ""

        text = result.content.strip()
        speech = text.partition(META_DELIMITER)[0] if META_DELIMITER in text else text
        return clean_speech(speech.strip())
