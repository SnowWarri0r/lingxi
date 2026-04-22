"""Feishu (Lark) bot channel with WebSocket long connection + streaming card replies.

Uses lark-oapi SDK's WebSocket mode - no public IP needed.

Flow:
1. SDK connects to Feishu via WebSocket (outbound, auto-reconnect)
2. User sends message -> SDK dispatches event -> our handler
3. Create streaming card (CardKit v1) -> send to chat -> stream LLM chunks -> finish

Requires: FEISHU_APP_ID, FEISHU_APP_SECRET
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import threading
import time
import uuid

import httpx
import lark_oapi as lark
from lark_oapi.api.im.v1 import P2ImMessageReceiveV1

from lingxi.channels.outbound import ChannelRegistry, OutboundChannel
from lingxi.conversation.engine import ConversationEngine
from lingxi.inner_life.simulator import LifeSimulator
from lingxi.temporal.proactive import ProactiveConfig, ProactiveScheduler
from lingxi.temporal.reflection import ReflectionConfig, ReflectionLoop

FEISHU_BASE = "https://open.feishu.cn/open-apis"


def build_annotation_footer_elements(turn_id: str) -> list[dict]:
    """Three annotation buttons to append at the bottom of the reply card.

    👍 像 → positive
    👎 不像 → negative
    ✏️ 应该说 → opens a form for correction (stub — user can use /bad in CLI or POST API)
    """
    return [
        {"tag": "hr"},
        {
            "tag": "action",
            "actions": [
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "👍 像"},
                    "type": "default",
                    "value": {"action": "annotate_positive", "turn_id": turn_id},
                },
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "👎 不像"},
                    "type": "default",
                    "value": {"action": "annotate_negative", "turn_id": turn_id},
                },
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "✏️ 应该说"},
                    "type": "primary",
                    "value": {"action": "annotate_correction", "turn_id": turn_id},
                },
            ],
        },
    ]


_CARD_STREAMING_CONFIG = {
    "schema": "2.0",
    "config": {
        "streaming_mode": True,
        "streaming_config": {
            "print_frequency_ms": {"default": 50},
            "print_step": {"default": 2},
            "print_strategy": "fast",
        },
    },
    "body": {
        "elements": [
            {
                "tag": "markdown",
                "content": "💭 思考中...",
                "element_id": "md_stream",
            }
        ]
    },
}


class FeishuTokenManager:
    """Manages tenant_access_token with auto-refresh."""

    def __init__(self, app_id: str, app_secret: str):
        self.app_id = app_id
        self.app_secret = app_secret
        self._token: str = ""
        self._expires_at: float = 0

    def get_token(self) -> str:
        if self._token and time.time() < self._expires_at - 300:
            return self._token

        resp = httpx.post(
            f"{FEISHU_BASE}/auth/v3/tenant_access_token/internal",
            json={"app_id": self.app_id, "app_secret": self.app_secret},
        )
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"Feishu auth failed: {data}")

        self._token = data["tenant_access_token"]
        self._expires_at = time.time() + data.get("expire", 7200)
        return self._token

    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.get_token()}",
            "Content-Type": "application/json; charset=utf-8",
        }


class StreamingCardSender:
    """Manages creating and streaming updates to a Feishu card (async)."""

    def __init__(self, token_mgr: FeishuTokenManager, http: httpx.AsyncClient):
        self._token_mgr = token_mgr
        self._http = http
        self._card_id: str | None = None
        self._seq: int = 0

    async def create_card(self) -> str:
        headers = self._token_mgr.headers()
        resp = await self._http.post(
            f"{FEISHU_BASE}/cardkit/v1/cards",
            headers=headers,
            json={
                "type": "card_json",
                "data": json.dumps(_CARD_STREAMING_CONFIG, ensure_ascii=False),
            },
        )
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"Create card failed: {data}")
        self._card_id = data["data"]["card_id"]
        self._seq = 0
        return self._card_id

    async def send_to_chat(self, chat_id: str) -> str:
        headers = self._token_mgr.headers()
        resp = await self._http.post(
            f"{FEISHU_BASE}/im/v1/messages?receive_id_type=chat_id",
            headers=headers,
            json={
                "receive_id": chat_id,
                "msg_type": "interactive",
                "content": json.dumps({"type": "card", "data": {"card_id": self._card_id}}),
                "uuid": uuid.uuid4().hex,
            },
        )
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"Send card failed: {data}")
        return data["data"]["message_id"]

    async def update_content(self, text: str) -> None:
        if not self._card_id:
            return
        self._seq += 1
        headers = self._token_mgr.headers()
        await self._http.put(
            f"{FEISHU_BASE}/cardkit/v1/cards/{self._card_id}/elements/md_stream/content",
            headers=headers,
            json={"content": text, "sequence": self._seq},
        )

    async def finish(self, footer_elements: list[dict] | None = None) -> None:
        if not self._card_id:
            return

        if footer_elements:
            # Append footer elements (hr + action buttons) before disabling streaming.
            # Use the CardKit batch-update element endpoint to add each element.
            # We send a full-card update to append footer to the body elements.
            self._seq += 1
            headers = self._token_mgr.headers()
            try:
                await self._http.put(
                    f"{FEISHU_BASE}/cardkit/v1/cards/{self._card_id}/elements/batch_update",
                    headers=headers,
                    json={
                        "sequence": self._seq,
                        "operations": [
                            {
                                "action": "append",
                                "element": json.dumps(el, ensure_ascii=False),
                            }
                            for el in footer_elements
                        ],
                    },
                )
            except Exception as e:
                print(f"[feishu] footer append failed (non-fatal): {e}")

        self._seq += 1
        headers = self._token_mgr.headers()
        await self._http.patch(
            f"{FEISHU_BASE}/cardkit/v1/cards/{self._card_id}/settings",
            headers=headers,
            json={
                "settings": json.dumps({"config": {"streaming_mode": False}}),
                "sequence": self._seq,
            },
        )


class FeishuBot(OutboundChannel):
    """Feishu bot using WebSocket long connection (no public IP needed).

    Also implements OutboundChannel so the ProactiveScheduler can push
    messages via this bot.
    """

    def __init__(
        self,
        engine: ConversationEngine,
        app_id: str | None = None,
        app_secret: str | None = None,
        update_interval: float = 0.3,
        ignore_stale_seconds: float = 30,
        proactive_config: ProactiveConfig | None = None,
    ):
        self.engine = engine
        self.app_id = app_id or os.environ["FEISHU_APP_ID"]
        self.app_secret = app_secret or os.environ["FEISHU_APP_SECRET"]
        self.token_mgr = FeishuTokenManager(self.app_id, self.app_secret)
        self._update_interval = update_interval
        self._seen: set[str] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._startup_time_ms = int((time.time() - ignore_stale_seconds) * 1000)

        # Proactive scheduler (enabled by default)
        self._proactive_config = proactive_config or ProactiveConfig()
        self._channel_registry = ChannelRegistry()
        self._channel_registry.register(self)
        self._proactive_scheduler: ProactiveScheduler | None = None
        self._reflection_config = ReflectionConfig()
        self._reflection_loop: ReflectionLoop | None = None
        self._life_simulator: LifeSimulator | None = None

    @property
    def channel_name(self) -> str:
        return "feishu"

    async def send_message(self, recipient_id: str, text: str) -> None:
        """OutboundChannel implementation - send proactive message as a card."""
        try:
            await self._send_proactive_card(recipient_id, text)
        except Exception:
            # Fallback to plain text if card fails
            await self._send_text_async(recipient_id, text)

    async def _send_proactive_card(self, chat_id: str, text: str) -> None:
        """Send a message as a markdown card (visually consistent with chat replies)."""
        async with httpx.AsyncClient(timeout=30) as http:
            card = StreamingCardSender(self.token_mgr, http)
            await card.create_card()
            await card.send_to_chat(chat_id)
            await card.update_content(text)
            await card.finish()

    async def _send_card_or_text(self, chat_id: str, text: str) -> None:
        """Send as card, fallback to text on failure."""
        try:
            await self._send_proactive_card(chat_id, text)
        except Exception:
            await self._send_text_async(chat_id, text)

    def start(self) -> None:
        """Start the WebSocket long connection (blocking).

        The lark SDK runs its own event loop internally.
        We keep a reference to an asyncio loop for running async engine calls.
        """
        # Create a dedicated asyncio loop for engine calls
        self._loop = asyncio.new_event_loop()
        loop_thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        loop_thread.start()

        # Start proactive scheduler on our dedicated loop
        if self._proactive_config.enabled and self.engine.interaction_tracker:
            self._proactive_scheduler = ProactiveScheduler(
                config=self._proactive_config,
                tracker=self.engine.interaction_tracker,
                channel_registry=self._channel_registry,
                engine=self.engine,
            )
            asyncio.run_coroutine_threadsafe(
                self._proactive_scheduler.start(), self._loop
            )

        # Start reflection loop
        if self._reflection_config.enabled and self.engine.interaction_tracker:
            self._reflection_loop = ReflectionLoop(
                config=self._reflection_config,
                tracker=self.engine.interaction_tracker,
                engine=self.engine,
            )
            asyncio.run_coroutine_threadsafe(
                self._reflection_loop.start(), self._loop
            )

        # Start life simulator (Aria has a life running in the background)
        if self.engine.inner_life_store is not None:
            self._life_simulator = LifeSimulator(
                persona=self.engine.persona,
                llm=self.engine.llm,
                store=self.engine.inner_life_store,
                tick_interval_minutes=30,
            )
            asyncio.run_coroutine_threadsafe(
                self._life_simulator.start(), self._loop
            )

        # Build event handler
        handler = (
            lark.EventDispatcherHandler.builder("", "")
            .register_p2_im_message_receive_v1(self._on_message)
            .build()
        )
        # TODO(task-14-follow-up): Wire card action callback (button clicks → annotation)
        # Feishu sends card.action.trigger events as HTTP POST to a configured callback URL,
        # NOT via the WebSocket connection.  To handle 👍/👎/✏️ button clicks:
        #   1. Expose a FastAPI endpoint POST /feishu/card_action (or reuse the Web API app)
        #   2. Use lark.CardActionHandler.builder("", "").register(self._on_card_action).build()
        #   3. Route incoming HTTP requests to that handler
        #   4. In _on_card_action, read card.action.value["action"] and card.action.value["turn_id"]
        #      then dispatch to engine.annotation_store / AnnotationCollector
        # Until then, users can annotate via CLI :good/:bad or POST /turns/{id}/annotate.

        # Build WebSocket client
        ws_client = lark.ws.Client(
            app_id=self.app_id,
            app_secret=self.app_secret,
            event_handler=handler,
            log_level=lark.LogLevel.INFO,
        )

        proactive_hint = "开启" if self._proactive_config.enabled else "关闭"
        print(f"\n{'='*50}")
        print(f"  飞书机器人已启动 (WebSocket 长连接)")
        print(f"  人设: {self.engine.persona.name}")
        print(f"  主动消息: {proactive_hint}")
        print(f"  在飞书中找到机器人，发消息即可对话")
        print(f"{'='*50}\n")

        ws_client.start()

    def _on_message(self, event: P2ImMessageReceiveV1) -> None:
        """Handle incoming message event.

        CRITICAL: This must return FAST so the SDK can ACK to Feishu.
        Feishu retries if no ACK within 3 seconds.
        Actual LLM work is dispatched to the background event loop.
        """
        message = event.event.message
        msg_id = message.message_id

        # Skip messages from before bot startup (reject backlog from previous runs)
        try:
            create_time_ms = int(message.create_time)
            if create_time_ms < self._startup_time_ms:
                print(f"[stale] skipping pre-startup msg {msg_id}")
                return
        except (ValueError, TypeError):
            pass

        # Dedup within this session
        if msg_id in self._seen:
            print(f"[dedup] skipping retry of {msg_id}")
            return
        self._seen.add(msg_id)
        if len(self._seen) > 10000:
            self._seen = set(list(self._seen)[-5000:])

        chat_id = message.chat_id
        msg_type = message.message_type

        # Parse text and collect image_keys
        text = ""
        image_keys: list[str] = []

        try:
            content_data = json.loads(message.content) if message.content else {}
        except (json.JSONDecodeError, TypeError):
            content_data = {}

        if msg_type == "text":
            text = content_data.get("text", "").strip()
        elif msg_type == "image":
            image_keys.append(content_data.get("image_key", ""))
        elif msg_type == "post":
            # Rich text: extract text and image_keys from all lines
            text_parts = []
            for line in content_data.get("content", []):
                for node in line:
                    ntype = node.get("tag", "")
                    if ntype == "text":
                        text_parts.append(node.get("text", ""))
                    elif ntype == "img":
                        image_keys.append(node.get("image_key", ""))
                    elif ntype == "a":
                        text_parts.append(node.get("text", ""))
                    elif ntype == "at":
                        pass  # skip mentions
            text = "".join(text_parts).strip()
        else:
            asyncio.run_coroutine_threadsafe(
                self._send_card_or_text(chat_id, f"暂不支持 {msg_type} 类型的消息 ☺️"),
                self._loop,
            )
            return

        # Remove @mentions from text
        if message.mentions:
            for m in message.mentions:
                if m.key:
                    text = text.replace(m.key, "").strip()

        image_keys = [k for k in image_keys if k]

        if not text and not image_keys:
            return

        print(f"[recv] {msg_id} text={text[:50]!r} images={len(image_keys)}")

        # Dispatch to background loop and return immediately
        asyncio.run_coroutine_threadsafe(
            self._handle_reply_safe(chat_id, text, image_keys, msg_id),
            self._loop,
        )

    async def _handle_reply_safe(
        self,
        chat_id: str,
        text: str,
        image_keys: list[str],
        msg_id: str,
    ) -> None:
        """Wrapper that catches errors so background task doesn't die silently."""
        try:
            # Slash command: inspect memory without invoking LLM
            if text.startswith("/"):
                reply = await self._handle_command(chat_id, text)
                if reply is not None:
                    await self._send_card_or_text(chat_id, reply)
                    print(f"[cmd] {msg_id}: {text}")
                    return

            # Download images
            images: list[dict] = []
            if image_keys:
                async with httpx.AsyncClient(timeout=30) as http:
                    for key in image_keys:
                        img = await self._download_image(http, msg_id, key)
                        if img:
                            images.append(img)

            await self._stream_reply(chat_id, text, images=images)

            # Persist interaction tracker so proactive loop sees latest state
            if self.engine.interaction_tracker:
                await self.engine.interaction_tracker.save()

            # Periodic end_session: every N turns trigger memory consolidation
            # and relationship evaluation
            if self.engine.interaction_tracker:
                rec = self.engine.interaction_tracker.get_record("feishu", chat_id)
                if rec and rec.total_turns > 0 and rec.total_turns % 10 == 0:
                    print(f"[end_session] {chat_id} (turn {rec.total_turns})")
                    try:
                        result = await self.engine.end_session(
                            channel="feishu", recipient_id=chat_id
                        )
                        if result.get("relationship_level_changed"):
                            print(f"[relationship] new level: {result['new_relationship_level']}")
                    except Exception as e:
                        print(f"[end_session] error: {e}")

            print(f"[done] {msg_id}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            try:
                await self._send_card_or_text(chat_id, f"出了点小问题：{e}")
            except Exception:
                pass

    async def _download_image(
        self,
        http: httpx.AsyncClient,
        message_id: str,
        image_key: str,
    ) -> dict | None:
        """Download an image from Feishu by its key. Returns {media_type, data(b64)}."""
        headers = self.token_mgr.headers()
        # Remove Content-Type for GET
        headers = {k: v for k, v in headers.items() if k.lower() != "content-type"}
        url = (
            f"{FEISHU_BASE}/im/v1/messages/{message_id}/resources/{image_key}"
            f"?type=image"
        )
        resp = await http.get(url, headers=headers)
        if resp.status_code != 200:
            print(f"[image] download failed {resp.status_code}: {resp.text[:200]}")
            return None

        media_type = resp.headers.get("content-type", "image/png").split(";")[0]
        data_b64 = base64.standard_b64encode(resp.content).decode("ascii")
        print(f"[image] downloaded {image_key}: {len(resp.content)} bytes, {media_type}")
        return {"media_type": media_type, "data": data_b64}

    async def _send_text_async(self, chat_id: str, text: str) -> None:
        headers = self.token_mgr.headers()
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{FEISHU_BASE}/im/v1/messages?receive_id_type=chat_id",
                headers=headers,
                json={
                    "receive_id": chat_id,
                    "msg_type": "text",
                    "content": json.dumps({"text": text}),
                },
            )

    async def _handle_command(self, chat_id: str, text: str) -> str | None:
        """Handle /xxx slash commands. Returns reply text, or None to fall through to LLM."""
        recipient_key = f"feishu:{chat_id}"
        cmd, *rest = text.strip().split(maxsplit=1)
        arg = rest[0] if rest else ""

        if cmd == "/help":
            return (
                "可用命令：\n"
                "/stats - 记忆状态\n"
                "/mood - 当前心情\n"
                "/memories <关键词> - 搜索记忆\n"
                "/entities - 实体图谱\n"
                "/episodes - 最近session摘要\n"
                "/reveal <turn_id> - 查看 Aria 当时的内心独白\n"
                "/forget - 清空我的记忆（慎用）"
            )

        if cmd == "/stats":
            stats = self.engine.memory.get_stats()
            await self.engine.memory.entity_graph.load()
            ent_stats = self.engine.memory.entity_graph.stats()
            rec = None
            if self.engine.interaction_tracker:
                rec = self.engine.interaction_tracker.get_record("feishu", chat_id)
            lvl = rec.relationship_level if rec else 1
            turns = rec.total_turns if rec else 0
            return (
                f"📊 状态\n"
                f"短期: {stats['short_term_turns']} 轮\n"
                f"长期: {stats['long_term_entries']} 条\n"
                f"情景: {stats['episodes']} 段\n"
                f"实体: {ent_stats['entity_count']} 个 / {ent_stats['total_links']} 个链接\n"
                f"关系等级: {lvl} | 总对话轮数: {turns}"
            )

        if cmd == "/mood":
            top = self.engine._emotion_state.top_k(k=5)
            lines = [f"💭 心情: {self.engine._current_mood}"]
            for name, val in top:
                bar = "█" * int(val * 10)
                lines.append(f"  {name} {val:.2f} {bar}")
            return "\n".join(lines)

        if cmd == "/memories":
            query = arg or "最近"
            ctx = await self.engine.memory.assemble_context(
                query, long_term_limit=8, episode_limit=3,
                recipient_key=recipient_key,
            )
            lines = [f"🔍 \"{query}\" 的记忆:"]
            lines.append(f"事实 ({len(ctx.long_term_facts)}):")
            for f in ctx.long_term_facts[:8]:
                lines.append(f"  • [{f.importance:.1f}] {f.content[:60]}")
            lines.append(f"\nsession ({len(ctx.relevant_episodes)}):")
            for ep in ctx.relevant_episodes:
                lines.append(f"  • [{ep.timestamp.strftime('%m-%d')}] {ep.summary[:80]}")
            return "\n".join(lines)

        if cmd == "/entities":
            await self.engine.memory.entity_graph.load()
            ents = self.engine.memory.entity_graph.all_entities()
            ents.sort(key=lambda e: e.mention_count, reverse=True)
            lines = [f"🌐 实体图谱 ({len(ents)} 个):"]
            for e in ents[:15]:
                lines.append(f"  • {e.name} ({e.type}) ×{e.mention_count}")
            return "\n".join(lines)

        if cmd == "/episodes":
            eps = await self.engine.memory.episodic.get_recent(limit=8)
            lines = [f"📚 最近 session ({len(eps)} 段):"]
            for ep in eps:
                lines.append(
                    f"  • [{ep.timestamp.strftime('%m-%d %H:%M')}] "
                    f"({ep.emotional_tone}) {ep.summary[:80]}"
                )
            return "\n".join(lines)

        if cmd == "/test-proactive":
            if self._proactive_scheduler is None:
                return "⚠️ 主动调度器未启用"
            results = await self._proactive_scheduler.trigger_manually()
            lines = ["🧪 主动消息测试触发结果："]
            for r in results:
                lines.append(f"  • {r.get('status')}: {r}")
            return "\n".join(lines)

        if cmd == "/forget":
            return (
                "⚠️ 清空记忆需要二次确认。请发送 /forget confirm "
                "（这会删除所有针对你的对话记忆和情绪状态，不可恢复）"
            )

        if cmd == "/reveal":
            turn_id = arg.strip()
            if not turn_id:
                return "用法: /reveal <turn_id>"
            if self.engine.annotation_store is None:
                return "未启用标注存储"
            turn = await self.engine.annotation_store.get_turn(turn_id)
            if turn is None:
                return f"未找到 {turn_id}"
            return f"💭 Aria 当时想的：\n{turn.inner_thought or '(无)'}"

        # Unknown command - let LLM handle it
        return None

    async def _stream_reply(
        self,
        chat_id: str,
        user_text: str,
        images: list[dict] | None = None,
    ) -> None:
        """Create streaming card and push LLM chunks."""
        async with httpx.AsyncClient(timeout=60) as http:
            card = StreamingCardSender(self.token_mgr, http)
            await card.create_card()
            await card.send_to_chat(chat_id)

            accumulated = ""
            last_update = 0.0
            turn_id: str | None = None

            async for event in self.engine.chat_stream_events(
                user_text,
                images=images,
                channel="feishu",
                recipient_id=chat_id,
            ):
                if event.type == "chunk":
                    accumulated += event.content
                    now = time.time()
                    if now - last_update >= self._update_interval:
                        try:
                            await card.update_content(accumulated)
                        except Exception:
                            pass
                        last_update = now

                elif event.type == "turn_id":
                    turn_id = event.content

                elif event.type == "done":
                    # Final render: pure speech, no meta leaked to user
                    final = event.content
                    try:
                        await card.update_content(final)
                    except Exception:
                        pass

            # Append annotation footer buttons when turn_id is available
            footer = build_annotation_footer_elements(turn_id) if turn_id else None
            try:
                await card.finish(footer_elements=footer)
            except Exception:
                pass
