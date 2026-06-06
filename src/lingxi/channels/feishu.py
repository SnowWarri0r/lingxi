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
from lingxi.temporal.proactive import ProactiveConfig, ProactiveScheduler

FEISHU_BASE = "https://open.feishu.cn/open-apis"


def build_annotation_footer_elements(turn_id: str) -> list[dict]:
    """Annotation footer: 👍/👎 quick callback buttons + ✏️ correction form.

    Layout (CardKit v2):
      ──────────────────────
      [👍 像]   [👎 不像]
      [应该说输入框…]
      [✏️ 提交修正]
    """

    def _callback_button(text: str, action_kind: str) -> dict:
        return {
            "tag": "button",
            "text": {"tag": "plain_text", "content": text},
            "type": "default",
            "width": "default",
            "behaviors": [
                {
                    "type": "callback",
                    "value": {"action": action_kind, "turn_id": turn_id},
                }
            ],
        }

    quick_row = {
        "tag": "column_set",
        "horizontal_spacing": "small",
        "columns": [
            {
                "tag": "column",
                "width": "weighted",
                "weight": 1,
                "elements": [_callback_button("👍 像", "annotate_positive")],
            },
            {
                "tag": "column",
                "width": "weighted",
                "weight": 1,
                "elements": [_callback_button("👎 不像", "annotate_negative")],
            },
        ],
    }

    correction_form = {
        "tag": "form",
        "name": f"correction_{turn_id[:8]}",
        "elements": [
            {
                "tag": "input",
                "name": "correction_text",
                "placeholder": {
                    "tag": "plain_text",
                    "content": "应该说的话…",
                },
                "max_length": 200,
                "width": "fill",
            },
            {
                "tag": "button",
                "name": f"submit_correction_{turn_id[:8]}",
                "text": {"tag": "plain_text", "content": "✏️ 提交修正"},
                "type": "primary",
                "width": "default",
                "form_action_type": "submit",
                "behaviors": [
                    {
                        "type": "callback",
                        "value": {
                            "action": "annotate_correction",
                            "turn_id": turn_id,
                        },
                    }
                ],
            },
        ],
    }

    return [
        {"tag": "hr"},
        quick_row,
        correction_form,
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

    async def finish(self) -> None:
        if not self._card_id:
            return

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

    async def append_elements(self, elements: list[dict]) -> None:
        """Insert elements after the streaming element.

        Uses POST /cardkit/v1/cards/{id}/elements with type=insert_after,
        target_element_id=md_stream. Must be called after finish() has
        disabled streaming_mode.
        """
        if not self._card_id or not elements:
            print(f"[feishu] append_elements skipped: card_id={self._card_id}, elements={len(elements) if elements else 0}")
            return

        self._seq += 1
        headers = self._token_mgr.headers()
        body = {
            "type": "insert_after",
            "target_element_id": "md_stream",
            "sequence": self._seq,
            "elements": json.dumps(elements, ensure_ascii=False),
        }
        print(f"[feishu] append_elements → card={self._card_id} seq={self._seq} body_preview={body['elements'][:120]}", flush=True)
        try:
            resp = await self._http.post(
                f"{FEISHU_BASE}/cardkit/v1/cards/{self._card_id}/elements",
                headers=headers,
                json=body,
            )
            print(f"[feishu] append_elements ← HTTP {resp.status_code}: {resp.text[:500]}", flush=True)
        except Exception as e:
            print(f"[feishu] append_elements exception: {e}", flush=True)


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
        self._world_scheduler = None  # WorldScheduler — started below
        self._social_scheduler = None  # SocialScheduler — started below

    @property
    def channel_name(self) -> str:
        return "feishu"

    async def send_message(
        self,
        recipient_id: str,
        text: str,
        turn_id: str | None = None,
    ) -> None:
        """OutboundChannel implementation - send proactive message as a card.

        If `turn_id` is provided, append the 👍/👎/✏️ annotation footer so
        the user can rate the proactive message.
        """
        try:
            await self._send_proactive_card(recipient_id, text, turn_id=turn_id)
        except Exception:
            # Fallback to plain text if card fails
            await self._send_text_async(recipient_id, text)

    async def _send_proactive_card(
        self,
        chat_id: str,
        text: str,
        turn_id: str | None = None,
    ) -> None:
        """Send a message as a markdown card (visually consistent with chat replies)."""
        async with httpx.AsyncClient(timeout=30) as http:
            card = StreamingCardSender(self.token_mgr, http)
            await card.create_card()
            await card.send_to_chat(chat_id)
            await card.update_content(text)
            await card.finish()
            if turn_id:
                try:
                    await card.append_elements(
                        build_annotation_footer_elements(turn_id)
                    )
                except Exception as e:
                    print(f"[feishu] append proactive buttons failed: {e}", flush=True)

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

        # Shared data_dir for proactive + social schedulers
        data_dir = (
            getattr(self.engine.memory, "data_dir", None)
            or os.environ.get("MEMORY_DATA_DIR", "./data/memory")
        )

        # Build share_intent_store early so both ProactiveScheduler and
        # SocialPromoter share the same queue (promoter writes, proactive reads).
        _share_intent_store = None
        if getattr(self.engine, "life_writer", None) is not None:
            from lingxi.proactive.share_intent import ShareIntentStore
            _share_intent_store = ShareIntentStore(data_dir)

        # Start proactive scheduler on our dedicated loop
        if self._proactive_config.enabled and self.engine.interaction_tracker:
            # Pass the memory data_dir so anti-repetition history persists
            # across restarts (otherwise Aria forgets what she just sent)
            self._proactive_scheduler = ProactiveScheduler(
                config=self._proactive_config,
                tracker=self.engine.interaction_tracker,
                channel_registry=self._channel_registry,
                engine=self.engine,
                data_dir=str(data_dir),
                share_intent_store=_share_intent_store,
                fact_retriever=getattr(self.engine, "fact_retriever", None),
            )
            asyncio.run_coroutine_threadsafe(
                self._proactive_scheduler.start(), self._loop
            )

        # Reflection is handled by facts/Reflector + ReflectionTrigger (wired
        # in create_engine). Aria's life runs through DailyPlanner +
        # PlanExecutor (started below) writing aria.* facts — the old
        # ReflectionLoop and LifeSimulator were retired.

        # World scheduler — fetches today's news briefing each morning.
        # Requires the LLM provider's api_key to issue the web_search call.
        # Skip silently if no api_key available (e.g. dev/test config).
        api_key_for_world = getattr(self.engine.llm, "_api_key", "") or ""
        if (
            getattr(self.engine, "world_writer", None) is not None
            and api_key_for_world
        ):
            from lingxi.world.scheduler import WorldScheduler
            self._world_scheduler = WorldScheduler(
                api_key=api_key_for_world,
                world_writer=self.engine.world_writer,
                fact_retriever=getattr(self.engine, "fact_retriever", None),
            )
            asyncio.run_coroutine_threadsafe(
                self._world_scheduler.start(), self._loop
            )

        # Social scheduler — generates NPC events on cron ticks so the
        # "身边的人" block has fresh material. Daytime-only (8-22, every 2h).
        # Promoter pushes significance≥0.6 events to Aria's facts store +
        # ShareIntentStore so the proactive opener can surface them.
        if (
            getattr(self.engine, "social_graph", None) is not None
            and getattr(self.engine, "social_store", None) is not None
            and getattr(self.engine.social_graph, "npcs", None)  # skip when NPC roster empty
        ):
            from lingxi.social.promoter import SocialPromoter
            from lingxi.social.scheduler import SocialScheduler

            promoter_hook = None
            if _share_intent_store is not None:
                # Reuse the same ShareIntentStore instance created above so
                # promoter writes and proactive reads share one queue file.
                promoter = SocialPromoter(
                    life_writer=self.engine.life_writer,
                    share_intent_store=_share_intent_store,
                    social_store=self.engine.social_store,
                )
                promoter_hook = promoter.maybe_promote
            self._social_scheduler = SocialScheduler(
                llm=self.engine.llm,
                graph=self.engine.social_graph,
                store=self.engine.social_store,
                on_event_written=promoter_hook,
                npc_writer=self.engine.npc_writer,
                life_writer=getattr(self.engine, "life_writer", None),
                retriever=getattr(self.engine, "fact_retriever", None),
                fact_retriever=getattr(self.engine, "fact_retriever", None),
            )
            asyncio.run_coroutine_threadsafe(
                self._social_scheduler.start(), self._loop
            )

        # Plan executor tick (every 30min) + morning planner (daily 7am).
        # DailyPlanner and PlanExecutor are constructed in create_engine and
        # attached as engine.daily_planner / engine.plan_executor.
        if getattr(self.engine, "plan_executor", None) is not None:
            asyncio.run_coroutine_threadsafe(
                self._start_plan_loops(), self._loop
            )

        # Build event handler
        handler = (
            lark.EventDispatcherHandler.builder("", "")
            .register_p2_im_message_receive_v1(self._on_message)
            .register_p2_card_action_trigger(self._on_card_action)
            .build()
        )

        # Build WebSocket client
        ws_client = lark.ws.Client(
            app_id=self.app_id,
            app_secret=self.app_secret,
            event_handler=handler,
            log_level=lark.LogLevel.DEBUG,
        )

        proactive_hint = "开启" if self._proactive_config.enabled else "关闭"
        print(f"\n{'='*50}")
        print("  飞书机器人已启动 (WebSocket 长连接)")
        print(f"  人设: {self.engine.persona.name}")
        print(f"  主动消息: {proactive_hint}")
        print("  在飞书中找到机器人，发消息即可对话")
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

    def _on_card_action(self, event) -> None:
        """Handle 👍/👎/✏️ button clicks from annotation cards."""
        from lark_oapi.event.callback.model.p2_card_action_trigger import (
            P2CardActionTriggerResponse,
        )

        print("[feishu] _on_card_action fired", flush=True)
        try:
            action_obj = event.event.action if event.event else None
            value = (action_obj.value or {}) if action_obj else {}
            form_value = (action_obj.form_value or {}) if action_obj else {}
            action = value.get("action", "")
            turn_id = value.get("turn_id", "")
            # Lark P2CardActionTrigger event carries the originating chat_id
            event_data = getattr(event, "event", None)
            open_chat_id = getattr(event_data, "open_chat_id", None) or ""
            print(
                f"[feishu] action={action!r} turn_id={turn_id!r} "
                f"chat={open_chat_id[:12]}... form_value={form_value!r}",
                flush=True,
            )

            if not turn_id or not action.startswith("annotate_"):
                return P2CardActionTriggerResponse({})

            correction: str | None = None
            if action == "annotate_correction":
                correction = (form_value.get("correction_text") or "").strip()
                if not correction:
                    tip = "先在输入框里写点东西再提交吧"
                    return P2CardActionTriggerResponse({
                        "toast": {
                            "type": "warning",
                            "content": tip,
                            "i18n": {"zh_cn": tip},
                        },
                    })

            # Dispatch to annotation in background (handler returns synchronously).
            # Pass open_chat_id so the handler can verify the turn belongs to
            # this chat before acting on it.
            asyncio.run_coroutine_threadsafe(
                self._handle_card_annotation(action, turn_id, correction, open_chat_id),
                self._loop,
            )

            # Show a lightweight toast back to the user
            toast_content = {
                "annotate_positive": "👍 记下了",
                "annotate_negative": "👎 记下了",
                "annotate_correction": f"✏️ 记下「{correction}」了" if correction else "✏️ 记下了",
            }.get(action, "已记录")

            print(f"[feishu] returning toast: {toast_content}", flush=True)
            return P2CardActionTriggerResponse({
                "toast": {
                    "type": "info",
                    "content": toast_content,
                    "i18n": {"zh_cn": toast_content},
                },
            })
        except Exception as e:
            import traceback
            print(f"[feishu] card action handler failed: {e}\n{traceback.format_exc()}", flush=True)
            return P2CardActionTriggerResponse({})

    def _npc_registry(self):
        """Return the list of NPC objects from the engine's social_graph (may be empty)."""
        graph = getattr(self.engine, "social_graph", None)
        if graph is None:
            return []
        return list(graph.npcs)

    async def _start_plan_loops(self) -> None:
        """Start DailyPlanner morning tick + PlanExecutor 30-min tick.

        Called once from start() inside the dedicated asyncio loop.
        Runs two concurrent tasks:
          - _executor_loop:  PlanExecutor.tick() every 30 min
          - _morning_planner_loop: DailyPlanner.plan_aria() at 7am each day,
            plus an immediate run on startup if today has no plan yet.
        """
        import asyncio
        from datetime import datetime, timedelta
        from lingxi.facts.models import FactType

        plan_executor = self.engine.plan_executor
        daily_planner = getattr(self.engine, "daily_planner", None)
        facts_store = getattr(self.engine, "_facts_store", None)
        # Fall through to fact_retriever's store if not directly on engine
        if facts_store is None:
            fr = getattr(self.engine, "fact_retriever", None)
            if fr is not None:
                facts_store = getattr(fr, "_store", None) or getattr(fr, "store", None)

        async def _executor_loop() -> None:
            while True:
                try:
                    await plan_executor.tick()
                except Exception as e:
                    print(f"[executor] tick error: {e}", flush=True)
                await asyncio.sleep(1800)  # 30 min

        async def _morning_planner_loop() -> None:
            while True:
                now = datetime.now()
                next_7am = now.replace(hour=7, minute=0, second=0, microsecond=0)
                if next_7am <= now:
                    next_7am += timedelta(days=1)
                await asyncio.sleep((next_7am - now).total_seconds())
                try:
                    if daily_planner is not None:
                        await daily_planner.plan_aria()
                except Exception as e:
                    print(f"[planner] morning tick failed: {e}", flush=True)
                for npc in self._npc_registry():
                    try:
                        if daily_planner is not None:
                            await daily_planner.plan_npc(npc.id, display_name=npc.name)
                    except Exception as e:
                        print(f"[planner] NPC {npc.id} plan failed: {e}", flush=True)

        async def _ensure_today_plan() -> None:
            if daily_planner is None or facts_store is None:
                return
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            try:
                todays = await facts_store.query(
                    subject="aria", type=FactType.PLAN, since=today_start, limit=1
                )
                if not todays:
                    await daily_planner.plan_aria()
            except Exception as e:
                print(f"[planner] startup plan failed: {e}", flush=True)
            for npc in self._npc_registry():
                try:
                    npc_plans = await facts_store.query(
                        subject=f"npc:{npc.id}", type=FactType.PLAN,
                        since=today_start, limit=1,
                    )
                    if not npc_plans:
                        await daily_planner.plan_npc(npc.id, display_name=npc.name)
                except Exception as e:
                    print(f"[planner] startup plan_npc({npc.id}) failed: {e}", flush=True)

        asyncio.ensure_future(_ensure_today_plan())
        asyncio.ensure_future(_executor_loop())
        asyncio.ensure_future(_morning_planner_loop())

    async def _handle_card_annotation(
        self,
        action: str,
        turn_id: str,
        correction: str | None = None,
        open_chat_id: str = "",
    ) -> None:
        """Dispatch the annotation action to AnnotationCollector.

        Verifies the turn's recipient_key matches the chat the action came
        from — prevents one user clicking a forged button on another user's
        turn id from training/exposing that other user's data.
        """
        print(f"[feishu] _handle_card_annotation start: action={action}, turn={turn_id[:8]}, correction={correction!r}", flush=True)
        if self.engine.annotation_store is None or self.engine.fewshot_store is None:
            print("[feishu] skip: annotation_store or fewshot_store is None", flush=True)
            return

        # Recipient match check
        if open_chat_id:
            expected = f"feishu:{open_chat_id}"
            turn = await self.engine.annotation_store.get_turn(turn_id)
            if turn is None or turn.recipient_key != expected:
                print(
                    f"[feishu] reject card-action: turn recipient={getattr(turn, 'recipient_key', None)!r} "
                    f"!= expected={expected!r}",
                    flush=True,
                )
                return

        from lingxi.fewshot.collector import AnnotationCollector
        from lingxi.fewshot.summarizer import AnnotationSummarizer

        embedder = self.engine.memory.embedding_provider or (
            self.engine.fewshot_retriever.embedder
            if self.engine.fewshot_retriever else None
        )
        if embedder is None:
            print("[feishu] skip: no embedder available (need ARK_API_KEY for fewshot pool)", flush=True)
            return

        print(f"[feishu] collector ready, embedder={type(embedder).__name__}", flush=True)
        collector = AnnotationCollector(
            annotation_store=self.engine.annotation_store,
            fewshot_store=self.engine.fewshot_store,
            embedder=embedder,
            summarizer=AnnotationSummarizer(self.engine.llm),
        )

        try:
            if action == "annotate_positive":
                print("[feishu] calling record_positive", flush=True)
                await collector.record_positive(turn_id)
            elif action == "annotate_negative":
                print("[feishu] calling record_negative", flush=True)
                await collector.record_negative(turn_id)
            elif action == "annotate_correction" and correction:
                print("[feishu] calling record_correction", flush=True)
                await collector.record_correction(turn_id, correction)
            print(f"[feishu] annotation recorded: {action}", flush=True)
        except Exception as e:
            import traceback
            print(f"[feishu] annotation dispatch failed: {e}\n{traceback.format_exc()}", flush=True)

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

    async def _send_image(self, chat_id: str, file_path: str) -> None:
        """Upload a local image to Feishu and send it as an image message.

        Two calls: POST /im/v1/images (multipart) -> image_key, then
        POST /im/v1/messages with msg_type=image. Failures are logged, not
        raised, so a sticker problem never breaks the turn.
        """
        from pathlib import Path as _Path
        try:
            data = _Path(file_path).read_bytes()
        except Exception as e:
            print(f"[sticker] read failed {file_path}: {e}", flush=True)
            return

        # Upload: multipart, no JSON Content-Type (httpx sets the boundary).
        headers = self.token_mgr.headers()
        headers = {k: v for k, v in headers.items()
                   if k.lower() != "content-type"}
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                up = await client.post(
                    f"{FEISHU_BASE}/im/v1/images",
                    headers=headers,
                    data={"image_type": "message"},
                    files={"image": (_Path(file_path).name, data)},
                )
                up_data = up.json()
                if up_data.get("code") != 0:
                    print(f"[sticker] upload failed: {up_data}", flush=True)
                    return
                image_key = up_data["data"]["image_key"]

                send = await client.post(
                    f"{FEISHU_BASE}/im/v1/messages?receive_id_type=chat_id",
                    headers=self.token_mgr.headers(),
                    json={
                        "receive_id": chat_id,
                        "msg_type": "image",
                        "content": json.dumps({"image_key": image_key}),
                    },
                )
                send_data = send.json()
                if send_data.get("code") != 0:
                    print(f"[sticker] send failed: {send_data}", flush=True)
        except Exception as e:
            print(f"[sticker] _send_image error: {e}", flush=True)

    async def _send_static_card_async(self, chat_id: str, text: str) -> str:
        """Send a non-streaming card with the given text already baked in.

        Used for multi-bubble extras: visually consistent with the first
        bubble's streaming card (border/styling) but without the typing
        animation since the content is already determined.

        The markdown element gets element_id="md_stream" so that
        append_elements (used to attach the annotation footer to the LAST
        card in a multi-bubble turn) works the same way it does on the
        streaming card.

        Returns the created card_id so the caller can append elements to
        it later.
        """
        headers = self.token_mgr.headers()
        static_card = {
            "schema": "2.0",
            "body": {
                "elements": [
                    {
                        "tag": "markdown",
                        "content": text,
                        "element_id": "md_stream",
                    },
                ]
            },
        }
        async with httpx.AsyncClient() as client:
            create = await client.post(
                f"{FEISHU_BASE}/cardkit/v1/cards",
                headers=headers,
                json={
                    "type": "card_json",
                    "data": json.dumps(static_card, ensure_ascii=False),
                },
            )
            create_data = create.json()
            if create_data.get("code") != 0:
                raise RuntimeError(f"Create static card failed: {create_data}")
            card_id = create_data["data"]["card_id"]

            send = await client.post(
                f"{FEISHU_BASE}/im/v1/messages?receive_id_type=chat_id",
                headers=headers,
                json={
                    "receive_id": chat_id,
                    "msg_type": "interactive",
                    "content": json.dumps(
                        {"type": "card", "data": {"card_id": card_id}}
                    ),
                    "uuid": uuid.uuid4().hex,
                },
            )
            send_data = send.json()
            if send_data.get("code") != 0:
                raise RuntimeError(f"Send static card failed: {send_data}")
            return card_id

    async def _append_to_card_id(self, card_id: str, elements: list[dict]) -> None:
        """Append elements to an existing card by id.

        Same wire shape as StreamingCardSender.append_elements, but
        callable on a card created by _send_static_card_async (which
        doesn't return a sender object).
        """
        if not card_id or not elements:
            return
        headers = self.token_mgr.headers()
        body = {
            "type": "insert_after",
            "target_element_id": "md_stream",
            "sequence": 1,
            "elements": json.dumps(elements, ensure_ascii=False),
        }
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(
                    f"{FEISHU_BASE}/cardkit/v1/cards/{card_id}/elements",
                    headers=headers,
                    json=body,
                )
                if resp.status_code != 200:
                    print(
                        f"[feishu] append to {card_id} failed: "
                        f"HTTP {resp.status_code}: {resp.text[:200]}",
                        flush=True,
                    )
            except Exception as e:
                print(f"[feishu] append to {card_id} exception: {e}", flush=True)

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
                "/memories <关键词> - 搜索 facts\n"
                "/reveal <turn_id> - 查看 Aria 当时的内心独白\n"
                "/good [turn_id] - 标记某轮回复像 (默认最后一轮)\n"
                "/bad <turn_id> <应该说的话> - 提交修正\n"
                "/forget - 清空我的记忆（慎用）"
            )

        if cmd == "/stats":
            stats = self.engine.memory.get_stats()
            rec = None
            if self.engine.interaction_tracker:
                rec = self.engine.interaction_tracker.get_record("feishu", chat_id)
            lvl = rec.relationship_level if rec else 1
            turns = rec.total_turns if rec else 0
            fact_count = "?"
            if getattr(self.engine, "fact_retriever", None) is not None:
                try:
                    cat = await self.engine.fact_retriever.catalog()
                    fact_count = sum(cat.values())
                except Exception:
                    fact_count = "?"
            return (
                f"📊 状态\n"
                f"短期: {stats['short_term_turns']} 轮\n"
                f"facts.db: {fact_count} 条\n"
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
            query = arg or None
            from lingxi.facts.retriever import FactQuery
            facts = await self.engine.fact_retriever.fetch(
                FactQuery(subject=f"user:{recipient_key}", semantic=query, limit=8)
            )
            lines = [f"🔍 \"{query or '最近'}\" 的 facts ({len(facts)}):"]
            for f in facts:
                lines.append(f"  • [{f.importance}] {f.content[:60]}")
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

        expected_recipient = f"feishu:{chat_id}"

        if cmd == "/reveal":
            turn_id = arg.strip()
            if not turn_id:
                return "用法: /reveal <turn_id>"
            if self.engine.annotation_store is None:
                return "未启用标注存储"
            turn = await self._resolve_turn(turn_id, expected_recipient=expected_recipient)
            if turn is None:
                return f"未找到 turn {turn_id}"
            return f"💭 Aria 当时想的（{turn.turn_id[:8]}）：\n{turn.inner_thought or '(无)'}"

        if cmd == "/bad":
            return await self._cmd_correction(arg.strip(), expected_recipient=expected_recipient)

        if cmd == "/good":
            return await self._cmd_good(arg.strip(), expected_recipient=expected_recipient)

        # Unknown command - let LLM handle it
        return None

    _TURN_REF_RE = __import__("re").compile(r"^[0-9a-fA-F-]{6,36}$")

    async def _resolve_turn(self, ref: str, expected_recipient: str | None = None):
        """Resolve `ref` to an AnnotationTurn, with optional recipient match.

        - `ref` must be hex/dash only (UUID format) — guards against glob
          meta-chars like `?` `[` `*` that would expand the match set.
        - When `expected_recipient` is given, the resolved turn's
          `recipient_key` must equal it. Prevents one user from poking at
          another user's turn ids by guessing 8-char prefixes.
        """
        if self.engine.annotation_store is None or not ref:
            return None
        if not self._TURN_REF_RE.match(ref):
            return None

        turn = await self.engine.annotation_store.get_turn(ref)
        if turn is None:
            # Prefix resolution — glob is now safe because ref is hex-only.
            from pathlib import Path
            turns_dir = Path(self.engine.annotation_store.turns_dir)
            if not turns_dir.exists():
                return None
            for p in turns_dir.glob(f"{ref}*.json"):
                turn = await self.engine.annotation_store.get_turn(p.stem)
                if turn is not None:
                    break

        if turn is None:
            return None
        if expected_recipient and turn.recipient_key != expected_recipient:
            return None
        return turn

    async def _cmd_correction(self, arg: str, expected_recipient: str | None = None) -> str:
        """`/bad <turn_id_or_prefix> <correction>` — record a user_correction sample."""
        if self.engine.annotation_store is None or self.engine.fewshot_store is None:
            return "未启用标注闭环"
        parts = arg.split(maxsplit=1)
        if len(parts) < 2:
            return "用法: /bad <turn_id 前 8 位> <应该说的话>"
        ref, correction = parts[0], parts[1].strip()
        turn = await self._resolve_turn(ref, expected_recipient=expected_recipient)
        if turn is None:
            return f"未找到 turn {ref}"

        from lingxi.fewshot.collector import AnnotationCollector
        from lingxi.fewshot.summarizer import AnnotationSummarizer

        embedder = self.engine.memory.embedding_provider or (
            self.engine.fewshot_retriever.embedder
            if self.engine.fewshot_retriever else None
        )
        if embedder is None:
            return "没有可用的 embedding provider"

        collector = AnnotationCollector(
            annotation_store=self.engine.annotation_store,
            fewshot_store=self.engine.fewshot_store,
            embedder=embedder,
            summarizer=AnnotationSummarizer(self.engine.llm),
        )
        try:
            await collector.record_correction(turn.turn_id, correction)
            return f"✏️ 记下「{correction}」了（turn {turn.turn_id[:8]}）"
        except Exception as e:
            return f"失败: {e}"

    async def _cmd_good(self, arg: str, expected_recipient: str | None = None) -> str:
        """`/good [turn_id_or_prefix]` — record as positive (defaults to last turn)."""
        if self.engine.annotation_store is None or self.engine.fewshot_store is None:
            return "未启用标注闭环"
        ref = arg.strip()
        if not ref:
            last = getattr(self.engine, "_last_output", None)
            ref = last.turn_id if last else ""
        turn = (
            await self._resolve_turn(ref, expected_recipient=expected_recipient)
            if ref else None
        )
        if turn is None:
            return "未找到 turn（用 /good <前 8 位>）"

        from lingxi.fewshot.collector import AnnotationCollector
        from lingxi.fewshot.summarizer import AnnotationSummarizer

        embedder = self.engine.memory.embedding_provider or (
            self.engine.fewshot_retriever.embedder
            if self.engine.fewshot_retriever else None
        )
        if embedder is None:
            return "没有可用的 embedding provider"

        collector = AnnotationCollector(
            annotation_store=self.engine.annotation_store,
            fewshot_store=self.engine.fewshot_store,
            embedder=embedder,
            summarizer=AnnotationSummarizer(self.engine.llm),
        )
        try:
            await collector.record_positive(turn.turn_id)
            return f"👍 记下了（turn {turn.turn_id[:8]}）"
        except Exception as e:
            return f"失败: {e}"

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
            pending_sticker: str | None = None

            stream_error: Exception | None = None
            try:
                async for event in self.engine.chat_stream_events(
                    user_text,
                    images=images,
                    channel="feishu",
                    recipient_id=chat_id,
                ):
                    if event.type == "thinking":
                        try:
                            preview = event.content.strip()
                            if preview:
                                await card.update_content(f"💭 {preview}…")
                        except Exception:
                            pass

                    elif event.type == "chunk":
                        accumulated += event.content
                        now = time.time()
                        if now - last_update >= self._update_interval:
                            try:
                                # Only stream the FIRST bubble into the card.
                                # If `\n\n` has appeared, the rest belongs to
                                # later bubbles and would otherwise briefly
                                # flash here before being cut to first_bubble
                                # at stream-end (visible jank — full message
                                # appears, then card "shrinks").
                                first_so_far = accumulated.split("\n\n", 1)[0]
                                await card.update_content(first_so_far)
                            except Exception:
                                pass
                            last_update = now

                    elif event.type == "sticker":
                        pending_sticker = event.content

                    elif event.type == "turn_id":
                        turn_id = event.content

                    elif event.type == "done":
                        final_full = event.content
            except Exception as e:
                stream_error = e
                print(f"[feishu] stream raised: {e}", flush=True)
                # Replace the "💭 thinking..." placeholder with a graceful
                # error so the user isn't staring at it forever.
                try:
                    await card.update_content("嗯 网络抽了一下 你再说一次")
                except Exception:
                    pass

            from lingxi.conversation.response_cleaner import split_into_bubbles
            bubbles: list[str] = []
            extras: list[str] = []

            # Stream-error path: keep the explicit error message we just
            # wrote, do NOT run multi-bubble logic (which would either blank
            # the card with an empty final_full or stamp a partial answer
            # over our error text).
            if stream_error is None:
                # Multi-bubble split: model can emit speech with `\n\n` breaks
                # to send 2-3 separate IM messages. First bubble updates the
                # streaming card; extras send as plain text after.
                final_speech = locals().get("final_full") or accumulated
                bubbles = split_into_bubbles(final_speech, max_bubbles=3)
                first_bubble = bubbles[0] if bubbles else final_speech
                extras = bubbles[1:] if len(bubbles) > 1 else []
                if first_bubble:
                    try:
                        await card.update_content(first_bubble)
                    except Exception:
                        pass

            try:
                await card.finish()
            except Exception as e:
                print(f"[feishu] finish() failed: {e}", flush=True)

            print(
                f"[feishu] stream done, turn_id={turn_id!r}, err={stream_error!r}, "
                f"bubbles={len(bubbles)}",
                flush=True,
            )

            # Send extras as static cards. Track the LAST card sent so we
            # can attach annotation buttons there — buttons under the most
            # recent message read more naturally than buttons under the
            # first bubble (which can be 2-3 messages above by the time
            # the user wants to react).
            last_extra_card_id: str | None = None
            for extra in extras:
                try:
                    last_extra_card_id = await self._send_static_card_async(
                        chat_id, extra
                    )
                except Exception as e:
                    print(f"[feishu] extra bubble send failed: {e}", flush=True)

            if turn_id:
                footer = build_annotation_footer_elements(turn_id)
                try:
                    if last_extra_card_id is not None:
                        # Annotation footer on the LAST extra card
                        await self._append_to_card_id(last_extra_card_id, footer)
                    else:
                        # No extras → buttons on the first (only) card
                        await card.append_elements(footer)
                except Exception as e:
                    print(f"[feishu] append buttons failed: {e}", flush=True)

            # Send the sticker as a separate image message after the text.
            if pending_sticker:
                try:
                    await self._send_image(chat_id, pending_sticker)
                except Exception as e:
                    print(f"[feishu] sticker send failed: {e}", flush=True)

