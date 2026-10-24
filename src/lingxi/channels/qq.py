"""QQ private chat through the official QQ bot platform (API v2, C2C).

Events come over the platform's WebSocket gateway, the same shape of
long connection the Feishu bot uses, so nothing has to be reachable from the
internet. Replies are passive messages bound to his message (msg_id, up to
four per message within the reply window); openers are active messages.

A personal bot reaches only the QQ accounts added as experience users in its
console. An account that sends the pairing code is linked to the person she
already knows (see identity.py), so she talks to him as herself, not to a
stranger; any other account is its own person.
"""

from __future__ import annotations

import asyncio
import base64
import hmac
import json
import os
import time

import httpx

from lingxi.channels.identity import IdentityMap
from lingxi.channels.outbound import OutboundChannel, RecipientUnreachable
from lingxi.conversation.response_cleaner import split_into_bubbles

TOKEN_URL = "https://bots.qq.com/app/getAppAccessToken"
API_BASE = os.environ.get("QQ_BOT_API_BASE", "https://api.bot.qq.com")

# GROUP_AND_C2C_EVENT: private (C2C) and group @-messages.
INTENTS = 1 << 25

# Close codes after which reconnecting cannot help (bad intent, bot offline
# or banned). Everything else is retried.
_FATAL_CLOSE = {4001, 4002, 4010, 4011, 4012, 4013, 4014, 4914, 4915}

# He has switched off messages from the bot in QQ. Retrying cannot help, and
# the opener should go to another channel he uses.
_REFUSED = {40054013}

# A message from him can be answered this many times.
_PASSIVE_REPLIES = 4


class QQApiError(RuntimeError):
    def __init__(self, code: int, message: str):
        super().__init__(f"{code} {message}")
        self.code = code


class QQAuth:
    def __init__(self, app_id: str, secret: str):
        self._app_id, self._secret = app_id, secret
        self._token, self._expires = "", 0.0

    async def token(self, http: httpx.AsyncClient) -> str:
        if self._token and time.time() < self._expires - 60:
            return self._token
        r = await http.post(TOKEN_URL, json={"appId": self._app_id,
                                             "clientSecret": self._secret})
        data = r.json()
        if "access_token" not in data:
            raise QQApiError(int(data.get("code") or 0), f"token: {data}")
        self._token = data["access_token"]
        self._expires = time.time() + int(data.get("expires_in") or 7200)
        return self._token


def check(data: dict, what: str) -> dict:
    """The platform reports failure in the body; raise on it."""
    code = data.get("code")
    if code not in (None, 0):
        err = int(code)
        if err in _REFUSED:
            raise RecipientUnreachable(f"{what}: {err} {data.get('message')}")
        raise QQApiError(err, f"{what}: {data.get('message')}")
    return data


class QQBot(OutboundChannel):
    def __init__(self, engine, app_id: str, secret: str, *,
                 identities: IdentityMap, link_home: str = "",
                 link_code: str = ""):
        self.engine = engine
        self._auth = QQAuth(app_id, secret)
        self._ids = identities
        # Linking hands an account everything she knows about a person, so it
        # takes proof: the account must send `link_code`, a secret set in the
        # environment, and is then linked to `link_home` ("feishu:oc_xxx").
        # Binding whoever wrote first would give his memory to any other
        # account that reached the bot before he did.
        self._link_home = link_home
        self._link_code = link_code.strip()
        self._seen: dict[str, float] = {}
        self._http: httpx.AsyncClient | None = None
        self._task: asyncio.Task | None = None

    @property
    def channel_name(self) -> str:
        return "qq"

    def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=30)
        return self._http

    async def _api(self, method: str, path: str, body: dict | None = None) -> dict:
        http = self._client()
        token = await self._auth.token(http)
        r = await http.request(method, f"{API_BASE}{path}", json=body,
                               headers={"Authorization": f"QQBot {token}"})
        try:
            data = r.json()
        except ValueError:
            data = {"code": r.status_code, "message": r.text[:200]}
        if r.status_code >= 400 and "code" not in data:
            data["code"] = r.status_code
        return check(data, f"{method} {path}")

    # --- sending --------------------------------------------------------

    async def _send_text(self, openid: str, text: str, *,
                         msg_id: str | None = None, seq: int = 1) -> None:
        body = {"msg_type": 0, "content": text}
        if msg_id:
            body.update(msg_id=msg_id, msg_seq=seq)
        await self._api("POST", f"/v2/users/{openid}/messages", body)

    async def _send_image(self, openid: str, file_path: str, *,
                          msg_id: str | None = None, seq: int = 1) -> None:
        with open(file_path, "rb") as f:
            data = base64.standard_b64encode(f.read()).decode("ascii")
        up = await self._api("POST", f"/v2/users/{openid}/files",
                             {"file_type": 1, "file_data": data,
                              "srv_send_msg": False})
        body = {"msg_type": 7, "media": {"file_info": up["file_info"]}}
        if msg_id:
            body.update(msg_id=msg_id, msg_seq=seq)
        await self._api("POST", f"/v2/users/{openid}/messages", body)

    async def send_message(self, recipient_id: str, text: str,
                           turn_id: str | None = None) -> None:
        for bubble in split_into_bubbles(text):
            await self._send_text(recipient_id, bubble)

    async def send_sticker(self, recipient_id: str, file_path: str) -> None:
        await self._send_image(recipient_id, file_path)

    # --- receiving ------------------------------------------------------

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        asyncio.run_coroutine_threadsafe(self._run(), loop)

    async def _run(self) -> None:
        import websockets
        session, seq, delay = None, None, 1.0
        while True:
            try:
                gw = await self._api("GET", "/gateway")
                async with websockets.connect(gw["url"]) as ws:
                    hello = json.loads(await ws.recv())
                    interval = hello["d"]["heartbeat_interval"] / 1000
                    token = f"QQBot {await self._auth.token(self._client())}"
                    if session:
                        await ws.send(json.dumps({"op": 6, "d": {
                            "token": token, "session_id": session, "seq": seq}}))
                    else:
                        await ws.send(json.dumps({"op": 2, "d": {
                            "token": token, "intents": INTENTS, "shard": [0, 1],
                            "properties": {}}}))
                    beat = asyncio.create_task(self._heartbeat(ws, interval,
                                                               lambda: seq))
                    try:
                        async for raw in ws:
                            p = json.loads(raw)
                            if p.get("s") is not None:
                                seq = p["s"]
                            if p.get("op") == 7:      # server asks to reconnect
                                break
                            if p.get("op") == 9:      # session invalid
                                session = None
                                break
                            if p.get("op") != 0:
                                continue
                            if p.get("t") == "READY":
                                session = p["d"]["session_id"]
                                delay = 1.0
                                print("[qq] connected", flush=True)
                            elif p.get("t") == "C2C_MESSAGE_CREATE":
                                asyncio.create_task(self._on_message(p["d"]))
                    finally:
                        beat.cancel()
                    code = ws.close_code
                if code in _FATAL_CLOSE:
                    print(f"[qq] gateway closed with {code}; not reconnecting",
                          flush=True)
                    return
            except Exception as e:
                print(f"[qq] gateway error: {e!r}", flush=True)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60)

    @staticmethod
    async def _heartbeat(ws, interval: float, last_seq) -> None:
        while True:
            await asyncio.sleep(interval)
            await ws.send(json.dumps({"op": 1, "d": last_seq()}))

    def _try_link(self, openid: str, text: str) -> bool:
        """Link this account if the message is the pairing code."""
        if not (self._link_code and self._link_home):
            return False
        if not hmac.compare_digest(text.encode(), self._link_code.encode()):
            return False
        # Single use: once his account is linked the code links nothing
        # else, so a code that leaks later cannot hand his memory over.
        if self._ids.has_link("qq", self._link_home):
            print("[qq] pairing code reused after linking; ignored", flush=True)
            return False
        self._ids.link("qq", openid, self._link_home)
        print(f"[qq] linked {openid[:8]}… to {self._link_home}", flush=True)
        return True

    async def _on_message(self, d: dict) -> None:
        msg_id = d.get("id", "")
        now = time.time()
        if msg_id in self._seen:          # the gateway may deliver twice
            return
        self._seen = {k: t for k, t in self._seen.items() if now - t < 3600}
        self._seen[msg_id] = now
        openid = (d.get("author") or {}).get("user_openid", "")
        if not openid:
            return
        text = (d.get("content") or "").strip()
        images = await self._images(d.get("attachments") or [])
        if not text and not images:
            text = "（对方发来一个表情包）"
        if self._try_link(openid, text):
            try:
                await self._send_text(openid, "好啦 认出你了！", msg_id=msg_id)
            except Exception as e:
                print(f"[qq] link ack failed: {e!r}", flush=True)
            return
        channel, recipient = self._ids.resolve("qq", openid)
        try:
            await self._reply(openid, msg_id, channel, recipient, text, images)
        except Exception as e:
            print(f"[qq] reply failed: {e!r}", flush=True)
        tracker = getattr(self.engine, "interaction_tracker", None)
        if tracker is not None:
            tracker.set_route(channel, recipient, "qq", openid)
            await tracker.save()

    async def _images(self, attachments: list[dict]) -> list[dict]:
        out = []
        for a in attachments:
            if not str(a.get("content_type", "")).startswith("image"):
                continue
            url = a.get("url", "")
            if url.startswith("//"):
                url = "https:" + url
            try:
                r = await self._client().get(url)
                r.raise_for_status()
            except Exception as e:
                print(f"[qq] image download failed: {e!r}", flush=True)
                continue
            out.append({"media_type": a.get("content_type", "image/png"),
                        "data": base64.standard_b64encode(r.content).decode("ascii")})
        return out

    async def _reply(self, openid: str, msg_id: str, channel: str,
                     recipient: str, text: str, images: list[dict]) -> None:
        speech, sticker = "", None
        async for ev in self.engine.chat_stream_events(
                text, images=images or None, channel=channel,
                recipient_id=recipient):
            if ev.type == "done":
                speech = ev.content
            elif ev.type == "sticker":
                sticker = ev.content
        seq = 0
        for bubble in split_into_bubbles(speech)[:_PASSIVE_REPLIES]:
            seq += 1
            await self._send_text(openid, bubble, msg_id=msg_id, seq=seq)
        if sticker and seq < _PASSIVE_REPLIES:
            try:
                await self._send_image(openid, sticker, msg_id=msg_id, seq=seq + 1)
            except Exception as e:
                print(f"[qq] sticker failed (non-fatal): {e!r}", flush=True)
        print(f"[qq] replied {seq} bubble(s) to {openid[:8]}…", flush=True)
