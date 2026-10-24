"""Check a QQ bot's credentials and whether it may send openers.

    python scripts/qq_probe.py                 # token + gateway only
    python scripts/qq_probe.py <user_openid>   # also send one active message

The open question for a personal bot is whether active (proactive) messages
are allowed at all; the answer is the platform's reply to the second form.
Reads QQ_BOT_APP_ID / QQ_BOT_APP_SECRET from the environment.
"""

import asyncio
import os
import sys

import httpx

from lingxi.channels.qq import API_BASE, QQAuth


async def main() -> None:
    auth = QQAuth(os.environ["QQ_BOT_APP_ID"], os.environ["QQ_BOT_APP_SECRET"])
    async with httpx.AsyncClient(timeout=20) as http:
        token = await auth.token(http)
        print("token ok")
        h = {"Authorization": f"QQBot {token}"}
        print("gateway:", (await http.get(f"{API_BASE}/gateway", headers=h)).json())
        if len(sys.argv) > 1:
            r = await http.post(f"{API_BASE}/v2/users/{sys.argv[1]}/messages",
                                headers=h, json={"msg_type": 0, "content": "测试一下主动消息"})
            print("active message:", r.status_code, r.text[:300])


asyncio.run(main())
