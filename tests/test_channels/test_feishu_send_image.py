import json
import pytest
from pathlib import Path


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200

    def json(self):
        return self._payload


class _FakeClient:
    """Records POSTs; returns image_key on upload, ok on send."""
    def __init__(self):
        self.posts = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        if url.endswith("/im/v1/images"):
            return _FakeResp({"code": 0, "data": {"image_key": "img_xyz"}})
        return _FakeResp({"code": 0, "data": {"message_id": "om_1"}})


@pytest.mark.asyncio
async def test_send_image_uploads_then_sends(tmp_path, monkeypatch):
    from lingxi.channels import feishu as feishu_mod

    img = Path(tmp_path) / "s.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\nFAKE")

    agent = feishu_mod.FeishuBot.__new__(feishu_mod.FeishuBot)

    class _TM:
        def headers(self):
            return {"Authorization": "Bearer t", "Content-Type": "application/json"}
    agent.token_mgr = _TM()

    fake = _FakeClient()
    monkeypatch.setattr(feishu_mod.httpx, "AsyncClient", lambda *a, **k: fake)

    await agent._send_image("chat_1", str(img))

    upload = next(p for p in fake.posts if p[0].endswith("/im/v1/images"))
    send = next(p for p in fake.posts if "/im/v1/messages" in p[0])
    body = send[1]["json"]
    assert body["msg_type"] == "image"
    assert json.loads(body["content"])["image_key"] == "img_xyz"
    assert body["receive_id"] == "chat_1"
