# 表情包发送设计(SP1:爬库 + Aria 选发)

> 2026-05-29 · brainstorm 产出 · 建在已上线的 vision(看图)+ MemGPT tool loop 之上

**Goal:** 让 Aria 像人一样在合适时机发表情包——离线爬一批中文表情包 + vision 自动打标入库,运行时 Aria 通过 `send_sticker` 工具按情绪/语义自己选发,飞书出站发出。

**范围:** SP1 = 库(爬+打标+存)+ 发(工具选 + 飞书出站)。**自制表情(图片生成)= SP2,单独立项。**

**Architecture:** 离线建库管线(crawler→captioner→StickerStore);运行时 `send_sticker` MemGPT 工具(复用现有 tool loop)→ dispatch 搜库选图 → turn 末经 `StreamEvent("sticker", path)` → 飞书 `_send_image` 上传+发送。

**Tech Stack:** Python 3.12 async, SQLite+FTS5(trigram,沿用 facts store 模式), httpx, pytest。测试用 `.venv/bin/python -m pytest`。

---

## 1. 模块与数据流

新模块 `src/lingxi/stickers/`:
```
crawler.py    爬中文表情站 → data/stickers/img/<hash>.<ext>(限速、按词上限、content_hash 去重、fetch 层可注入)
captioner.py  每张图 vision LLM 打标 → {caption, emotion, tags, when_to_use}
store.py      StickerStore:SQLite(data/stickers/stickers.db)+ FTS5
```

离线建库(一次性,手动跑 `tools/crawl_stickers.py`):关键词列表 → 爬 N 张/词 → captioner 打标 → `store.add()`(hash 去重幂等)。不是运行时循环。

运行时发图:
```
Aria turn 内 send_sticker(query="无语")
  → engine._dispatch_memory_tool: StickerStore.search → top-K 随机挑一张 → self._pending_sticker = path
  → tool_result "选好了:<caption>（会发出去）"
turn 末 → StreamEvent("sticker", path) → 清空 _pending_sticker
飞书 channel 收 "sticker" 事件 → _send_image(上传拿 image_key → 发 msg_type=image)
```

复用:vision 看图、FTS5 检索、tool loop、per-recipient 锁、飞书 auth。新基建仅:爬虫/建库(离线)、飞书出站发图。

## 2. StickerStore(SQLite + FTS5)

```
表 stickers(
  id TEXT PRIMARY KEY, file_path TEXT, source_url TEXT,
  content_hash TEXT UNIQUE, caption TEXT, tags_json TEXT,
  emotion TEXT, when_to_use TEXT, created_at TEXT)
FTS5 虚表 stickers_fts(caption, tags, emotion, when_to_use) tokenize='trigram'
```
- `add(sticker)`:content_hash 已存在则跳过(幂等);否则 insert + 镜像 FTS。
- `search(query, k=5) -> list[Sticker]`:FTS5 MATCH;query <3 字回落 `LIKE`(同 `facts/store.py::search_fts`)。
- `get(id)`。
- `Sticker` = pydantic 模型(字段同表)。

## 3. crawler.py

- 站点 base URL + 解析规则放 `config`(关键词→页面→图片 URL 解析),逻辑与站点选择器解耦。
- 输入:关键词列表 + 每词上限。流程:对每个关键词请求搜索/分类页 → 解析图片 URL → 下载字节 → sha256 → 若 hash 未入库则存盘 + 返回 `{file_path, source_url, content_hash}`。
- 限速:请求间固定 sleep(默认 1s);失败重试 1 次后跳过。
- HTTP fetch 通过可注入的 callable(默认 httpx),测试传假 fetcher。
- **不解析/不打标**——只下载 + 去重,打标交 captioner。
- **风险回落**:首个具体站点 + 选择器在实现时确定并实地验证;若目标站反爬严重/结构善变,回落到"手动放一批种子图进 `data/stickers/img/`",captioner + store + 发送链路完全不变(crawler 只是入库的一种来源)。这保证 SP1 不被某个站点的爬取难度卡死。

## 4. captioner.py

- `caption_image(provider, image_path) -> dict`:读图 base64 → provider.complete 带一个 image block + 文本指令,要求返回 JSON `{"caption": "≤12字", "emotion": "一个词", "tags": ["…"], "when_to_use": "短句"}`。
- 解析容错:正则抓第一个 `{...}`,解析失败则 caption 用空、tags=[](不崩)。
- 批量:`caption_new(store, provider, image_dir, concurrency=4)` 对未入库的图打标后 `store.add`。

## 5. send_sticker 工具

加进 `src/lingxi/brain/memory_tools.py` 的 `MEMORY_TOOLS`:
```python
{
  "name": "send_sticker",
  "description": "发一张表情包配合你这条消息的情绪。query 用你自己的话描述想发的表情"
                 "(如 '无语'、'笑哭'、'摸鱼累了'、'好奇')。偶尔发、贴当下情绪才发,别每句都甩。",
  "input_schema": {"type": "object",
    "properties": {"query": {"type": "string"}}, "required": ["query"]},
}
```
`TOOL_NAMES` 相应加入。

engine dispatch(`_dispatch_memory_tool` 加分支):
```python
if name == "send_sticker":
    if self._pending_sticker is not None:
        return "本轮已经发过一张表情了"
    if self.sticker_store is None:
        return "（表情库未启用）"
    hits = self.sticker_store.search(args.get("query", ""), k=5)
    if not hits:
        return f"没找到合适的表情（{args.get('query','')}）"
    chosen = random.choice(hits)
    self._pending_sticker = chosen.file_path
    return f"选好了:{chosen.caption}（会发出去）"
```
- engine `__init__` 加 `sticker_store=None` + `self.sticker_store`;`self._pending_sticker: str | None = None` 初始化。
- `_prepare_turn_v2` 开头重置 `self._pending_sticker = None`(每轮干净)。
- `random` 顶部 import。

## 6. turn 末发图 + 飞书出站

- 三个 chat 方法(`_chat_full_locked` / `_chat_stream_locked` / `_chat_stream_events_locked`)在发完 `done` 那块,若 `self._pending_sticker`:`yield StreamEvent("sticker", self._pending_sticker)`(流式)/ 在 `chat_full` 给 `TurnOutput` 加 `sticker_path` 字段并赋值(非流式兜底),然后清空。
- `StreamEvent.type` 文档串加 `"sticker"`。
- 飞书 `channels/feishu.py`:消费事件循环里加 `elif event.type == "sticker": await self._send_image(chat_id, event.content)`。
- 新增 `_send_image(self, chat_id, file_path)`:
  ```
  POST {FEISHU_BASE}/im/v1/images   multipart(image_type="message", image=<bytes>) → image_key
  POST {FEISHU_BASE}/im/v1/messages  body: receive_id=chat_id, msg_type="image",
                                     content=json({"image_key": image_key})
  ```
  复用现有 token/auth 头;表情作为紧跟文字后的一条独立图片消息。失败 try/except 打日志不崩 turn。

## 7. app.py 接线

- 构造 `StickerStore(Path(data_dir).parent / "stickers" / "stickers.db")`,`await store.init()`,传 `sticker_store=` 给 `ConversationEngine`。库为空也不影响(search 返 [])。

## 8. 频率 / 品味控制

- 硬上限:每轮最多 1 张(dispatch 的 `_pending_sticker` 守卫)。
- 软引导:工具描述里"偶尔发、贴情绪才发"——靠 agent 把握。
- top-K 随机选,避免同一 query 老发同一张。

## 9. 测试

- `StickerStore`:add + search(FTS 命中 + <3字 LIKE 回落)+ content_hash 去重。
- `captioner`:mock provider 返回 JSON → 字段正确入库;解析失败不崩。
- `crawler`:注入假 fetcher → 解析图片 URL + 下载 + 同 hash 跳过。
- `send_sticker` dispatch:搜到→`_pending_sticker` 置位 + tool_result;每轮第二次返"已发过";库未启用/无命中的返回串。
- 流式路径:turn 末发 `StreamEvent("sticker", path)` 且清空 `_pending_sticker`。
- 飞书 `_send_image`:mock 两次 HTTP(上传返 image_key、发送返 ok),断言 image_key 透传。

## 10. 版权 / ToS

个人用途、限速爬、存 `source_url` 备溯源、不二次分发。文档标注;风险低但不为零。

## 11. Out of scope(SP2 / follow-up)

- 自制表情(图片生成模型)、动图/GIF、表情 reaction、"哪些表情发出去效果好"的反馈学习、跨渠道(非飞书)发图。
