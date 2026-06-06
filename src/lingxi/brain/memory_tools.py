"""Anthropic tool schemas for MemGPT-style agent memory management.

The agent calls these mid-turn; engine._dispatch_memory_tool executes them,
mapping `scope` to a concrete subject by the current recipient_key so the
subject-ownership invariant holds (the agent cannot target arbitrary subjects).
"""

MEMORY_TOOLS = [
    {
        "name": "archival_memory_search",
        "description": "搜索你的长期记忆（facts.db）。当前上下文里没有、但你需要的细节，用这个查。",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索关键词/问题"},
                "scope": {"type": "string", "enum": ["self", "user", "world"],
                          "description": "self=你自己 user=当前对话对象 world=身边的人和世界"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "archival_memory_insert",
        "description": "把一条值得长期记住的事实写进长期记忆。只在确实重要时用。",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "scope": {"type": "string", "enum": ["self", "user"],
                          "description": "self=关于你自己 user=关于当前对话对象"},
            },
            "required": ["content"],
        },
    },
    {
        "name": "core_memory_append",
        "description": "往常驻核心记忆块追加一行。persona=你的自我小结，human=你对当前对象的长期印象。",
        "input_schema": {
            "type": "object",
            "properties": {
                "block": {"type": "string", "enum": ["persona", "human"]},
                "content": {"type": "string"},
            },
            "required": ["block", "content"],
        },
    },
    {
        "name": "core_memory_replace",
        "description": "替换核心记忆块里的一段文字（用于更新/纠正/精简）。",
        "input_schema": {
            "type": "object",
            "properties": {
                "block": {"type": "string", "enum": ["persona", "human"]},
                "old": {"type": "string", "description": "要替换掉的原文（必须是块里现有的子串）"},
                "new": {"type": "string"},
            },
            "required": ["block", "old", "new"],
        },
    },
    {
        "name": "conversation_search",
        "description": "搜索你和当前对象最近的对话记录。",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "send_sticker",
        "description": (
            "发一张表情包配合你这条消息的情绪。query 用你自己的话描述想发的表情"
            "(如 '无语'、'笑哭'、'摸鱼累了'、'好奇')。偶尔发、贴当下情绪才发,别每句都甩。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
]

TOOL_NAMES = {t["name"] for t in MEMORY_TOOLS}
CORE_BLOCK_MAX_CHARS = 1500
