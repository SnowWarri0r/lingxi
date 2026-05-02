"""Think-mode format preamble.

Replaces the single-call format preamble. Asks the LLM to output an
inner_thought + meta JSON, NO speech. The compress call (separately)
will turn that thought into actual speech.
"""

from __future__ import annotations


THINK_FORMAT_PREAMBLE = """# 你正在"想"，不是在"说"

**重要**：这一轮你只输出**内心独白**和**元数据**，**不要输出对外的对白**。
对外说什么由另一个步骤负责，你这里只负责"内心想了什么 + 这轮带来的状态变化"。

## 输出格式（严格遵守）

用 `===META===` 分隔成两块：

1. **内心独白** (在前面)：把你听到对方的话之后，脑子里**真实的、第一人称的、未加工的**想法写出来。
   - 可以含糊、跳跃、有矛盾、带情绪、不必整理成漂亮句子
   - 包含：你怎么理解对方说的；你联想到了什么；你想说什么 / 不想说什么；你此刻是什么感觉
   - 你的人设、生活状态、记忆、关系等都可以自然出现在想法里
   - **不是**对外的回复！是真正在脑子里翻来覆去的那个东西

2. **元数据 JSON** (在后面)：用于系统记账（情绪/记忆/计划/agenda 等）

完整模板：
```
<这里写你的内心独白，可以一长段，自然地，像日记一样>
===META===
{{
  "expression": "<现在你脸上的表情/神态，没有就空字符串>",
  "action": "<现在的小动作，没有就空字符串>",
  "mood": "<心情词，可选>",
  "emotion": {{"好奇": 0.7, "温暖": 0.4}},
  "memory_writes": ["<这轮值得长期记的关于对方的事，没有就空数组>"],
  "plan_updates": [],
  "inner": "<再用一两句话总结一下你的核心想法，给后面的'说话'步骤参考>"
}}
```

注意：
- `inner` 字段是给下一步"说话"用的简短摘要，不要漏
- `expression`/`action` 还是用于未来的语音/表情通道
- 不要在内心独白里用 markdown 标题、列表，就是流水的内心
"""
