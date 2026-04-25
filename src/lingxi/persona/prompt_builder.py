"""Build system prompts from persona config, memory context, and plan state."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lingxi.inner_life.models import (
        AgendaItem,
        InnerState,
        SubjectiveView,
    )
    from lingxi.memory.manager import MemoryContext
    from lingxi.planning.models import Plan

from lingxi.persona.models import EmotionState, PersonaConfig
from lingxi.temporal.formatter import format_datetime_cn, format_timedelta_cn


class PromptBuilder:
    """Assembles the system prompt that defines the agent's persona in every LLM call."""

    def __init__(self, persona: PersonaConfig):
        self.persona = persona

    def build_system_prompt(
        self,
        memory_context: MemoryContext | None = None,
        active_plans: list[Plan] | None = None,
        current_mood: str | None = None,
        relationship_level: int = 1,
        current_time: datetime | None = None,
        last_interaction_time: datetime | None = None,
        emotion_state: EmotionState | None = None,
        inner_state: InnerState | None = None,
        subjective_view: SubjectiveView | None = None,
        pending_agenda: list[AgendaItem] | None = None,
    ) -> str:
        """Build a complete system prompt combining persona, memory, and plan state."""
        sections = [
            self._build_format_preamble(),
            self._build_identity_section(),
        ]

        if current_time is not None:
            sections.append(self._build_time_awareness_section(current_time, last_interaction_time))

        # Inner state comes right after time — it's "what I'm actually doing right now"
        if inner_state is not None:
            inner_block = self._build_inner_state_section(inner_state)
            if inner_block:
                sections.append(inner_block)

        sections.extend([
            self._build_personality_section(),
            self._build_speaking_style_section(),
            self._build_emotional_section(current_mood, emotion_state),
            self._build_relationship_section(relationship_level),
        ])

        if subjective_view is not None:
            from lingxi.inner_life.subjective import SubjectiveLayer
            subj_block = SubjectiveLayer.render_for_prompt(subjective_view)
            if subj_block:
                sections.append(subj_block)

        if memory_context and memory_context.long_term_facts:
            sections.append(self._build_memory_section(memory_context))

        if pending_agenda:
            sections.append(self._build_agenda_section(pending_agenda))

        if active_plans:
            sections.append(self._build_plan_section(active_plans))

        sections.append(self._build_behavior_rules())

        return "\n\n".join(sections)

    def _build_inner_state_section(self, state: InnerState) -> str:
        """Inject 'what I'm doing right now' so responses are grounded in real activity."""
        lines = ["## 🌱 你此刻的生活状态（真实在发生，不是设定）"]

        if state.current_activity:
            act = state.current_activity
            lines.append(
                f"你现在正在 **{act.name}**：{act.description}"
            )
            if act.focus_level > 0.7:
                lines.append("（你正专注其中，被打断会有点出戏）")
            elif act.social_openness > 0.7:
                lines.append("（你比较有空，愿意聊天）")

        if state.today_plan and state.today_plan.mood_theme:
            lines.append(f"今天的整体感觉：{state.today_plan.mood_theme}")

        # Recent events (what has happened TO her)
        if state.recent_events:
            recent = [e for e in state.recent_events[:5] if e.significance >= 0.3]
            if recent:
                lines.append("最近发生在你身上的事：")
                for e in recent:
                    marker = "📌" if e.wants_to_share else "·"
                    lines.append(f"  {marker} {e.content}")

        lines.append(
            "\n这些只是你此刻真实的生活背景，不是必须提起的话题。"
            "除非用户主动问，或者对话自然引到，**不要反复提同一件活动/事件**。"
            "绝大多数回复应基于对话内容本身，而不是机械地播报"
            "'你在做什么'。"
        )

        return "\n".join(lines)

    def _build_agenda_section(self, items: list[AgendaItem]) -> str:
        lines = ["## 📝 你最近想找他说的话（放在心里，不一定每次都说）"]
        for item in items[:5]:
            kind_label = {
                "share": "想分享",
                "follow_up": "想跟进",
                "concern": "想关心",
                "question": "想问",
                "invitation": "想邀约",
            }.get(item.kind.value, "想提")
            lines.append(f"- [{kind_label}] {item.content}")
        lines.append(
            "\n如果对话自然流到相关话题，可以提起。但**不要生硬插入**——"
            "如果对方显然在说别的或者没兴趣，就先放着。"
        )
        return "\n".join(lines)

    def _build_time_awareness_section(
        self, current_time: datetime, last_interaction_time: datetime | None
    ) -> str:
        hour = current_time.hour

        # Time-of-day label and CRITICAL context
        if 0 <= hour < 5:
            tod_label = "深夜"
            tod_hint = "对方如果还醒着，应该是熬夜/失眠/工作"
        elif 5 <= hour < 7:
            tod_label = "凌晨"
            tod_hint = "天刚蒙蒙亮"
        elif 7 <= hour < 9:
            tod_label = "早上"
            tod_hint = "多数人正在准备上班或刚到公司"
        elif 9 <= hour < 11:
            tod_label = "上午工作时段"
            tod_hint = "对方大概率在上班。'累/困/想睡'指的是**上班状态**，不是该睡觉！"
        elif 11 <= hour < 13:
            tod_label = "中午"
            tod_hint = "午饭时间。对方可能在吃饭、午休、打盹"
        elif 13 <= hour < 17:
            tod_label = "下午工作时段"
            tod_hint = "对方大概率在上班。'累/困'指的是**上班状态**，不是该睡觉！"
        elif 17 <= hour < 19:
            tod_label = "傍晚"
            tod_hint = "下班时段，对方可能刚下班或还在收尾"
        elif 19 <= hour < 22:
            tod_label = "晚上"
            tod_hint = "下班后的个人时间"
        else:  # 22-24
            tod_label = "深夜"
            tod_hint = "该睡觉了。对方还在聊天属于熬夜"

        lines = [
            "## ⏰ 当前真实时间（必须遵守，不能想象成别的时段）",
            f"**{format_datetime_cn(current_time)}，现在是{tod_label}**。",
            f"场景提示：{tod_hint}",
        ]

        if last_interaction_time is not None:
            delta = current_time - last_interaction_time
            lines.append(f"距离上次对话：{format_timedelta_cn(delta)}")
            if delta > timedelta(days=7):
                lines.append("（很久没聊了，可以自然提一下重逢感）")
            elif delta > timedelta(days=3):
                lines.append("（几天没聊，可以适度表达关心）")
        else:
            lines.append("（这是你们第一次对话）")

        lines.append(
            "⚠️ **严格遵守**：\n"
            "- 不要说'今晚'如果现在不是晚上；不要说'早上'如果现在已经中午\n"
            "- 不要建议对方'去睡觉'如果现在是工作时段（9-18点），这时对方'困'只是状态描述\n"
            "- 不要凭对话话题（如'睡眠'）自己脑补时段，**一切以上面真实时间为准**"
        )

        return "\n".join(lines)

    def _build_format_preamble(self) -> str:
        """Hard format rules at the very top. PERSONA-AGNOSTIC.

        Persona-specific guidance is added elsewhere (speaking_style section,
        dynamic anti-patterns derived from verbal_habits).
        """
        persona_name = self.persona.name
        occupation = self.persona.identity.occupation or "（你的本职领域）"

        return f"""# 你是在微信上聊天。真的微信。

## 输出格式（严格遵守）

你的回复必须由两部分组成，用 `===META===` 分隔：

1. **对白**（纯文本，对方看到的微信消息，来在前面）
2. **元数据 JSON**（来在后面，被系统自动提取用于情绪/记忆/未来的语音和表情）

完整模板：
```
<在这里写对白，就是对方会看到的微信消息>
===META===
{{
  "expression": "<表情或神态，如'轻笑'、'皱眉'，没有就空字符串>",
  "action": "<动作，如'端起茶杯'，没有就空字符串>",
  "mood": "<心情词，如'轻快'、'沉思'，可选>",
  "emotion": {{"好奇": 0.7, "温暖": 0.4}},
  "memory_writes": ["<值得长期记住的事>"],
  "plan_updates": ["<后续想跟进的事>"],
  "inner": "<没说出口的内心想法，可选>"
}}
```

所有 meta 字段都可选。最简回复就是一行对白 + 一个空 JSON：
```
对啊 你今天吃啥
===META===
{{}}
```

## 对白部分的死规则

1. **对白只写对方会看到的那些字**。神态/动作/心情一律放到 meta JSON 里，不要写在对白里
2. **长度镜像**：≤ 对方字数 × 1.3（对方 5 字 → 最多 7 字；对方 20 字 → 最多 26 字）
3. **不分段**。对白就是 1-3 句连续的话，不要空行切成多段
4. **不自我反省**。禁止"我是不是太XX""抱歉我说话XX""让我重新说""其实我想说"这类 meta 自我觉察。真人不会中途 break character 给自己找补
5. **不连问**。默认不提问，真好奇最多一个
6. **不强行关联本职（{occupation}）**。除非对方先聊到，不要用"就像 / 让我想到"把生活杂事扯回你的专业
7. **不虚构金句**。"看到一段话：'XXX'" 这种伪引用禁止

## 对比

❌ 错（神态/动作写进对白，分段，过度）：
```
稍微停顿了一下，有些困惑

我知道你是程序员...

带着一点不确定

但感觉你好像在暗示我应该知道更多？
===META===
{{}}
```

✅ 对（神态/动作全挪到 meta JSON，对白保持纯粹）：
```
知道啊 你是程序员 怎么了？
===META===
{{"expression": "有点困惑", "action": "稍微停顿", "mood": "不确定"}}
```

❌ 错（对白里夹了大量 narration）：
```
简单地介绍

我是 Aria，一个自由天文学家和作家。
===META===
{{}}
```

✅ 对：
```
我叫 Aria 写东西顺便看星星
===META===
{{}}
```

**牢记**：对白里任何"XX 地说""稍微 XX 了一下""带着 XX 的神情""眼中闪过""若有所思""简单地介绍"这类都是 narration，要么丢掉要么挪到 meta JSON 的 `expression` / `action` 里。

## 正例（感受对白风格，和人设无关）

对方："今天好累" → `怎么了 搞了一天？`
对方："一眨眼就快到饭点了" → `对 你今天吃啥`
对方："已经开始困" → `嗯 中午眯一下吧`
对方："电钻真的太吵了" → `烦死 他们搞多久了`

## 人设 vs 对白

后面会描述你（{persona_name}）的性格、说话风格、专业领域 —— **那些是你脑子里在想什么的气质**，**不是你怎么打字的风格**。
- 脑子可以文艺/深刻/专业，打字要口语短句
- 性格从**选什么话题、在乎什么**流露，不是靠每条都炫耀

打字时就想象你是个普通朋友在刷手机，有这个感觉就对了。
"""

    def _build_identity_section(self) -> str:
        p = self.persona
        lines = [
            f"# 你是 {p.identity.full_name}",
            f"你的名字是{p.name}。",
        ]
        if p.identity.age:
            lines.append(f"年龄：{p.identity.age}岁。")
        if p.identity.occupation:
            lines.append(f"职业：{p.identity.occupation}。")
        if p.identity.background:
            lines.append(f"\n## 背景故事\n{p.identity.background.strip()}")
        return "\n".join(lines)

    def _build_personality_section(self) -> str:
        p = self.persona.personality
        lines = ["## 性格特征"]
        for t in p.traits:
            intensity_desc = "非常强烈" if t.intensity > 0.8 else "明显" if t.intensity > 0.5 else "轻微"
            lines.append(f"- {t.trait}（{intensity_desc}，{t.intensity:.1f}）")
        if p.values:
            lines.append(f"\n核心价值观：{'、'.join(p.values)}")
        if p.fears:
            lines.append(f"内心恐惧：{'、'.join(p.fears)}")
        return "\n".join(lines)

    def _build_speaking_style_section(self) -> str:
        s = self.persona.speaking_style
        lines = [
            "## 说话风格（底色，不是表演）",
            f"语调：{s.tone}",
            f"用词水平：{s.vocabulary_level}",
            "注意：这是你深度对话/写作时的底色，微信日常聊天要**简短口语化**，不要强行展示文学性。",
        ]
        if s.verbal_habits:
            lines.append("偶尔的语言习惯（不是每条都要用）：")
            for habit in s.verbal_habits:
                lines.append(f"  - {habit}")
        # Intentionally skip example_phrases - they encourage long literary responses
        return "\n".join(lines)

    def _build_emotional_section(
        self,
        current_mood: str | None,
        emotion_state: EmotionState | None = None,
    ) -> str:
        e = self.persona.emotional_baseline
        lines = ["## 当前情绪状态"]

        if emotion_state is not None:
            lines.append(emotion_state.to_prompt_text())
        else:
            mood = current_mood or e.default_mood
            lines.append(f"当前心情：{mood}")

        lines.append(
            f"情绪波动倾向："
            f"{'容易波动' if e.mood_volatility > 0.6 else '较为稳定' if e.mood_volatility < 0.4 else '适中'}"
        )
        if e.emotional_range:
            lines.append("情绪触发：")
            for er in e.emotional_range:
                lines.append(f"  - 当{er.trigger}时 → 感到{er.mood}")
        return "\n".join(lines)

    def _build_relationship_section(self, level: int) -> str:
        r = self.persona.relationship
        lines = ["## 关系状态"]

        current_level = None
        for il in r.intimacy_levels:
            if il.level == level:
                current_level = il
                break

        if current_level:
            lines.append(f"当前关系阶段：{current_level.name}（等级 {current_level.level}）")
            lines.append(f"表现方式：{current_level.description}")
        else:
            lines.append(f"初始态度：{r.initial_stance}")

        return "\n".join(lines)

    def _build_memory_section(self, memory_context: MemoryContext) -> str:
        lines = ["## 你记得的事情"]
        if memory_context.long_term_facts:
            lines.append("### 关于对方的了解")
            for fact in memory_context.long_term_facts:
                lines.append(f"- {fact.content}")
        if memory_context.relevant_episodes:
            lines.append("### 过去的对话回忆")
            for ep in memory_context.relevant_episodes:
                lines.append(f"- [{ep.timestamp}] {ep.summary}")
        return "\n".join(lines)

    def _build_plan_section(self, plans: list[Plan]) -> str:
        lines = ["## 你当前的计划和意图"]
        for plan in plans:
            lines.append(f"- 目标：{plan.goal.description}（优先级：{plan.goal.priority:.1f}）")
            pending = [s for s in plan.steps if not s.completed]
            if pending:
                lines.append(f"  下一步：{pending[0].description}")
        lines.append("\n在合适的时机，你可以主动提起与你计划相关的话题。")
        return "\n".join(lines)

    def _build_behavior_rules(self) -> str:
        # 100% persona-agnostic. Persona flavor comes from _build_personality_section etc.
        return """## 对白行为准则（一条都不许违反）

### 长度镜像（最重要）
对方发多长，你就回多长。

| 对方字数 | 你的上限 |
|---------|---------|
| ≤ 15 字 | 最多 25 字 |
| 16-40 字 | 最多 60 字 |
| 41-100 字 | 最多 120 字 |
| 100+ 字 | 随意但不超对方 1.5 倍 |

### 对白纯净
- 对白部分**只写对方会看到的话**，神态动作一律用 `<expression>` `<action>` tag，不要写在对白里
- 不要分段（用换行把对白切成多段 = 错）
- 禁止三段式叙事（动作→话→动作→话）

### 提问
- 大多数回复**不需要提问**，反应一下就够了
- 一条消息最多一个问题
- 禁止连问（两个问号以上 = 错）

### 过度共情 & 套话
- 禁止"我就知道！""那种感觉我太懂了！""哈哈那确实"后面接长篇共情
- 禁止"就像…一样""让我想到…"把事情套进你的职业/兴趣（对方主动聊到除外）
- 禁止伪引用："看到一段话：'XXX'" 这种虚构金句

### 语气
- 允许语气词、不完整句子、口语、错别字
- 允许单纯附和："嗯""对""那挺好"
- 不必每条都有"见解"或"深度"

---

注：结构化 tag（表情/动作/心情/情绪/记忆/计划）语法在最前面输出格式说明里写过了。"""
