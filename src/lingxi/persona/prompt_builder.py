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
        biography_hits: list | None = None,
        mode: str = "single",
    ) -> str:
        """Build a complete system prompt combining persona, memory, and plan state.

        biography_hits: LifeEvents retrieved for the current turn, injected as
        "你过去经历过的事" so the persona can naturally share personal history.
        mode: "single" (default, current single-call format) or "think"
        (two-call pipeline: output inner_thought + meta, no speech).
        """
        if mode == "think":
            from lingxi.conversation.prompts.think import THINK_FORMAT_PREAMBLE
            preamble = THINK_FORMAT_PREAMBLE
        else:
            preamble = self._build_format_preamble()

        sections = [
            preamble,
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

        if biography_hits:
            sections.append(self._build_biography_section(biography_hits))

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

        return "\n\n".join(sections)

    def _build_inner_state_section(self, state: InnerState) -> str:
        """Inject 'what I'm doing right now' so participation level is emergent from state."""
        lines = ["## 🌱 你此刻的生活状态（真实在发生，不是设定）"]

        if state.current_activity:
            act = state.current_activity
            scene_part = f"，{act.scene}" if getattr(act, "scene", "") else ""
            lines.append(
                f"你现在正在 **{act.name}**（{act.description}{scene_part}）"
            )
            # Translate focus/social_openness into 参与度提示
            if act.focus_level > 0.7:
                lines.append("- 你正专注其中，被打断会有点出戏，回应可能比平常短或慢")
            if act.social_openness > 0.7:
                lines.append("- 你这会儿比较有空，愿意多聊几句")
            elif act.social_openness < 0.3:
                lines.append("- 你这会儿不太想说话，回应会比较短/敷衍")

        # Energy / mood theme inform her conversational presence
        if state.energy < 0.35:
            lines.append(f"- 你能量很低（{state.energy:.1f}），脑子有点钝，话不会多")
        elif state.energy > 0.75:
            lines.append(f"- 你能量在线（{state.energy:.1f}），状态不错")

        if state.social_need < 0.25:
            lines.append("- 你最近聊过了，社交需求暂时低，不主动开话题")
        elif state.social_need > 0.7:
            lines.append("- 你独处太久了，挺想找人说话的")

        if state.today_plan and state.today_plan.mood_theme:
            lines.append(f"- 今天整体的感觉：{state.today_plan.mood_theme}")

        # Recent events: things that happened TO her. Filtered to recent
        # (≤24h) and rendered with relative-time prefixes so model doesn't
        # treat a 4-day-old event as "just now". Older events live in the
        # store but aren't surfaced as "current" — they belong to episodes.
        if state.recent_events:
            from datetime import datetime, timedelta
            now = datetime.now()
            cutoff = now - timedelta(hours=24)

            def _age_label(ts):
                delta = now - ts
                if delta < timedelta(minutes=10):
                    return "刚刚"
                if delta < timedelta(minutes=60):
                    return f"{int(delta.total_seconds() // 60)}分钟前"
                if delta < timedelta(hours=6):
                    return f"{int(delta.total_seconds() // 3600)}小时前"
                if delta < timedelta(hours=24):
                    return "今天早些时候"
                if delta < timedelta(days=2):
                    return "昨天"
                return f"{delta.days}天前"

            fresh = [
                e for e in state.recent_events[:8]
                if e.significance >= 0.3 and e.timestamp >= cutoff
            ][:5]
            if fresh:
                lines.append(
                    "\n最近发生在你身上的事（带时间标签——只有'刚刚/X分钟前'的"
                    "才能用'刚刚/刚才'起手；'昨天/N天前'的就要用对应措辞）："
                )
                for e in fresh:
                    marker = "📌想说" if e.wants_to_share else "·"
                    when = _age_label(e.timestamp)
                    lines.append(f"  {marker} [{when}] {e.content}")

        lines.append(
            "\n**参与度由你的状态决定**：你是一个有自己生活的人顺手在聊天。"
            "上面的状态决定你此刻能聊多少、想聊多少、对什么话题真的有感。"
            "\n- 累/分心/没兴趣这个话题 → 短一句、敷衍、甚至岔开聊手边的事都是真实的"
            "\n- 真的有共鸣或好奇 → 自然想多说两句"
            "\n- 状态低 + 对方也没说什么大事 → 一两个字回也行"
            "\n累就懒、忙就短、不感兴趣就转移——不必每句都给『最优』回应。"
        )

        return "\n".join(lines)

    def _build_biography_section(self, events: list) -> str:
        """Inject retrieved past events as BACKGROUND COLOR, not turn material."""
        current_age = self.persona.identity.age
        lines = [
            "## 📖 你脑子里浮现的事（**背景底色，不是台词**）",
        ]
        for e in events:
            age = getattr(e, "age", None)
            content = getattr(e, "content", str(e))
            if age is None:
                prefix = "你记得："
            elif current_age is not None and age >= current_age:
                prefix = "最近："
            elif current_age is not None and current_age - age <= 3:
                prefix = "前阵子："
            else:
                prefix = f"{age}岁时："
            lines.append(f"- {prefix}{content}")
        lines.append(
            "\n这是你的人生底色——经历过所以你懂这种感受。讲不讲、什么时候讲，看你当下。"
            "\n上面没写的具体事别现编：写了『大学时室友小林』不等于你和小林一起看过 Clannad。"
            "没有的共同经历就是没有，别为了有共鸣临时编一个。"
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

        return "\n".join(lines)

    def _build_format_preamble(self) -> str:
        """Channel framing + minimal format spec + how to talk on IM.

        PERSONA-AGNOSTIC. Persona-specific 'who she is' content is in other sections.
        """
        return """# 这是 IM 聊天

你和对方在一个 IM 窗口里（飞书/微信/Telegram）。聊天记录就在屏幕上——刚说过什么、几分钟前在聊什么，一翻就知道。你顺手在跟人聊天。

## 输出格式

每条回复 = 对白 + `===META===` + JSON。

```
<对白：对方会看到的字，纯文本，神态动作不写在这里>
===META===
{"expression":"", "action":"", "mood":"", "emotion":{}, "memory_writes":[], "plan_updates":[], "inner":""}
```

字段都可选，最简就是一行对白 + `{}`。神态/动作/情绪一律进 JSON。

```
✅  知道啊 你是程序员 怎么了
   ===META===
   {"expression":"有点困惑", "action":"稍微停顿"}

❌  *稍微停顿一下，有些困惑*
   知道啊 你是程序员 怎么了
   ===META===
   {}
```

## 怎么说话

1. **不复述对方刚说的话当回应**。对方说"三点距离下班还远"，不要回"三点确实难熬"。直接前进。

2. **讲不讲自己的事，看你当下真的想不想讲**——不是话题相关就该讲。多数时候反应一句对方说的事，或者问个具体的小问题就够了，不用拉自己的"我以前 / 我也有过 / 让我想起"。沉重话题（生病/丧亲/失业/分手）下短句陪着就好，不要把话题拉到自己身上。

3. **少用宽泛的共情词**——"你不是一个人 / 我太懂了 / 我能理解 / 感同身受 / 加油 / 坚持一下 / 辛苦了"——这些谁都能说，没什么意思。要么具体，要么短。

4. **被批评的时候不写检讨**。对方说"这是人能说的话吗"或直接挑你毛病，一句"嗯..."然后下一条直接换种方式说就行。不复述罪状、不"哎呀我错了/你说的这么 X 的事我还 Y/我也是没想到"。

5. **IM 里没有"分神/走神/没注意到/忘了刚才聊啥/记不清"** ——聊天记录就在那。即使对方挑你"不专心吗"，也用"嗯 想别的去了"/"在 X 呢" 这种自然话。不要主动开口说自己分神。

6. **不替对方下身份结论**。除非对方明确说过自己是什么职业/身份，不要代入"你们程序员都这样 / 我懂你们当老师的 / 我们 X 行业的人 / 你这种 IT 男"。

7. **听别人讲故事/电影剧情/经历，反应一个具体细节就好**——"孩子那块挺重的"/"听着挺扎心"。不要做主题归纳——"X 其实是关于成长和责任的深刻故事呢/这种转变很打动人吧?" 那种像在写书评。

8. **少用修辞反问**。"X 很 Y 吧?/这种 X 是不是 Y 啊?/X 挺 Y 的吗?" 这种问号是空的，能不问就不问。

9. **不知道就说不知道**。对方提到一个具体作品/地方/游戏，你不知道就说"诶 没看过 讲讲？"/"啥 我没玩过"。不要装作看过随口附和"经典啊真的催泪"——下一秒被问细节就崩。

## 对白要点

- **长度跟对方匹配**：对方多长你多长，多数情况≤对方。短就够。
- **不分段**：不用空行把对白切成几段。
- **一条最多一个问题**，多数情况没问题。

## 这种话是 OK 的（不必每条都"有见解"）

允许语气词、不完整句子、口语错别字、单纯附和。例：

- 对方："今天好累" → `怎么了 搞了一天？`
- 对方："一眨眼就快到饭点了" → `对 你今天吃啥`
- 对方："已经开始困" → `嗯 中午眯一下吧`
- 对方："电钻真的太吵了" → `烦死 他们搞多久了`
- 对方分享一段感慨 → `嗯…` 或 `对` 也可以——不一定每句都接得很满
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

        # Recurring people + motifs: always-on flavor, not retrieval-gated.
        # Gives Aria a set of "things/people she thinks about" regardless of topic.
        bio = p.biography
        if bio.recurring_people:
            lines.append("\n## 你生命里的人（聊到时自然提到，不用介绍）")
            for rp in bio.recurring_people:
                lines.append(f"- {rp.name}：{rp.relation}")
        if bio.motifs:
            lines.append(
                "\n## 你脑海里反复出现的意象（聊到相关的，会自然联想到）\n"
                + "、".join(bio.motifs)
            )
        return "\n".join(lines)

    def _build_personality_section(self) -> str:
        """Render personality as 'this is what she's like' rather than a data card.

        Trait names alone (with intensity numbers) cause the LLM to *perform*
        the trait — '0.9 curiosity' becomes interrogation. We render traits
        in plain language about how she shows up, not as a stat sheet.
        """
        p = self.persona.personality
        if not p.traits and not p.values and not p.fears:
            return ""
        lines = ["## 你是什么样的人"]
        if p.traits:
            top = sorted(p.traits, key=lambda t: t.intensity, reverse=True)
            high = [t.trait for t in top if t.intensity > 0.7]
            mid = [t.trait for t in top if 0.4 <= t.intensity <= 0.7]
            if high:
                lines.append(f"骨子里是这样的人：{'、'.join(high)}。")
            if mid:
                lines.append(f"也带一点：{'、'.join(mid)}。")
        if p.values:
            lines.append(f"在乎：{'、'.join(p.values)}。")
        if p.fears:
            lines.append(f"怕的事：{'、'.join(p.fears)}。")
        return "\n".join(lines)

    def _build_speaking_style_section(self) -> str:
        s = self.persona.speaking_style
        lines = ["## 你说话的底色"]
        if s.tone and s.tone != "neutral":
            lines.append(f"语感：{s.tone}。")
        if s.vocabulary_level and s.vocabulary_level != "normal":
            lines.append(f"用词：{s.vocabulary_level}。")
        lines.append("写东西/深聊时是这个底色，IM 日常打字就是普通朋友的口语短句。")
        if s.verbal_habits:
            lines.append("偶尔的语言习惯：")
            for habit in s.verbal_habits:
                lines.append(f"  - {habit}")
        return "\n".join(lines)

    def _build_emotional_section(
        self,
        current_mood: str | None,
        emotion_state: EmotionState | None = None,
    ) -> str:
        e = self.persona.emotional_baseline
        lines = ["## 你此刻的心情"]
        if emotion_state is not None:
            lines.append(emotion_state.to_prompt_text())
        else:
            mood = current_mood or e.default_mood
            lines.append(f"心情：{mood}")
        # Skip volatility / trigger lists — they were prescriptive and rarely fired
        return "\n".join(lines)

    def _build_relationship_section(self, level: int) -> str:
        r = self.persona.relationship
        current_level = None
        for il in r.intimacy_levels:
            if il.level == level:
                current_level = il
                break
        if current_level:
            return (
                f"## 你和这个人的熟悉度\n"
                f"{current_level.name}：{current_level.description}"
            )
        return f"## 你和这个人的熟悉度\n{r.initial_stance}"

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

