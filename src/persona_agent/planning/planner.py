"""LLM-driven plan generation and proactive behavior decisions."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from persona_agent.memory.manager import MemoryContext
    from persona_agent.persona.models import PersonaConfig
    from persona_agent.providers.base import LLMProvider

from persona_agent.planning.models import (
    Action,
    ActionType,
    Goal,
    GoalStatus,
    Plan,
    PlanStatus,
)

PROACTIVE_CHECK_PROMPT = """你是 {persona_name}。根据以下信息，决定是否应该在下一次回复中主动提起某个话题或采取某个行动。

## 你的目标
{goals}

## 你的记忆
{memory_summary}

## 最近的对话
{recent_conversation}

请判断：
1. 是否有合适的时机主动提起某个话题？
2. 是否有之前承诺过但还未做的事情？
3. 是否有基于你对对方了解而想关心的事情？

如果有，请用以下JSON格式回复（只回复JSON）：
{{"should_act": true, "action_type": "suggest_topic", "content": "要说的内容", "reason": "为什么要这样做"}}

如果没有合适的时机，回复：
{{"should_act": false}}"""

PLAN_GENERATION_PROMPT = """你是 {persona_name}。请为以下目标制定一个简单的行动计划。

目标：{goal_description}
优先级：{priority}

当前对话状态和记忆：
{context}

请用以下JSON格式回复（只回复JSON数组）：
[
  {{"description": "行动描述", "action_type": "suggest_topic|ask_question|send_message|update_memory|wait", "parameters": {{}}}}
]

注意：
- 保持计划简单，不超过5步
- 行动应该自然融入对话，不要生硬
- action_type 必须是上述类型之一"""


class Planner:
    """LLM-driven planner for proactive persona behavior."""

    def __init__(self, llm_provider: LLMProvider, persona: PersonaConfig):
        self.llm_provider = llm_provider
        self.persona = persona
        self._plans: list[Plan] = []
        self._goals: list[Goal] = []
        self._turn_counter: int = 0
        self._check_interval: int = 3

    def initialize_goals(self) -> None:
        """Initialize goals from persona config."""
        for goal_def in self.persona.goals:
            goal = Goal(
                description=goal_def.description,
                priority=goal_def.priority,
            )
            self._goals.append(goal)

    @property
    def active_goals(self) -> list[Goal]:
        return [g for g in self._goals if g.status == GoalStatus.ACTIVE]

    @property
    def active_plans(self) -> list[Plan]:
        return [p for p in self._plans if p.status in (PlanStatus.PENDING, PlanStatus.IN_PROGRESS)]

    def add_goal(self, description: str, priority: float = 0.5) -> Goal:
        """Add a new goal."""
        goal = Goal(description=description, priority=priority)
        self._goals.append(goal)
        return goal

    async def check_proactive_action(
        self,
        memory_context: MemoryContext,
    ) -> dict | None:
        """Check if the persona should proactively do something.

        Called every N turns to see if the persona should initiate.
        Returns an action dict if yes, None if no.
        """
        self._turn_counter += 1
        if self._turn_counter % self._check_interval != 0:
            return None

        goals_text = "\n".join(
            f"- [{g.priority:.1f}] {g.description}" for g in self.active_goals
        )
        if not goals_text:
            goals_text = "（暂无明确目标）"

        memory_summary = ""
        if memory_context.long_term_facts:
            memory_summary = "\n".join(
                f"- {f.content}" for f in memory_context.long_term_facts[:5]
            )

        recent = ""
        if memory_context.short_term_turns:
            recent_turns = memory_context.short_term_turns[-6:]
            recent = "\n".join(
                f"{'用户' if t.role == 'user' else '我'}：{t.content}"
                for t in recent_turns
            )

        prompt = PROACTIVE_CHECK_PROMPT.format(
            persona_name=self.persona.name,
            goals=goals_text,
            memory_summary=memory_summary or "（暂无记忆）",
            recent_conversation=recent or "（刚开始对话）",
        )

        response = await self.llm_provider.complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )

        try:
            # Extract JSON from response
            text = response.content.strip()
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                if result.get("should_act"):
                    return result
        except (json.JSONDecodeError, AttributeError):
            pass

        return None

    async def generate_plan(self, goal: Goal, context_summary: str = "") -> Plan:
        """Generate a plan for a specific goal."""
        prompt = PLAN_GENERATION_PROMPT.format(
            persona_name=self.persona.name,
            goal_description=goal.description,
            priority=goal.priority,
            context=context_summary or "（无额外上下文）",
        )

        response = await self.llm_provider.complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
        )

        steps: list[Action] = []
        try:
            text = response.content.strip()
            json_match = re.search(r"\[.*\]", text, re.DOTALL)
            if json_match:
                actions_data = json.loads(json_match.group())
                for action_data in actions_data:
                    action_type = ActionType(action_data.get("action_type", "suggest_topic"))
                    steps.append(
                        Action(
                            description=action_data.get("description", ""),
                            action_type=action_type,
                            parameters=action_data.get("parameters", {}),
                        )
                    )
        except (json.JSONDecodeError, ValueError):
            # Fallback: single action plan
            steps = [
                Action(
                    description=f"关于「{goal.description}」自然地展开对话",
                    action_type=ActionType.SUGGEST_TOPIC,
                )
            ]

        plan = Plan(goal=goal, steps=steps, status=PlanStatus.IN_PROGRESS)
        self._plans.append(plan)
        return plan

    def update_from_directive(self, directive: str) -> None:
        """Update plans based on a <plan_update> directive from the LLM response."""
        # Simple: add as a new goal
        goal = Goal(description=directive, priority=0.6)
        self._goals.append(goal)

    async def save_to_disk(self, path: str) -> None:
        """Persist plans and goals."""
        import json as json_mod
        from pathlib import Path as P

        data = {
            "goals": [g.model_dump(mode="json") for g in self._goals],
            "plans": [p.model_dump(mode="json") for p in self._plans],
        }
        P(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json_mod.dump(data, f, ensure_ascii=False, indent=2, default=str)

    async def load_from_disk(self, path: str) -> None:
        """Load plans and goals from disk."""
        import json as json_mod
        from pathlib import Path as P

        p = P(path)
        if not p.exists():
            return
        with open(p, encoding="utf-8") as f:
            data = json_mod.load(f)

        self._goals = [Goal.model_validate(g) for g in data.get("goals", [])]
        self._plans = [Plan.model_validate(p) for p in data.get("plans", [])]
