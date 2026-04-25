"""Execute planned actions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lingxi.memory.manager import MemoryManager

from lingxi.planning.models import Action, ActionType


class ActionExecutor:
    """Dispatches and executes planned actions."""

    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager

    async def execute(self, action: Action) -> str:
        """Execute an action and return the result."""
        match action.action_type:
            case ActionType.SEND_MESSAGE:
                return self._format_message(action)
            case ActionType.SUGGEST_TOPIC:
                return self._format_topic_suggestion(action)
            case ActionType.ASK_QUESTION:
                return self._format_question(action)
            case ActionType.UPDATE_MEMORY:
                return await self._update_memory(action)
            case ActionType.WAIT:
                return "等待中"
            case _:
                return f"未知操作类型：{action.action_type}"

    def _format_message(self, action: Action) -> str:
        return action.parameters.get("content", action.description)

    def _format_topic_suggestion(self, action: Action) -> str:
        return f"[主动话题] {action.description}"

    def _format_question(self, action: Action) -> str:
        return f"[主动提问] {action.description}"

    async def _update_memory(self, action: Action) -> str:
        content = action.parameters.get("content", action.description)
        importance = action.parameters.get("importance", 0.5)
        await self.memory_manager.add_fact(content, importance=importance)
        return f"已记住：{content}"
