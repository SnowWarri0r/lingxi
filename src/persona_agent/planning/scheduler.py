"""Time-based and event-based action scheduling."""

from __future__ import annotations

from datetime import datetime

from persona_agent.planning.models import Action, Plan


class Scheduler:
    """Manages scheduled actions and event triggers."""

    def __init__(self):
        self._event_triggers: list[dict] = []

    def get_due_actions(self, plans: list[Plan]) -> list[tuple[Plan, Action]]:
        """Get all actions that are due to be executed now."""
        now = datetime.now()
        due: list[tuple[Plan, Action]] = []

        for plan in plans:
            action = plan.next_action()
            if action and not action.completed:
                if action.scheduled_at is None or action.scheduled_at <= now:
                    due.append((plan, action))

        # Sort by goal priority (highest first)
        due.sort(key=lambda x: x[0].goal.priority, reverse=True)
        return due

    def register_event_trigger(
        self,
        keyword: str,
        action_description: str,
        one_shot: bool = True,
    ) -> None:
        """Register a trigger that fires when a keyword appears in conversation."""
        self._event_triggers.append({
            "keyword": keyword.lower(),
            "action_description": action_description,
            "one_shot": one_shot,
            "fired": False,
        })

    def check_event_triggers(self, user_message: str) -> list[str]:
        """Check if any event triggers match the user's message.

        Returns list of action descriptions to inject.
        """
        message_lower = user_message.lower()
        triggered: list[str] = []

        for trigger in self._event_triggers:
            if trigger["fired"] and trigger["one_shot"]:
                continue
            if trigger["keyword"] in message_lower:
                triggered.append(trigger["action_description"])
                trigger["fired"] = True

        return triggered
