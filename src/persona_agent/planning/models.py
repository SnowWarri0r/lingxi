"""Data models for the planning system."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class GoalStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class PlanStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ActionType(str, Enum):
    SEND_MESSAGE = "send_message"  # Proactively say something
    UPDATE_MEMORY = "update_memory"  # Store something in memory
    ASK_QUESTION = "ask_question"  # Ask the user a question
    SUGGEST_TOPIC = "suggest_topic"  # Suggest a topic of conversation
    WAIT = "wait"  # Wait for a condition


class Goal(BaseModel):
    """A goal the persona wants to achieve."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    description: str
    priority: float = Field(default=0.5, ge=0.0, le=1.0)
    status: GoalStatus = GoalStatus.ACTIVE
    created_at: datetime = Field(default_factory=datetime.now)
    deadline: datetime | None = None
    metadata: dict = Field(default_factory=dict)


class Action(BaseModel):
    """A single action within a plan."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    description: str
    action_type: ActionType
    parameters: dict = Field(default_factory=dict)
    scheduled_at: datetime | None = None
    completed: bool = False
    result: str | None = None


class Plan(BaseModel):
    """A plan to achieve a goal, consisting of ordered actions."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    goal: Goal
    steps: list[Action] = Field(default_factory=list)
    status: PlanStatus = PlanStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def next_action(self) -> Action | None:
        """Get the next uncompleted action."""
        for step in self.steps:
            if not step.completed:
                return step
        return None

    def complete_current(self, result: str = "") -> None:
        """Mark the current action as completed."""
        action = self.next_action()
        if action:
            action.completed = True
            action.result = result
            self.updated_at = datetime.now()

        # Check if all steps are done
        if all(s.completed for s in self.steps):
            self.status = PlanStatus.COMPLETED
            self.goal.status = GoalStatus.COMPLETED
