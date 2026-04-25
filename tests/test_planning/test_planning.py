"""Tests for the planning system."""

import pytest

from persona_agent.planning.models import Action, ActionType, Goal, GoalStatus, Plan, PlanStatus
from persona_agent.planning.scheduler import Scheduler


class TestPlanModels:
    def test_plan_next_action(self):
        plan = Plan(
            goal=Goal(description="Test goal"),
            steps=[
                Action(description="Step 1", action_type=ActionType.SUGGEST_TOPIC),
                Action(description="Step 2", action_type=ActionType.ASK_QUESTION),
            ],
        )

        next_action = plan.next_action()
        assert next_action is not None
        assert next_action.description == "Step 1"

    def test_complete_current_action(self):
        plan = Plan(
            goal=Goal(description="Test goal"),
            steps=[
                Action(description="Step 1", action_type=ActionType.SUGGEST_TOPIC),
                Action(description="Step 2", action_type=ActionType.ASK_QUESTION),
            ],
        )

        plan.complete_current("Done")
        assert plan.steps[0].completed
        assert plan.next_action().description == "Step 2"

    def test_plan_completes_when_all_done(self):
        plan = Plan(
            goal=Goal(description="Test goal"),
            steps=[
                Action(description="Step 1", action_type=ActionType.SUGGEST_TOPIC),
            ],
        )

        plan.complete_current()
        assert plan.status == PlanStatus.COMPLETED
        assert plan.goal.status == GoalStatus.COMPLETED


class TestScheduler:
    def test_event_triggers(self):
        scheduler = Scheduler()
        scheduler.register_event_trigger(
            keyword="travel",
            action_description="Suggest stargazing spots",
        )

        triggered = scheduler.check_event_triggers("I'm planning to travel next week")
        assert len(triggered) == 1
        assert "stargazing" in triggered[0]

    def test_one_shot_trigger(self):
        scheduler = Scheduler()
        scheduler.register_event_trigger(
            keyword="birthday",
            action_description="Wish happy birthday",
            one_shot=True,
        )

        triggered1 = scheduler.check_event_triggers("It's my birthday!")
        assert len(triggered1) == 1

        triggered2 = scheduler.check_event_triggers("Remember my birthday?")
        assert len(triggered2) == 0
