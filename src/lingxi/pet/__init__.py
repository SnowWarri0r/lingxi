"""Desktop pet — ambient visualization of Aria's current inner state.

Independent PyQt6 process. Polls /pet/state on the running lingxi-feishu
agent and swaps a sprite based on (engagement_mode, emotion_family,
current_activity, hour_of_day). Pet death does not affect IM.

Entry: `lingxi-pet` (see pyproject.toml [project.scripts]).
"""

from lingxi.pet.sprite_mapper import SpriteName, pick_sprite

__all__ = ["SpriteName", "pick_sprite"]
