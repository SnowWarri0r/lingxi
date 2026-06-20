"""Desktop-companion layer: lets the pet sense what the user is doing with
their AI coding agent (Claude Code / Codex / Cursor) and react in-character.

This is the "she's alive on your desktop" half of the pet — distinct from the
sprite window (lingxi.pet). The sensor + companion run inside the bot process
(which owns the persona engine + doubao voice); the pet window just displays
what they surface through /pet/state.
"""
