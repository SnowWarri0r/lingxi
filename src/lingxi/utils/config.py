"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def load_config(config_path: str | Path = "config/default.yaml") -> dict[str, Any]:
    """Load configuration from YAML file with environment variable overrides."""
    load_dotenv()

    path = Path(config_path)
    if not path.exists():
        return _default_config()

    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    return config


def _default_config() -> dict[str, Any]:
    return {
        "llm": {
            "provider": "claude",
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 4096,
            "temperature": 0.7,
        },
        "memory": {
            "short_term": {"max_turns": 30},
            "long_term": {"max_entries": 10000, "retrieval_top_k": 10},
            "episodic": {"max_episodes": 500},
        },
        "planning": {"enabled": True},
    }


def get_nested(config: dict, *keys: str, default: Any = None) -> Any:
    """Safely get a nested config value."""
    current = config
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
        else:
            return default
        if current is None:
            return default
    return current
