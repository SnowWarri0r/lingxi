"""Load persona configuration from YAML files."""

from __future__ import annotations

from pathlib import Path

import yaml

from persona_agent.persona.models import PersonaConfig


def load_persona(path: str | Path) -> PersonaConfig:
    """Load and validate a persona configuration from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Persona file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return PersonaConfig.model_validate(data)


def load_persona_from_dict(data: dict) -> PersonaConfig:
    """Load and validate a persona configuration from a dictionary."""
    return PersonaConfig.model_validate(data)
