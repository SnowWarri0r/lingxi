"""Load seed FewShotSamples from a YAML file."""

from __future__ import annotations

from pathlib import Path

import yaml

from lingxi.fewshot.models import FewShotSample


def load_seeds(path: str | Path) -> list[FewShotSample]:
    """Parse a seeds YAML file into FewShotSamples.

    Expected schema:
        seeds:
          - id: <str>
            context_summary: <str>
            inner_thought: <str>
            corrected_speech: <str>
            tags: [<str>, ...]
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw_seeds = (data or {}).get("seeds", [])

    samples: list[FewShotSample] = []
    for entry in raw_seeds:
        samples.append(FewShotSample(
            id=entry["id"],
            inner_thought=entry.get("inner_thought", ""),
            corrected_speech=entry["corrected_speech"],
            context_summary=entry["context_summary"],
            tags=list(entry.get("tags", [])),
            recipient_key=None,
            source="seed",
        ))
    return samples
