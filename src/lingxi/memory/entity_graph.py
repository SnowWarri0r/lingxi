"""Lightweight entity graph for memory: tracks which facts mention which entities.

Stored as a sidecar JSON file. Entities are extracted by LLM on memory insert,
and the graph allows looking up "all facts about X" quickly.
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from lingxi.providers.base import LLMProvider


ENTITY_EXTRACTION_PROMPT = """从下面这条记忆中提取关键实体（人名、地名、事物、特定话题等）。

记忆：{content}

只提取真正具体的命名实体，不要虚词和泛指。
回复 JSON 数组，每个实体一个对象：
[{{"name": "实体名", "type": "person|place|thing|topic"}}]

如果没有实体，回复 []
"""


class Entity(BaseModel):
    name: str
    type: str = "thing"  # person, place, thing, topic
    fact_ids: set[str] = Field(default_factory=set)
    first_seen: datetime = Field(default_factory=datetime.now)
    last_seen: datetime = Field(default_factory=datetime.now)
    mention_count: int = 0

    def model_dump_for_save(self) -> dict:
        d = self.model_dump(mode="json")
        d["fact_ids"] = sorted(self.fact_ids)
        return d


class EntityGraph:
    """Sidecar entity index: entity_name -> Entity (with linked fact_ids)."""

    def __init__(self, data_dir: Path | str):
        self._path = Path(data_dir) / "entities.json"
        # Key: lowercase entity name (for case-insensitive lookup)
        self._entities: dict[str, Entity] = {}
        self._lock = asyncio.Lock()
        self._loaded = False

    async def load(self) -> None:
        if self._loaded:
            return
        if self._path.exists():
            try:
                with open(self._path, encoding="utf-8") as f:
                    data = json.load(f)
                for key, e in data.items():
                    e["fact_ids"] = set(e.get("fact_ids", []))
                    self._entities[key] = Entity.model_validate(e)
            except (json.JSONDecodeError, OSError):
                pass
        self._loaded = True

    async def save(self) -> None:
        async with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = {k: v.model_dump_for_save() for k, v in self._entities.items()}
            tmp = self._path.with_suffix(".tmp")

            def _write():
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2, default=str)
                tmp.rename(self._path)

            await asyncio.to_thread(_write)

    def _key(self, name: str) -> str:
        return name.strip().lower()

    def link(self, name: str, entity_type: str, fact_id: str) -> None:
        """Add a fact to an entity, creating the entity if new."""
        key = self._key(name)
        if not key:
            return
        now = datetime.now()
        if key in self._entities:
            ent = self._entities[key]
            ent.fact_ids.add(fact_id)
            ent.last_seen = now
            ent.mention_count += 1
        else:
            self._entities[key] = Entity(
                name=name,
                type=entity_type,
                fact_ids={fact_id},
                mention_count=1,
            )

    def get(self, name: str) -> Entity | None:
        return self._entities.get(self._key(name))

    def find_in_text(self, text: str) -> list[Entity]:
        """Find any known entity whose name appears in the text."""
        text_lower = text.lower()
        found = []
        for key, ent in self._entities.items():
            if key in text_lower:
                found.append(ent)
        return found

    def all_entities(self) -> list[Entity]:
        return list(self._entities.values())

    def stats(self) -> dict:
        return {
            "entity_count": len(self._entities),
            "total_links": sum(len(e.fact_ids) for e in self._entities.values()),
        }


class EntityExtractor:
    """Uses LLM to extract entities from memory content."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    async def extract(self, content: str) -> list[dict]:
        """Returns list of {name, type} dicts."""
        if not content or len(content) < 5:
            return []

        try:
            result = await self.llm.complete(
                messages=[{"role": "user", "content": ENTITY_EXTRACTION_PROMPT.format(content=content)}],
                max_tokens=300,
                temperature=0.2,
            )
        except Exception:
            return []

        text = result.content.strip()
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            return []

        try:
            data = json.loads(match.group())
            if not isinstance(data, list):
                return []
            entities = []
            for item in data:
                if isinstance(item, dict) and item.get("name"):
                    entities.append({
                        "name": str(item["name"])[:50],
                        "type": str(item.get("type", "thing"))[:20],
                    })
            return entities
        except (json.JSONDecodeError, ValueError):
            return []
