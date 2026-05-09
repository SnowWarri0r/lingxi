"""LLM-based extractor for new relational memory entries.

Called periodically by the reflection loop with recent dialogue + the
existing memory (for deduplication context). Returns structured deltas
that the caller merges into the per-recipient store.

Conservative by design:
- Most categories will be empty in most extractions — that's correct.
  The point is to catch the genuine moments, not fabricate texture.
- The prompt explicitly tells the model to return empty lists for
  categories it didn't observe, instead of inventing.
- Items already present in the existing memory are listed in the prompt
  so the model knows not to re-extract them.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from lingxi.providers.base import LLMProvider
from lingxi.relational.models import (
    DailyPattern,
    FightPattern,
    InsideJoke,
    RelationalMemory,
    SharedPlace,
    SweetMoment,
)


_EXTRACTION_PROMPT = """你在帮 Aria 整理她和这位用户的关系记忆。

【已记下的关系细节】（不要重复抽取）
{existing}

【最近的对话】
{dialogue}

请从对话里抽取**真的发生过、值得记进关系记忆的**新细节，按这些类目：
- inside_jokes（只有你们俩懂的梗/暗号/特殊用法的词。要原句不要意译）
- shared_places（你们之间有意义的具体地点。不是泛指"咖啡馆"，是"楼下那家"）
- fight_patterns（吵架/冷战的典型模式：触发→她反应→怎么修复）
- sweet_moments（具体某一刻、有细节、值得心里留下的瞬间。不是"聊得很开心"）
- pet_names（彼此用的特殊称呼，原话）
- daily_patterns（他生活的规律：上下班时间、固定习惯。要从对话里他自己说出来或暗示的）
- relationship_summary_update（仅当关系基调有明显变化时给一段新叙述）

**关键规则**：
1. 大多数对话里**大多类目应该是空**——这是对的。
2. 不要基于一两句话就认定某个 pattern——要至少两次出现才能确认。
3. inside_jokes 要原句，不能是描述（❌ "他们有个关于X的梗" ✅ "蜘蛛会做梦的"）
4. 已经在【已记下】里的不要再列。
5. 没有就返回空 list，**不要凑数**。

输出严格 JSON（不要 markdown 包裹，不要解释）：
```
{{
  "inside_jokes": [{{"phrase": "原句", "origin": "为什么这个梗"}}, ...],
  "shared_places": [{{"name": "...", "significance": "..."}}, ...],
  "fight_patterns": [{{"trigger": "...", "her_pattern": "...", "typical_repair": "..."}}, ...],
  "sweet_moments": [{{"content": "具体一句话描述这个瞬间", "weight": "high|medium|low"}}, ...],
  "pet_names": ["..."],
  "daily_patterns": [{{"pattern": "...", "confidence": "high|medium|low"}}, ...],
  "relationship_summary_update": null
}}
```
"""


def _render_existing(mem: RelationalMemory) -> str:
    """Render the existing relational memory compactly for deduplication."""
    if mem.is_empty():
        return "（暂无）"
    lines: list[str] = []
    if mem.inside_jokes:
        lines.append("inside_jokes: " + " / ".join(j.phrase for j in mem.inside_jokes))
    if mem.shared_places:
        lines.append("shared_places: " + " / ".join(p.name for p in mem.shared_places))
    if mem.fight_patterns:
        lines.append(
            "fight_patterns: " + " / ".join(f.trigger for f in mem.fight_patterns)
        )
    if mem.sweet_moments:
        lines.append(
            "sweet_moments: " + " / ".join(m.content[:30] for m in mem.sweet_moments)
        )
    if mem.pet_names:
        lines.append("pet_names: " + " / ".join(mem.pet_names))
    if mem.daily_patterns:
        lines.append(
            "daily_patterns: " + " / ".join(d.pattern for d in mem.daily_patterns)
        )
    if mem.relationship_summary:
        lines.append(f"summary: {mem.relationship_summary}")
    return "\n".join(lines)


def _render_dialogue(turns: list[Any]) -> str:
    """Render a list of memory turns as 'role: content' lines."""
    if not turns:
        return "（无）"
    lines: list[str] = []
    for t in turns:
        role = getattr(t, "role", "?")
        content = (getattr(t, "content", "") or "").strip()
        if not content:
            continue
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _strip_json_fences(text: str) -> str:
    """Some models wrap JSON in ```json ... ```; tolerate it."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _coerce_to_list(val: Any) -> list:
    if isinstance(val, list):
        return val
    return []


async def extract_relational_deltas(
    llm: LLMProvider,
    existing: RelationalMemory,
    turns: list[Any],
    *,
    model: str | None = None,
) -> dict[str, Any]:
    """Call the LLM and parse a deltas dict.

    Returns an empty-buckets dict on parse failure (so callers can merge
    safely with no effect). Logs the failure reason.
    """
    prompt = _EXTRACTION_PROMPT.format(
        existing=_render_existing(existing),
        dialogue=_render_dialogue(turns),
    )

    try:
        kwargs: dict = {}
        if model:
            kwargs["model"] = model
        response = await llm.complete(
            messages=[{"role": "user", "content": prompt}],
            system="你在帮 Aria 维护她对一段关系的内部记忆。诚实、保守、不凑数。",
            max_tokens=1500,
            temperature=0.5,
            **kwargs,
        )
        text = _strip_json_fences(response.content if hasattr(response, "content") else str(response))
        data = json.loads(text)
    except Exception as e:
        print(f"[relational.extract] LLM/parse failed: {e}", flush=True)
        return _empty_deltas()

    if not isinstance(data, dict):
        return _empty_deltas()

    return {
        "inside_jokes": _coerce_to_list(data.get("inside_jokes")),
        "shared_places": _coerce_to_list(data.get("shared_places")),
        "fight_patterns": _coerce_to_list(data.get("fight_patterns")),
        "sweet_moments": _coerce_to_list(data.get("sweet_moments")),
        "pet_names": _coerce_to_list(data.get("pet_names")),
        "daily_patterns": _coerce_to_list(data.get("daily_patterns")),
        "relationship_summary_update": data.get("relationship_summary_update"),
    }


def _empty_deltas() -> dict[str, Any]:
    return {
        "inside_jokes": [],
        "shared_places": [],
        "fight_patterns": [],
        "sweet_moments": [],
        "pet_names": [],
        "daily_patterns": [],
        "relationship_summary_update": None,
    }


def merge_deltas_into_memory(
    memory: RelationalMemory,
    deltas: dict[str, Any],
    *,
    now: datetime | None = None,
) -> int:
    """Merge LLM-extracted deltas into the in-place memory object.

    De-duplicates against existing entries by primary key (phrase / name
    / trigger / content / pattern). Returns count of new entries added.
    Caller is responsible for persistence (typically via store.update_memory).
    """
    now = now or datetime.now()
    added = 0

    existing_jokes = {j.phrase for j in memory.inside_jokes}
    for raw in deltas.get("inside_jokes", []):
        if not isinstance(raw, dict):
            continue
        phrase = (raw.get("phrase") or "").strip()
        if not phrase or phrase in existing_jokes:
            continue
        memory.inside_jokes.append(
            InsideJoke(
                phrase=phrase,
                origin=(raw.get("origin") or "").strip(),
                last_used_at=now,
            )
        )
        existing_jokes.add(phrase)
        added += 1

    existing_places = {p.name for p in memory.shared_places}
    for raw in deltas.get("shared_places", []):
        if not isinstance(raw, dict):
            continue
        name = (raw.get("name") or "").strip()
        if not name or name in existing_places:
            continue
        memory.shared_places.append(
            SharedPlace(
                name=name,
                significance=(raw.get("significance") or "").strip(),
                last_referenced_at=now,
            )
        )
        existing_places.add(name)
        added += 1

    existing_triggers = {f.trigger for f in memory.fight_patterns}
    for raw in deltas.get("fight_patterns", []):
        if not isinstance(raw, dict):
            continue
        trigger = (raw.get("trigger") or "").strip()
        if not trigger or trigger in existing_triggers:
            continue
        memory.fight_patterns.append(
            FightPattern(
                trigger=trigger,
                her_pattern=(raw.get("her_pattern") or "").strip(),
                typical_repair=(raw.get("typical_repair") or "").strip(),
                last_occurred_at=now,
            )
        )
        existing_triggers.add(trigger)
        added += 1

    existing_moments = {m.content for m in memory.sweet_moments}
    for raw in deltas.get("sweet_moments", []):
        if not isinstance(raw, dict):
            continue
        content = (raw.get("content") or "").strip()
        if not content or content in existing_moments:
            continue
        weight = raw.get("weight", "medium")
        if weight not in ("high", "medium", "low"):
            weight = "medium"
        memory.sweet_moments.append(
            SweetMoment(timestamp=now, content=content, weight=weight)
        )
        existing_moments.add(content)
        added += 1

    existing_names = set(memory.pet_names)
    for raw in deltas.get("pet_names", []):
        if not isinstance(raw, str):
            continue
        name = raw.strip()
        if not name or name in existing_names:
            continue
        memory.pet_names.append(name)
        existing_names.add(name)
        added += 1

    existing_patterns = {p.pattern for p in memory.daily_patterns}
    for raw in deltas.get("daily_patterns", []):
        if not isinstance(raw, dict):
            continue
        pattern = (raw.get("pattern") or "").strip()
        if not pattern or pattern in existing_patterns:
            continue
        confidence = raw.get("confidence", "medium")
        if confidence not in ("high", "medium", "low"):
            confidence = "medium"
        memory.daily_patterns.append(
            DailyPattern(
                pattern=pattern,
                confidence=confidence,
                last_confirmed_at=now,
            )
        )
        existing_patterns.add(pattern)
        added += 1

    summary_update = deltas.get("relationship_summary_update")
    if isinstance(summary_update, str) and summary_update.strip():
        memory.relationship_summary = summary_update.strip()

    memory.last_extracted_at = now
    return added
