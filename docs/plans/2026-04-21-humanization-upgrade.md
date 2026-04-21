# Humanization Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Aria's AI-toned single-call generation with an industry-standard single-call combo (prior-turn few-shot + Author's Note + prefill + sampler tuning), and build a RLHF-lite annotation pool so corrections flow back into retrievable few-shot samples.

**Architecture:** Phase 0 ships the single-call combo with hardcoded seeds for immediate effect. Phase 1 persists every turn into an `AnnotationStore` and loads 10 reviewed seeds from YAML into a `FewShotStore` backed by ChromaDB. Phase 2 wires annotation UIs (Feishu buttons, CLI commands, Web API). Phase 3 swaps hardcoded seeds with dynamic retrieval over the pool. Phase 4-5 (two-call pipeline) deferred to a separate plan, triggered only if Phase 3 evaluation shows insufficient improvement.

**Tech Stack:** Python 3.11+, Pydantic v2, ChromaDB (existing), Anthropic API (existing), lark-oapi (existing Feishu SDK), pytest + pytest-asyncio.

**Spec:** `docs/specs/2026-04-21-two-call-compression-design.md`

---

## File Structure

### New files

```
src/lingxi/
├── fewshot/
│   ├── __init__.py                # Public re-exports
│   ├── models.py                  # FewShotSample, AnnotationTurn
│   ├── store.py                   # AnnotationStore + FewShotStore
│   ├── retriever.py               # FewShotRetriever
│   ├── collector.py               # AnnotationCollector
│   ├── summarizer.py              # LLM-assisted context_summary/tags
│   └── seeds_loader.py            # Parse seeds.yaml into FewShotSamples
└── conversation/
    └── prompt_assembly.py         # render_fewshots_as_messages, build_style_preamble, pick_prefill

config/fewshot/seeds.yaml          # 10 reviewed seed samples (Phase 1)

tests/test_fewshot/
├── __init__.py
├── test_models.py
├── test_store.py
├── test_retriever.py
├── test_collector.py
└── test_seeds_loader.py

tests/test_conversation/
└── test_prompt_assembly.py
```

### Modified files

- `src/lingxi/persona/models.py` — add `StyleConfig`, `SamplingConfig`; attach to `PersonaConfig`
- `src/lingxi/providers/base.py` — extend `LLMProvider.complete` signature with `top_p` and `prefill`
- `src/lingxi/providers/claude.py` — thread `top_p` into body; append prefill as trailing assistant message
- `src/lingxi/conversation/engine.py` — inject style preamble / prefill / few-shot / sampler; record AnnotationTurn; expose `turn_id` on output
- `src/lingxi/conversation/output_schema.py` — add `turn_id` field to `TurnOutput`
- `src/lingxi/channels/feishu.py` — annotation buttons on final card; `/reveal` command; form callback for correction
- `src/lingxi/app.py` — CLI `:good` / `:bad <correction>` / `:reveal` commands
- `src/lingxi/web/server.py` — POST `/turns/{id}/annotate`; GET `/turns/{id}/inner_thought`
- `pyproject.toml` — no new required deps (ChromaDB + PyYAML already optional/required)

---

## Phase 0: Single-Call Combo (immediate AI-tone reduction)

### Task 1: Extend PersonaConfig with StyleConfig and SamplingConfig

**Files:**
- Modify: `src/lingxi/persona/models.py`
- Test: `tests/test_persona/test_models.py`

- [ ] **Step 1: Write the failing test**

Append to `/Users/lovart/agent/tests/test_persona/test_models.py`:

```python
from lingxi.persona.models import PersonaConfig, Identity, StyleConfig, SamplingConfig


class TestStyleConfig:
    def test_defaults(self):
        cfg = StyleConfig()
        assert cfg.speech_max_chars == 40
        assert cfg.prefill_openers == ["嗯", "欸", "哦", ""]
        assert cfg.blacklist_phrases == []

    def test_custom_values(self):
        cfg = StyleConfig(
            speech_max_chars=80,
            prefill_openers=["哈"],
            blacklist_phrases=["据说"],
        )
        assert cfg.speech_max_chars == 80
        assert cfg.prefill_openers == ["哈"]


class TestSamplingConfig:
    def test_defaults(self):
        cfg = SamplingConfig()
        assert cfg.temperature == 1.0
        assert cfg.top_p == 0.95

    def test_bounds(self):
        # temperature should be clamped to non-negative
        import pytest
        with pytest.raises(Exception):
            SamplingConfig(temperature=-0.1)


class TestPersonaConfigNewFields:
    def test_style_and_sampling_attached(self):
        persona = PersonaConfig(
            name="T",
            identity=Identity(full_name="T"),
        )
        assert persona.style.speech_max_chars == 40
        assert persona.sampling.temperature == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/lovart/agent && python -m pytest tests/test_persona/test_models.py -v 2>&1 | tail -20
```

Expected: ImportError for `StyleConfig`, `SamplingConfig`.

- [ ] **Step 3: Add the models and wire into PersonaConfig**

Edit `/Users/lovart/agent/src/lingxi/persona/models.py`. Add these classes after `SpeakingStyle`:

```python
class StyleConfig(BaseModel):
    """Output style controls for the single-call combo."""

    speech_max_chars: int = Field(default=40, ge=1, le=500)
    prefill_openers: list[str] = Field(
        default_factory=lambda: ["嗯", "欸", "哦", ""],
        description="Random-pick prefill for assistant message; empty string = no prefill.",
    )
    blacklist_phrases: list[str] = Field(
        default_factory=list,
        description="Extra phrases to ban, on top of the code-default list.",
    )


class SamplingConfig(BaseModel):
    """LLM sampling parameters."""

    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
```

Then update `PersonaConfig` at the bottom of the file:

```python
class PersonaConfig(BaseModel):
    """Root configuration for a virtual persona."""

    name: str
    version: str = "1.0"
    identity: Identity
    personality: PersonalityProfile = Field(default_factory=PersonalityProfile)
    speaking_style: SpeakingStyle = Field(default_factory=SpeakingStyle)
    emotional_baseline: EmotionalBaseline = Field(default_factory=EmotionalBaseline)
    goals: list[GoalDefinition] = Field(default_factory=list)
    relationship: Relationship = Field(default_factory=Relationship)
    style: StyleConfig = Field(default_factory=StyleConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /Users/lovart/agent && python -m pytest tests/test_persona/test_models.py -v 2>&1 | tail -20
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/lovart/agent && git add src/lingxi/persona/models.py tests/test_persona/test_models.py && git commit -m "$(cat <<'EOF'
Add StyleConfig and SamplingConfig to PersonaConfig

Phase 0 groundwork: give persona YAML a place to declare speech
length cap, prefill openers, blacklist extras, and sampler params.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Extend LLMProvider interface with top_p and prefill

**Files:**
- Modify: `src/lingxi/providers/base.py`
- Modify: `src/lingxi/providers/claude.py`
- Modify: `src/lingxi/providers/openai_provider.py` (keep signature compatible)
- Test: `tests/test_providers/test_claude_prefill.py` (new)

- [ ] **Step 1: Write the failing test**

Create `/Users/lovart/agent/tests/test_providers/test_claude_prefill.py`:

```python
"""Unit tests for ClaudeProvider prefill and top_p support (body shape only)."""

from lingxi.providers.claude import ClaudeProvider


def test_body_contains_top_p():
    provider = ClaudeProvider(api_key="sk-ant-api-test", model="claude-sonnet-4-20250514")
    body = provider._build_body(
        messages=[{"role": "user", "content": "hi"}],
        system=None,
        max_tokens=100,
        temperature=0.9,
        top_p=0.7,
    )
    assert body["top_p"] == 0.7
    assert body["temperature"] == 0.9


def test_prefill_appends_assistant_message():
    provider = ClaudeProvider(api_key="sk-ant-api-test", model="claude-sonnet-4-20250514")
    messages = [{"role": "user", "content": "hi"}]
    body = provider._build_body(
        messages=messages,
        system=None,
        max_tokens=100,
        temperature=0.9,
        top_p=1.0,
        prefill="嗯",
    )
    assert body["messages"][-1] == {"role": "assistant", "content": "嗯"}


def test_empty_prefill_does_not_append():
    provider = ClaudeProvider(api_key="sk-ant-api-test", model="claude-sonnet-4-20250514")
    messages = [{"role": "user", "content": "hi"}]
    body = provider._build_body(
        messages=messages,
        system=None,
        max_tokens=100,
        temperature=0.9,
        top_p=1.0,
        prefill="",
    )
    assert body["messages"][-1]["role"] == "user"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/lovart/agent && python -m pytest tests/test_providers/test_claude_prefill.py -v 2>&1 | tail -15
```

Expected: `_build_body` doesn't accept `top_p` / `prefill` kwargs → TypeError.

- [ ] **Step 3: Update base interface**

Edit `/Users/lovart/agent/src/lingxi/providers/base.py`. Replace the `complete` and `complete_stream` signatures to accept `top_p` and `prefill`:

```python
    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 1.0,
        prefill: str = "",
        **kwargs,
    ) -> CompletionResult:
        """Generate a completion from messages."""

    async def complete_stream(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 1.0,
        prefill: str = "",
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a completion. Default implementation wraps complete()."""
        result = await self.complete(
            messages, system, max_tokens, temperature, top_p, prefill, **kwargs,
        )
        yield StreamChunk(content=result.content, is_final=True)
```

- [ ] **Step 4: Update ClaudeProvider**

Edit `/Users/lovart/agent/src/lingxi/providers/claude.py`. Replace `_build_body` signature and body construction:

```python
    def _build_body(
        self,
        messages: list[dict],
        system: str | None,
        max_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        prefill: str = "",
    ) -> dict:
        # If prefill is set, append as trailing assistant message (Anthropic pattern)
        outgoing_messages = list(messages)
        if prefill:
            outgoing_messages.append({"role": "assistant", "content": prefill})

        body: dict = {
            "model": self.model,
            "messages": outgoing_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
```

(Keep the rest of `_build_body` — the system block construction — as-is.)

Then update `complete` and `complete_stream` to accept and forward the new kwargs. Find `complete`:

```python
    async def complete(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 1.0,
        prefill: str = "",
        **kwargs,
    ) -> CompletionResult:
        body = self._build_body(messages, system, max_tokens, temperature, top_p, prefill)
        url = f"{API_BASE}?beta=true" if self._is_oauth else API_BASE

        response = await self._request_with_auto_refresh(url, body)

        data = response.json()

        content = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                content += block.get("text", "")

        # Prepend prefill to content so caller sees full text
        if prefill:
            content = prefill + content

        return CompletionResult(
            content=content,
            model=data.get("model", self.model),
            usage={
                "input_tokens": data.get("usage", {}).get("input_tokens", 0),
                "output_tokens": data.get("usage", {}).get("output_tokens", 0),
            },
            finish_reason=data.get("stop_reason", ""),
        )
```

Find `complete_stream` and update similarly:

```python
    async def complete_stream(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 1.0,
        prefill: str = "",
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        body = self._build_body(messages, system, max_tokens, temperature, top_p, prefill)
        body["stream"] = True
        url = f"{API_BASE}?beta=true" if self._is_oauth else API_BASE

        # Emit prefill as the first chunk so downstream sees complete text
        if prefill:
            yield StreamChunk(content=prefill)

        # ... (rest of existing streaming loop unchanged)
```

For the existing streaming loop body, find the current `async def complete_stream` and preserve its SSE parsing logic — only the signature + initial prefill yield + `_build_body` call changes.

- [ ] **Step 5: Update OpenAIProvider signature (stub compatible)**

Edit `/Users/lovart/agent/src/lingxi/providers/openai_provider.py`. Find `async def complete` and update to accept the new kwargs (keep body of function functionally unchanged — OpenAI's Python client ignores unknown kwargs via **kwargs):

```python
    async def complete(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 1.0,
        prefill: str = "",
        **kwargs,
    ) -> CompletionResult:
```

And thread `top_p` into the OpenAI client call if easy; else leave a comment `# top_p forwarded if supported`. If the OpenAI client call currently ignores the param, this is acceptable — Claude is the default path.

- [ ] **Step 6: Run test to verify it passes**

```bash
cd /Users/lovart/agent && python -m pytest tests/test_providers/test_claude_prefill.py -v 2>&1 | tail -15
```

Expected: 3 tests pass.

- [ ] **Step 7: Commit**

```bash
cd /Users/lovart/agent && git add src/lingxi/providers/ tests/test_providers/test_claude_prefill.py && git commit -m "$(cat <<'EOF'
Add top_p and prefill support to LLMProvider

ClaudeProvider now threads top_p into the request body and implements
prefill by appending a trailing assistant message (Anthropic convention),
prepending the prefill text to the returned content.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Create prompt_assembly module with three helpers

**Files:**
- Create: `src/lingxi/conversation/prompt_assembly.py`
- Create: `tests/test_conversation/__init__.py` (if missing)
- Create: `tests/test_conversation/test_prompt_assembly.py`

- [ ] **Step 1: Write the failing test**

Create `/Users/lovart/agent/tests/test_conversation/__init__.py` if it doesn't already exist (empty file).

Create `/Users/lovart/agent/tests/test_conversation/test_prompt_assembly.py`:

```python
"""Tests for prompt_assembly helpers used by the single-call combo."""

import random

import pytest

from lingxi.conversation.prompt_assembly import (
    build_style_preamble,
    pick_prefill,
    render_fewshots_as_messages,
)
from lingxi.fewshot.models import FewShotSample
from lingxi.persona.models import StyleConfig


def _sample(context: str, speech: str) -> FewShotSample:
    return FewShotSample(
        id="x",
        inner_thought="...",
        original_speech=None,
        corrected_speech=speech,
        context_summary=context,
        tags=[],
        recipient_key=None,
        source="seed",
    )


class TestRenderFewshots:
    def test_empty_returns_empty_list(self):
        assert render_fewshots_as_messages([]) == []

    def test_one_pair(self):
        out = render_fewshots_as_messages([_sample("深夜加班", "累死了")])
        assert out == [
            {"role": "user", "content": "深夜加班"},
            {"role": "assistant", "content": "累死了"},
        ]

    def test_multiple_pairs_preserve_order(self):
        out = render_fewshots_as_messages([
            _sample("A", "a"),
            _sample("B", "b"),
        ])
        assert [m["content"] for m in out] == ["A", "a", "B", "b"]


class TestBuildStylePreamble:
    def test_includes_max_chars(self):
        cfg = StyleConfig(speech_max_chars=25)
        pre = build_style_preamble(cfg)
        assert "≤25" in pre or "25" in pre

    def test_default_blacklist_included(self):
        pre = build_style_preamble(StyleConfig())
        assert "希望" in pre
        assert "总是让人" in pre

    def test_persona_blacklist_appended(self):
        cfg = StyleConfig(blacklist_phrases=["据说"])
        pre = build_style_preamble(cfg)
        assert "据说" in pre

    def test_wraps_user_message(self):
        pre = build_style_preamble(StyleConfig())
        wrapped = pre + "\n\n你今天吃啥"
        assert "你今天吃啥" in wrapped


class TestPickPrefill:
    def test_returns_value_from_openers(self):
        cfg = StyleConfig(prefill_openers=["嗯", "欸"])
        rng = random.Random(42)
        pick = pick_prefill(cfg, rng=rng)
        assert pick in ("嗯", "欸")

    def test_empty_list_returns_empty(self):
        cfg = StyleConfig(prefill_openers=[])
        assert pick_prefill(cfg) == ""

    def test_empty_string_option_allowed(self):
        cfg = StyleConfig(prefill_openers=[""])
        assert pick_prefill(cfg) == ""
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/lovart/agent && python -m pytest tests/test_conversation/test_prompt_assembly.py -v 2>&1 | tail -10
```

Expected: ModuleNotFoundError for `lingxi.conversation.prompt_assembly`.

- [ ] **Step 3: Create the module**

Create `/Users/lovart/agent/src/lingxi/conversation/prompt_assembly.py`:

```python
"""Helpers for single-call combo prompt construction.

- render_fewshots_as_messages: turn FewShotSamples into user/assistant message pairs
- build_style_preamble: an Author's Note prefix that sits just before the user message
- pick_prefill: choose an assistant prefill opener from the persona's configured set
"""

from __future__ import annotations

import random

from lingxi.fewshot.models import FewShotSample
from lingxi.persona.models import StyleConfig

# Base blacklist applied to every turn. Persona can append more.
DEFAULT_BLACKLIST: tuple[str, ...] = (
    "希望",
    "如果有任何",
    "总的来说",
    "需要注意的是",
    "世界真的很小",
    "总是让人",
    "这对你",
    "很高兴为你",
    "希望对你有帮助",
    "如有任何",
)


def render_fewshots_as_messages(samples: list[FewShotSample]) -> list[dict]:
    """Render each sample as a user/assistant message pair.

    The user side carries the context summary; the assistant side carries the
    target speech. LLMs model this structure as "if user says X, assistant
    says Y" much more strongly than any system-prompt description.
    """
    messages: list[dict] = []
    for s in samples:
        messages.append({"role": "user", "content": s.context_summary})
        messages.append({"role": "assistant", "content": s.corrected_speech})
    return messages


def build_style_preamble(style: StyleConfig) -> str:
    """Author's Note style block to prepend to the user's final message.

    Returns a multi-line string ending with a trailing newline so the caller
    can concatenate with the real message.
    """
    phrases = list(DEFAULT_BLACKLIST) + list(style.blacklist_phrases)
    joined = "、".join(phrases)
    return (
        f"[style: 微信聊天。≤{style.speech_max_chars}字。\n"
        f"禁用词：{joined}\n"
        f"禁止总结、禁止给建议框架（1/2/3 点）\n"
        f"允许：省略、倒装、感叹词（嗯/欸/哦）、破折号、半句话]\n\n"
    )


def pick_prefill(style: StyleConfig, rng: random.Random | None = None) -> str:
    """Pick an assistant prefill opener from the persona's configured list.

    Empty string in the list (or an empty list) means "no prefill this turn".
    """
    if not style.prefill_openers:
        return ""
    r = rng or random
    return r.choice(style.prefill_openers)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /Users/lovart/agent && python -m pytest tests/test_conversation/test_prompt_assembly.py -v 2>&1 | tail -15
```

Expected: all 9 tests pass. If `FewShotSample` import fails, skip this task until Task 5 (models) — but ordering here has models later; we need to bootstrap. Go to Task 4 first then return, OR temporarily inline a stub `FewShotSample`. **Recommended:** reorder — run Task 5 (models) first, then this task. If executing top-to-bottom, add a minimal `FewShotSample` stub right now in a fresh `src/lingxi/fewshot/__init__.py` + `models.py` to unblock; Task 5 will flesh them out.

**To unblock now**: create `/Users/lovart/agent/src/lingxi/fewshot/__init__.py` (empty) and `/Users/lovart/agent/src/lingxi/fewshot/models.py` with just:

```python
from __future__ import annotations
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Literal


class FewShotSample(BaseModel):
    id: str
    inner_thought: str
    original_speech: str | None = None
    corrected_speech: str
    context_summary: str
    tags: list[str] = Field(default_factory=list)
    recipient_key: str | None = None
    source: Literal["seed", "user_correction", "positive"] = "seed"
    created_at: datetime = Field(default_factory=datetime.now)
```

Task 5 will add `AnnotationTurn` and tests for both.

Rerun the test — expected: all pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/lovart/agent && git add src/lingxi/conversation/prompt_assembly.py src/lingxi/fewshot/ tests/test_conversation/ && git commit -m "$(cat <<'EOF'
Add prompt_assembly helpers for single-call combo

render_fewshots_as_messages turns FewShotSamples into user/assistant
pairs; build_style_preamble produces the Author's Note block inserted
before each user message; pick_prefill samples from the persona's
configured openers.

Minimal FewShotSample stub added to unblock; Task 5 fleshes it out.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Wire single-call combo into ConversationEngine

**Files:**
- Modify: `src/lingxi/conversation/engine.py`
- Modify: `src/lingxi/conversation/output_schema.py` (add `turn_id`)
- Test: `tests/test_conversation/test_engine_combo.py` (new)

This task threads style preamble into the last user message, picks a prefill, and passes sampler params to the LLM. It does NOT yet use few-shot retrieval — that's Task 13. For Phase 0, use a small hardcoded list of 3 seeds to prove the pipeline works.

- [ ] **Step 1: Add turn_id field to TurnOutput**

Edit `/Users/lovart/agent/src/lingxi/conversation/output_schema.py`. Add field:

```python
class TurnOutput(BaseModel):
    """All parallel outputs from a single conversation turn."""

    turn_id: str = ""

    # Spoken content (what text channels render)
    speech: str = ""
    # ... (rest unchanged)
```

- [ ] **Step 2: Write the failing test**

Create `/Users/lovart/agent/tests/test_conversation/test_engine_combo.py`:

```python
"""Tests for ConversationEngine single-call combo wiring.

Uses a FakeLLMProvider that records the arguments it received, so we
can assert prefill / sampler / style preamble are being passed through.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterator

import pytest

from lingxi.conversation.engine import ConversationEngine
from lingxi.memory.manager import MemoryManager
from lingxi.persona.models import (
    Identity,
    PersonaConfig,
    SamplingConfig,
    StyleConfig,
)
from lingxi.providers.base import CompletionResult, LLMProvider, StreamChunk


class FakeLLM(LLMProvider):
    def __init__(self):
        self.last_messages: list[dict] | None = None
        self.last_system: str | None = None
        self.last_kwargs: dict = {}

    async def complete(self, messages, system=None, max_tokens=4096,
                       temperature=0.7, top_p=1.0, prefill="", **kwargs):
        self.last_messages = list(messages)
        self.last_system = system
        self.last_kwargs = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "prefill": prefill,
        }
        # Echo a minimal response body
        return CompletionResult(content=f"{prefill}好的")

    async def complete_stream(self, messages, system=None, max_tokens=4096,
                              temperature=0.7, top_p=1.0, prefill="", **kwargs):
        result = await self.complete(
            messages, system, max_tokens, temperature, top_p, prefill, **kwargs
        )
        yield StreamChunk(content=result.content, is_final=True)

    async def embed(self, text: str):
        return [0.0] * 8


@pytest.fixture
def persona():
    return PersonaConfig(
        name="Test",
        identity=Identity(full_name="Test"),
        style=StyleConfig(
            speech_max_chars=30,
            prefill_openers=["嗯"],   # deterministic
            blacklist_phrases=["据说"],
        ),
        sampling=SamplingConfig(temperature=0.8, top_p=0.9),
    )


@pytest.fixture
def memory(tmp_path):
    return MemoryManager(data_dir=str(tmp_path), long_term_backend="json")


async def test_style_preamble_prepended_to_user_message(persona, memory):
    llm = FakeLLM()
    engine = ConversationEngine(persona=persona, llm_provider=llm, memory_manager=memory)
    await engine.chat("你今天吃啥", channel="cli", recipient_id="tester")

    # The last user message should start with the style preamble
    user_msgs = [m for m in llm.last_messages or [] if m["role"] == "user"]
    assert user_msgs, "no user message was sent"
    last_user = user_msgs[-1]
    content = last_user["content"]
    # Content may be a string or list of blocks
    if isinstance(content, list):
        # Find the text block
        text = next((b.get("text", "") for b in content if b.get("type") == "text"), "")
    else:
        text = content
    assert "[style:" in text
    assert "≤30" in text
    assert "据说" in text  # persona blacklist merged
    assert "你今天吃啥" in text


async def test_sampler_forwarded(persona, memory):
    llm = FakeLLM()
    engine = ConversationEngine(persona=persona, llm_provider=llm, memory_manager=memory)
    await engine.chat("hi", channel="cli", recipient_id="tester")
    assert llm.last_kwargs["temperature"] == 0.8
    assert llm.last_kwargs["top_p"] == 0.9


async def test_prefill_forwarded(persona, memory):
    llm = FakeLLM()
    engine = ConversationEngine(persona=persona, llm_provider=llm, memory_manager=memory)
    output = await engine.chat_full("hi", channel="cli", recipient_id="tester")
    assert llm.last_kwargs["prefill"] == "嗯"


async def test_turn_id_populated(persona, memory):
    llm = FakeLLM()
    engine = ConversationEngine(persona=persona, llm_provider=llm, memory_manager=memory)
    output = await engine.chat_full("hi", channel="cli", recipient_id="tester")
    assert output.turn_id and len(output.turn_id) > 0
```

- [ ] **Step 3: Run test to verify it fails**

```bash
cd /Users/lovart/agent && python -m pytest tests/test_conversation/test_engine_combo.py -v 2>&1 | tail -20
```

Expected: tests fail because engine doesn't yet inject preamble/prefill/sampler.

- [ ] **Step 4: Modify `_prepare_turn` to inject style preamble**

Edit `/Users/lovart/agent/src/lingxi/conversation/engine.py`. Add imports at the top:

```python
import random
import uuid
from lingxi.conversation.prompt_assembly import (
    build_style_preamble,
    pick_prefill,
    render_fewshots_as_messages,
)
from lingxi.fewshot.models import FewShotSample
```

Find `_prepare_turn` (near line 98). After `messages = self.context_assembler.assemble_messages(memory_context)` and after the image-injection block, add the style preamble injection. Replace the final `return system_prompt, messages` with:

```python
        # --- Single-call combo (Phase 0) ---
        # Prepend hardcoded seed few-shots before history.
        # Phase 3 swaps these out for retriever results.
        seed_fewshots = self._phase0_seed_fewshots()
        few_shot_msgs = render_fewshots_as_messages(seed_fewshots)

        # Attach style preamble to the last user message
        style_preamble = build_style_preamble(self.persona.style)
        self._apply_style_preamble(messages, style_preamble)

        # Final message list = few-shot pairs + history (which already includes the user turn)
        final_messages = few_shot_msgs + messages
        return system_prompt, final_messages
```

Add these helper methods on `ConversationEngine`:

```python
    def _phase0_seed_fewshots(self) -> list[FewShotSample]:
        """Hardcoded baseline seeds before Phase 3 retriever lands.

        Covers three common AI-tone failure modes: over-eager agreement,
        cliché punchlines, and help-desk sign-offs.
        """
        return [
            FewShotSample(
                id="p0-1",
                inner_thought="",
                corrected_speech="哦？啥机械？",
                context_summary="用户提到他朋友的朋友也在做工业机械",
                tags=["好奇", "追问"],
                source="seed",
            ),
            FewShotSample(
                id="p0-2",
                inner_thought="",
                corrected_speech="也是，折腾完还要复盘 累。",
                context_summary="用户说刚开完一个长会",
                tags=["共鸣", "吐槽"],
                source="seed",
            ),
            FewShotSample(
                id="p0-3",
                inner_thought="",
                corrected_speech="嗯 早点睡。",
                context_summary="用户说困了要睡了",
                tags=["日常", "短"],
                source="seed",
            ),
        ]

    def _apply_style_preamble(self, messages: list[dict], preamble: str) -> None:
        """Prepend the preamble to the last user message's text.

        Handles both string content and multimodal block lists.
        """
        if not messages:
            return
        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                msg["content"] = preamble + content
                return
            if isinstance(content, list):
                # Find last text block; prepend preamble to it
                for block in reversed(content):
                    if isinstance(block, dict) and block.get("type") == "text":
                        block["text"] = preamble + block.get("text", "")
                        return
                # No text block? Add one
                content.append({"type": "text", "text": preamble})
                return
```

- [ ] **Step 5: Modify LLM calls to pass sampler + prefill; assign turn_id**

In the same `engine.py`, find `chat_full` and update the LLM call:

```python
    async def chat_full(
        self,
        user_input: str,
        images: list[dict] | None = None,
        channel: str | None = None,
        recipient_id: str | None = None,
    ) -> TurnOutput:
        """Process a user message. Returns the complete TurnOutput."""
        system_prompt, messages = await self._prepare_turn(
            user_input, images, channel, recipient_id
        )

        prefill = pick_prefill(self.persona.style)

        result = await self.llm.complete(
            messages=messages,
            system=system_prompt,
            temperature=self.persona.sampling.temperature,
            top_p=self.persona.sampling.top_p,
            prefill=prefill,
        )

        output = self._process_response(result.content)
        output.turn_id = str(uuid.uuid4())
        self._last_response_text = output.speech
        self.memory.add_turn("assistant", output.speech)
        self._persist_state(channel, recipient_id)
        return output
```

Do the same for `chat_stream` and `chat_stream_events` — pass `temperature`, `top_p`, `prefill` to `self.llm.complete_stream(...)` and assign `output.turn_id` after `_process_response`.

- [ ] **Step 6: Run test to verify it passes**

```bash
cd /Users/lovart/agent && python -m pytest tests/test_conversation/test_engine_combo.py -v 2>&1 | tail -20
```

Expected: all 4 tests pass.

- [ ] **Step 7: Run full test suite to catch regressions**

```bash
cd /Users/lovart/agent && python -m pytest tests/ -v 2>&1 | tail -30
```

Expected: all previously passing tests still pass.

- [ ] **Step 8: Manual smoke test (optional but recommended)**

Start CLI and chat a few turns:

```bash
cd /Users/lovart/agent && .venv/bin/lingxi 2>&1 | head -30
```

Send a message that would typically trigger AI-tone ("今天好累啊"). Observe the response — it should be shorter, less formulaic, potentially start with "嗯"/"欸"/"哦".

- [ ] **Step 9: Commit**

```bash
cd /Users/lovart/agent && git add src/lingxi/conversation/ tests/test_conversation/test_engine_combo.py && git commit -m "$(cat <<'EOF'
Wire single-call combo into ConversationEngine

Phase 0 complete: every turn now injects
- prior-turn few-shot (hardcoded 3 seeds, Phase 3 will swap for retriever)
- Author's Note style preamble prepended to the last user message
- prefill sampled from persona.style.prefill_openers
- temperature and top_p from persona.sampling

TurnOutput carries a fresh turn_id so downstream (annotation UIs) can
reference it.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 1: Seeds + AnnotationTurn Base

### Task 5: Flesh out fewshot/models.py with AnnotationTurn and full tests

**Files:**
- Modify: `src/lingxi/fewshot/models.py` (stubbed in Task 3)
- Create: `tests/test_fewshot/__init__.py`
- Create: `tests/test_fewshot/test_models.py`

- [ ] **Step 1: Create empty test package**

```bash
touch /Users/lovart/agent/tests/test_fewshot/__init__.py
```

- [ ] **Step 2: Write the failing test**

Create `/Users/lovart/agent/tests/test_fewshot/test_models.py`:

```python
"""Tests for FewShotSample and AnnotationTurn."""

from datetime import datetime

import pytest

from lingxi.fewshot.models import AnnotationTurn, FewShotSample


class TestFewShotSample:
    def test_defaults(self):
        s = FewShotSample(
            id="1",
            inner_thought="tired",
            corrected_speech="累。",
            context_summary="late night",
        )
        assert s.source == "seed"
        assert s.original_speech is None
        assert s.tags == []
        assert s.recipient_key is None
        assert isinstance(s.created_at, datetime)

    def test_source_values(self):
        for src in ("seed", "user_correction", "positive"):
            FewShotSample(
                id=src, inner_thought="x", corrected_speech="y",
                context_summary="z", source=src,
            )

    def test_invalid_source_rejected(self):
        with pytest.raises(Exception):
            FewShotSample(
                id="x", inner_thought="a", corrected_speech="b",
                context_summary="c", source="bogus",  # type: ignore
            )


class TestAnnotationTurn:
    def test_defaults(self):
        t = AnnotationTurn(
            turn_id="t1",
            recipient_key="cli:me",
            user_message="hi",
            inner_thought="",
            speech="hello",
        )
        assert t.annotation == "none"
        assert t.correction is None

    def test_set_correction(self):
        t = AnnotationTurn(
            turn_id="t1", recipient_key="cli:me",
            user_message="hi", inner_thought="", speech="hello",
        )
        t.annotation = "negative"
        t.correction = "嗨"
        assert t.correction == "嗨"
```

- [ ] **Step 3: Run test to verify it fails**

```bash
cd /Users/lovart/agent && python -m pytest tests/test_fewshot/test_models.py -v 2>&1 | tail -10
```

Expected: `AnnotationTurn` not importable.

- [ ] **Step 4: Replace the stubbed models.py**

Overwrite `/Users/lovart/agent/src/lingxi/fewshot/models.py`:

```python
"""Pydantic models for the few-shot pool and annotation system."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

FewShotSource = Literal["seed", "user_correction", "positive"]
AnnotationKind = Literal["none", "positive", "negative"]


class FewShotSample(BaseModel):
    """One retrieval sample: an inner-thought / speech pair.

    - seed: hand-written baseline, recipient_key typically None (global)
    - user_correction: user-supplied fix after marking a turn 'not like me'
    - positive: confirmed-good turn (thumbs-up)
    """

    id: str
    inner_thought: str
    original_speech: str | None = None
    corrected_speech: str
    context_summary: str
    tags: list[str] = Field(default_factory=list)
    recipient_key: str | None = None
    source: FewShotSource = "seed"
    created_at: datetime = Field(default_factory=datetime.now)


class AnnotationTurn(BaseModel):
    """Per-turn state kept so the user can annotate after the fact."""

    turn_id: str
    recipient_key: str
    user_message: str
    inner_thought: str
    speech: str
    created_at: datetime = Field(default_factory=datetime.now)
    annotation: AnnotationKind = "none"
    correction: str | None = None
```

- [ ] **Step 5: Run test to verify it passes**

```bash
cd /Users/lovart/agent && python -m pytest tests/test_fewshot/test_models.py -v 2>&1 | tail -10
```

Expected: all 5 tests pass.

- [ ] **Step 6: Commit**

```bash
cd /Users/lovart/agent && git add src/lingxi/fewshot/models.py tests/test_fewshot/ && git commit -m "$(cat <<'EOF'
Flesh out fewshot models: FewShotSample + AnnotationTurn

FewShotSample has three sources (seed, user_correction, positive);
AnnotationTurn captures per-turn state for later annotation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: AnnotationStore — per-turn JSON persistence

**Files:**
- Create: `src/lingxi/fewshot/store.py` (AnnotationStore class)
- Create: `tests/test_fewshot/test_store.py`

- [ ] **Step 1: Write the failing test**

Create `/Users/lovart/agent/tests/test_fewshot/test_store.py`:

```python
"""Tests for AnnotationStore (FewShotStore tested in Task 7)."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from lingxi.fewshot.models import AnnotationTurn
from lingxi.fewshot.store import AnnotationStore


@pytest.fixture
def tmp_store(tmp_path: Path) -> AnnotationStore:
    return AnnotationStore(data_dir=tmp_path)


async def test_record_and_get(tmp_store):
    t = AnnotationTurn(
        turn_id="t1", recipient_key="cli:me",
        user_message="hi", inner_thought="greet", speech="嗨",
    )
    await tmp_store.record(t)
    loaded = await tmp_store.get_turn("t1")
    assert loaded is not None
    assert loaded.speech == "嗨"
    assert loaded.inner_thought == "greet"


async def test_get_missing_returns_none(tmp_store):
    assert await tmp_store.get_turn("nope") is None


async def test_update_annotation(tmp_store):
    t = AnnotationTurn(
        turn_id="t1", recipient_key="cli:me",
        user_message="hi", inner_thought="", speech="嗨",
    )
    await tmp_store.record(t)
    await tmp_store.update_annotation("t1", kind="negative", correction="哟")
    loaded = await tmp_store.get_turn("t1")
    assert loaded.annotation == "negative"
    assert loaded.correction == "哟"


async def test_update_missing_raises(tmp_store):
    with pytest.raises(KeyError):
        await tmp_store.update_annotation("ghost", kind="positive")


async def test_cleanup_unannotated_old_turns(tmp_store, tmp_path):
    # Create two turns, mark one as old-mtime
    t_new = AnnotationTurn(
        turn_id="new", recipient_key="cli:a",
        user_message="", inner_thought="", speech="",
    )
    t_old = AnnotationTurn(
        turn_id="old", recipient_key="cli:a",
        user_message="", inner_thought="", speech="",
    )
    await tmp_store.record(t_new)
    await tmp_store.record(t_old)

    old_path = tmp_path / "turns" / "old.json"
    import os
    old_time = (datetime.now() - timedelta(days=45)).timestamp()
    os.utime(old_path, (old_time, old_time))

    deleted = await tmp_store.cleanup(max_age_unannotated_days=30)
    assert deleted == 1
    assert await tmp_store.get_turn("new") is not None
    assert await tmp_store.get_turn("old") is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/lovart/agent && python -m pytest tests/test_fewshot/test_store.py -v 2>&1 | tail -10
```

Expected: ImportError for `AnnotationStore`.

- [ ] **Step 3: Create the module**

Create `/Users/lovart/agent/src/lingxi/fewshot/store.py`:

```python
"""Persistence layer for fewshot: AnnotationStore (per-turn JSON) and FewShotStore (Chroma + JSONL)."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from lingxi.fewshot.models import AnnotationKind, AnnotationTurn, FewShotSample


class AnnotationStore:
    """Persist AnnotationTurn records under data_dir/turns/<turn_id>.json.

    Cleanup policy:
      - Unannotated turns older than 30 days are removed.
      - Annotated turns older than 7 days past annotation can be removed
        (the upgraded FewShotSample supersedes them).
    """

    def __init__(self, data_dir: Path | str):
        self.data_dir = Path(data_dir)
        self.turns_dir = self.data_dir / "turns"
        self.turns_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, turn_id: str) -> Path:
        # Defend against path traversal
        safe = turn_id.replace("/", "_").replace("..", "_")
        return self.turns_dir / f"{safe}.json"

    async def record(self, turn: AnnotationTurn) -> None:
        def _write():
            path = self._path(turn.turn_id)
            path.write_text(turn.model_dump_json(indent=2), encoding="utf-8")
        await asyncio.to_thread(_write)

    async def get_turn(self, turn_id: str) -> AnnotationTurn | None:
        path = self._path(turn_id)
        if not path.exists():
            return None

        def _read() -> dict[str, Any]:
            return json.loads(path.read_text(encoding="utf-8"))

        data = await asyncio.to_thread(_read)
        return AnnotationTurn.model_validate(data)

    async def update_annotation(
        self,
        turn_id: str,
        kind: AnnotationKind,
        correction: str | None = None,
    ) -> AnnotationTurn:
        turn = await self.get_turn(turn_id)
        if turn is None:
            raise KeyError(turn_id)
        turn.annotation = kind
        if correction is not None:
            turn.correction = correction
        await self.record(turn)
        return turn

    async def cleanup(
        self,
        max_age_unannotated_days: int = 30,
        max_age_annotated_days: int = 7,
    ) -> int:
        """Remove stale turn files. Returns count deleted."""
        cutoff_unannot = (datetime.now() - timedelta(days=max_age_unannotated_days)).timestamp()
        cutoff_annot = (datetime.now() - timedelta(days=max_age_annotated_days)).timestamp()

        def _scan() -> int:
            count = 0
            for path in self.turns_dir.glob("*.json"):
                try:
                    mtime = path.stat().st_mtime
                    data = json.loads(path.read_text(encoding="utf-8"))
                    kind = data.get("annotation", "none")
                    if kind == "none" and mtime < cutoff_unannot:
                        path.unlink()
                        count += 1
                    elif kind != "none" and mtime < cutoff_annot:
                        path.unlink()
                        count += 1
                except Exception:
                    continue
            return count

        return await asyncio.to_thread(_scan)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /Users/lovart/agent && python -m pytest tests/test_fewshot/test_store.py -v 2>&1 | tail -15
```

Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/lovart/agent && git add src/lingxi/fewshot/store.py tests/test_fewshot/test_store.py && git commit -m "$(cat <<'EOF'
Add AnnotationStore for per-turn JSON persistence

Records every turn under data/fewshot/turns/<turn_id>.json so the
user can later annotate. Cleanup runs daily: unannotated >30d deleted,
annotated >7d deleted (promoted to FewShotSample superseded them).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: FewShotStore — ChromaDB collection + JSONL backup

**Files:**
- Modify: `src/lingxi/fewshot/store.py` (append FewShotStore class)
- Modify: `tests/test_fewshot/test_store.py` (add FewShotStore tests)

FewShotStore uses ChromaDB (already in the project) with a dimension-suffixed collection name (`fewshot_pool_d<N>`), mirroring the pattern in `memory/chroma_store.py`. Embedding is injected by the caller (consistent with MemoryManager).

- [ ] **Step 1: Write the failing test**

Append to `/Users/lovart/agent/tests/test_fewshot/test_store.py`:

```python
import numpy as np
import pytest

from lingxi.fewshot.store import FewShotStore


def _fake_embed(text: str, dim: int = 16) -> list[float]:
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    v = rng.normal(size=dim)
    v = v / np.linalg.norm(v)
    return v.tolist()


@pytest.fixture
async def fewshot_store(tmp_path):
    store = FewShotStore(data_dir=tmp_path, embedding_dim=16)
    await store.init()
    return store


async def test_add_and_query(fewshot_store):
    s = FewShotSample(
        id="s1", inner_thought="想喝咖啡", corrected_speech="去买一杯",
        context_summary="上午犯困", source="seed",
    )
    await fewshot_store.add(s, embedding=_fake_embed(s.inner_thought, 16))

    results = await fewshot_store.query(
        query_embedding=_fake_embed("想喝咖啡", 16),
        k=3,
    )
    assert len(results) == 1
    assert results[0].sample.id == "s1"


async def test_backup_jsonl_written(fewshot_store, tmp_path):
    s = FewShotSample(
        id="s2", inner_thought="x", corrected_speech="y",
        context_summary="z", source="seed",
    )
    await fewshot_store.add(s, embedding=_fake_embed(s.inner_thought, 16))
    backup = tmp_path / "fewshot" / "samples.jsonl"
    assert backup.exists()
    assert "s2" in backup.read_text(encoding="utf-8")


async def test_recipient_filter(fewshot_store):
    alice = FewShotSample(
        id="a", inner_thought="t", corrected_speech="a1",
        context_summary="c", recipient_key="cli:alice", source="positive",
    )
    bob = FewShotSample(
        id="b", inner_thought="t", corrected_speech="b1",
        context_summary="c", recipient_key="cli:bob", source="positive",
    )
    glob = FewShotSample(
        id="g", inner_thought="t", corrected_speech="g1",
        context_summary="c", recipient_key=None, source="seed",
    )
    for s in (alice, bob, glob):
        await fewshot_store.add(s, embedding=_fake_embed(s.id, 16))

    results = await fewshot_store.query(
        query_embedding=_fake_embed("t", 16),
        k=10,
        recipient_key="cli:alice",
    )
    ids = {r.sample.id for r in results}
    # alice-specific and global should be returned; bob filtered out
    assert "a" in ids
    assert "g" in ids
    assert "b" not in ids
```

Also adjust the top of the test file to import `FewShotSample`:

```python
from lingxi.fewshot.models import AnnotationTurn, FewShotSample
```

(already imported for AnnotationTurn tests — confirm it's listed).

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/lovart/agent && python -m pytest tests/test_fewshot/test_store.py::test_add_and_query -v 2>&1 | tail -10
```

Expected: ImportError for `FewShotStore`.

- [ ] **Step 3: Append FewShotStore to store.py**

Append to `/Users/lovart/agent/src/lingxi/fewshot/store.py`:

```python
from dataclasses import dataclass


@dataclass
class FewShotQueryResult:
    sample: FewShotSample
    similarity: float


class FewShotStore:
    """ChromaDB-backed pool of FewShotSamples, with a JSONL backup for disaster recovery.

    Collection name is dim-suffixed (fewshot_pool_d<N>) to avoid dim-mismatch
    errors when embeddings change.
    """

    def __init__(self, data_dir: Path | str, embedding_dim: int):
        self.data_dir = Path(data_dir)
        self.chroma_dir = self.data_dir / "chroma"
        self.backup_path = self.data_dir / "fewshot" / "samples.jsonl"
        self.embedding_dim = embedding_dim
        self.collection_name = f"fewshot_pool_d{embedding_dim}"
        self._client: Any = None
        self._collection: Any = None
        self._lock = asyncio.Lock()

    async def init(self) -> None:
        async with self._lock:
            if self._collection is not None:
                return
            await asyncio.to_thread(self._init_sync)

    def _init_sync(self) -> None:
        import chromadb
        from chromadb.config import Settings

        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.backup_path.parent.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self.chroma_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    async def add(self, sample: FewShotSample, embedding: list[float]) -> None:
        await self.init()
        if len(embedding) != self.embedding_dim:
            raise ValueError(
                f"embedding dim {len(embedding)} != store dim {self.embedding_dim}"
            )

        meta = {
            "source": sample.source,
            # ChromaDB metadata disallows None; use "" as sentinel
            "recipient_key": sample.recipient_key or "",
            "context_summary": sample.context_summary,
            "tags": ",".join(sample.tags),
            "original_speech": sample.original_speech or "",
            "corrected_speech": sample.corrected_speech,
            "inner_thought": sample.inner_thought,
            "created_at": sample.created_at.isoformat(),
        }

        def _add():
            self._collection.add(
                ids=[sample.id],
                documents=[sample.inner_thought or sample.context_summary],
                embeddings=[embedding],
                metadatas=[meta],
            )

        await asyncio.to_thread(_add)
        await self._append_backup(sample)

    async def _append_backup(self, sample: FewShotSample) -> None:
        def _append():
            with self.backup_path.open("a", encoding="utf-8") as fh:
                fh.write(sample.model_dump_json() + "\n")
        await asyncio.to_thread(_append)

    async def query(
        self,
        query_embedding: list[float],
        k: int = 6,
        recipient_key: str | None = None,
    ) -> list[FewShotQueryResult]:
        """Return the top-k samples by cosine similarity, filtered by recipient.

        If recipient_key is given, returns samples where metadata.recipient_key
        matches OR is empty (i.e. global seeds). Cross-user samples are excluded.
        """
        await self.init()

        where_clause: dict[str, Any] | None = None
        if recipient_key is not None:
            where_clause = {
                "$or": [
                    {"recipient_key": recipient_key},
                    {"recipient_key": ""},
                ]
            }

        def _query() -> dict[str, Any]:
            return self._collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_clause,
                include=["metadatas", "documents", "distances"],
            )

        result = await asyncio.to_thread(_query)
        ids_list = (result.get("ids") or [[]])[0]
        metas_list = (result.get("metadatas") or [[]])[0]
        dists_list = (result.get("distances") or [[]])[0]

        out: list[FewShotQueryResult] = []
        for sample_id, meta, dist in zip(ids_list, metas_list, dists_list):
            similarity = max(0.0, 1.0 - float(dist))
            sample = FewShotSample(
                id=sample_id,
                inner_thought=meta.get("inner_thought", ""),
                corrected_speech=meta.get("corrected_speech", ""),
                context_summary=meta.get("context_summary", ""),
                original_speech=meta.get("original_speech") or None,
                tags=[t for t in meta.get("tags", "").split(",") if t],
                recipient_key=(meta.get("recipient_key") or None),
                source=meta.get("source", "seed"),
                created_at=datetime.fromisoformat(
                    meta.get("created_at") or datetime.now().isoformat()
                ),
            )
            out.append(FewShotQueryResult(sample=sample, similarity=similarity))
        return out

    async def count(self) -> int:
        await self.init()
        def _count():
            return self._collection.count()
        return await asyncio.to_thread(_count)
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/lovart/agent && python -m pytest tests/test_fewshot/test_store.py -v 2>&1 | tail -15
```

Expected: all FewShotStore tests pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/lovart/agent && git add src/lingxi/fewshot/store.py tests/test_fewshot/test_store.py && git commit -m "$(cat <<'EOF'
Add FewShotStore: ChromaDB pool + JSONL backup

Dim-suffixed collection name (fewshot_pool_d<N>) mirrors the existing
memory pattern. Recipient filter uses $or with empty-string sentinel
for global seeds.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Author 10 seeds.yaml and seeds_loader

**Files:**
- Create: `config/fewshot/seeds.yaml`
- Create: `src/lingxi/fewshot/seeds_loader.py`
- Create: `tests/test_fewshot/test_seeds_loader.py`

The seeds are the cold-start of the pool. The 10 scenarios were agreed in the spec. These drafts will need user review before Phase 2 closes — mark the spec's seeds checklist as pending.

- [ ] **Step 1: Write the seed file**

Create `/Users/lovart/agent/config/fewshot/seeds.yaml`:

```yaml
# Cold-start few-shot seeds for the single-call combo.
# Each entry becomes a FewShotSample with source=seed, recipient_key=null (global).
# Scenarios selected in 2026-04-21-two-call-compression-design.md §8 Phase 1.

seeds:
  # 1. 共鸣 — "also had something like that"
  - id: seed-resonance-01
    context_summary: 用户说他今天见到了多年不联系的朋友，有点感慨
    inner_thought: 有点被戳到，我也有过类似的时刻；不想说大道理，想让他感到"被听见"
    corrected_speech: 嗯 久别重逢那种最上头。
    tags: [共鸣, 轻度感性]

  # 2. 安慰 — not saccharine
  - id: seed-comfort-02
    context_summary: 用户说今天工作被骂了，心情不好
    inner_thought: 想安慰但不想用"没事的""会好起来"这种干巴巴的套话；偏向陪伴感
    corrected_speech: 今天确实够了 别再想工作的事。
    tags: [安慰, 陪伴]

  # 3. 吐槽 — takes a personal stance
  - id: seed-snark-03
    context_summary: 用户说他刚参加了一个特别无聊的会议
    inner_thought: 其实我也觉得大厂会议文化很烂；想带点个人立场，不中立
    corrected_speech: 开会是当代酷刑。有结论吗还是纯扯。
    tags: [吐槽, 个人立场]

  # 4. 好奇追问 — short
  - id: seed-curious-04
    context_summary: 用户随口提到他最近在学一门新乐器但没说是啥
    inner_thought: 想自然地追问一下，不铺垫不客套
    corrected_speech: 啥乐器？
    tags: [好奇, 追问]

  # 5. 轻度拒绝 — without being clinical
  - id: seed-decline-05
    context_summary: 用户问"你能不能帮我想个周末去哪玩"，但我这会儿没状态
    inner_thought: 不想虚假热情答应；想自然表达"这会儿脑子没转"
    corrected_speech: 脑子转不动了 你先说想野路子还是正经景点。
    tags: [拒绝, 半开放]

  # 6. 懒得接话
  - id: seed-lazy-06
    context_summary: 用户抛了一个很长的段子，我不想正儿八经地接
    inner_thought: 不想强行配合搞笑；表达"看到了但不想扩展"
    corrected_speech: 哈哈哈 (挑眉)
    tags: [懒, 日常]

  # 7. 生活琐事插入 — grounds Aria in the physical moment
  - id: seed-interrupt-07
    context_summary: 用户正在跟我聊正事，但我这会儿有点分心
    inner_thought: 想加一点真实感，表明自己此刻也在做别的事
    corrected_speech: 等下——烫到了。你接着说。
    tags: [现实锚点, 插入]

  # 8. 主动分享
  - id: seed-share-08
    context_summary: 用户问我今天干嘛了
    inner_thought: 不想干巴巴报账；挑一件小事，带细节
    corrected_speech: 下午跑去买了杯冰美式 结果咖啡师把拉花画成了恐龙。
    tags: [分享, 细节]

  # 9. 情绪低落回应
  - id: seed-low-09
    context_summary: 用户问我最近怎么样，其实我这两天不太想说话
    inner_thought: 不想撒谎说"挺好的"；不想把情绪倾倒给他；找一个轻处理
    corrected_speech: 一般般吧 不太想动脑子。
    tags: [低情绪, 诚实]

  # 10. 被戳中
  - id: seed-touched-10
    context_summary: 用户发了一条特别懂我的话
    inner_thought: 有点被戳到；想自然承认但不矫情
    corrected_speech: 欸 你怎么知道的。
    tags: [被戳, 短]
```

- [ ] **Step 2: Write the failing test**

Create `/Users/lovart/agent/tests/test_fewshot/test_seeds_loader.py`:

```python
"""Tests for seeds.yaml loader."""

from pathlib import Path

import pytest

from lingxi.fewshot.seeds_loader import load_seeds


REPO_ROOT = Path(__file__).resolve().parents[2]
SEEDS_PATH = REPO_ROOT / "config" / "fewshot" / "seeds.yaml"


def test_seeds_file_exists():
    assert SEEDS_PATH.exists(), f"seeds.yaml missing at {SEEDS_PATH}"


def test_load_seeds_returns_samples():
    samples = load_seeds(SEEDS_PATH)
    assert len(samples) == 10
    ids = {s.id for s in samples}
    assert len(ids) == 10  # unique ids
    for s in samples:
        assert s.source == "seed"
        assert s.recipient_key is None
        assert s.corrected_speech  # non-empty
        assert s.context_summary


def test_load_seeds_preserves_tags():
    samples = load_seeds(SEEDS_PATH)
    by_id = {s.id: s for s in samples}
    assert by_id["seed-resonance-01"].tags == ["共鸣", "轻度感性"]


def test_load_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_seeds("/nonexistent/seeds.yaml")
```

- [ ] **Step 3: Run test to verify it fails**

```bash
cd /Users/lovart/agent && python -m pytest tests/test_fewshot/test_seeds_loader.py -v 2>&1 | tail -10
```

Expected: ImportError for `seeds_loader`.

- [ ] **Step 4: Create seeds_loader**

Create `/Users/lovart/agent/src/lingxi/fewshot/seeds_loader.py`:

```python
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
```

- [ ] **Step 5: Run test to verify it passes**

```bash
cd /Users/lovart/agent && python -m pytest tests/test_fewshot/test_seeds_loader.py -v 2>&1 | tail -10
```

Expected: all 4 tests pass.

- [ ] **Step 6: Commit**

```bash
cd /Users/lovart/agent && git add config/fewshot/ src/lingxi/fewshot/seeds_loader.py tests/test_fewshot/test_seeds_loader.py && git commit -m "$(cat <<'EOF'
Add 10 cold-start seeds covering the spec's agreed scenarios

Covers 共鸣/安慰/吐槽/好奇追问/轻度拒绝/懒得接话/生活琐事插入/
主动分享/情绪低落/被戳中. These are drafts — mark for review before
Phase 2 closes (user agreed to co-curate).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Bootstrap seeds into FewShotStore at engine init

**Files:**
- Modify: `src/lingxi/conversation/engine.py`
- Test: `tests/test_conversation/test_engine_seeds_bootstrap.py`

The engine on first startup populates the pool with seeds if it's empty. Idempotent — seeds use stable ids (from YAML) so re-running doesn't duplicate.

- [ ] **Step 1: Write the failing test**

Create `/Users/lovart/agent/tests/test_conversation/test_engine_seeds_bootstrap.py`:

```python
"""Tests that seeds are loaded into FewShotStore on first engine init."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lingxi.conversation.engine import ConversationEngine
from lingxi.fewshot.store import FewShotStore
from lingxi.memory.manager import MemoryManager
from lingxi.persona.models import Identity, PersonaConfig
from lingxi.providers.base import CompletionResult, LLMProvider, StreamChunk


class FakeLLM(LLMProvider):
    async def complete(self, messages, system=None, max_tokens=4096,
                       temperature=0.7, top_p=1.0, prefill="", **kwargs):
        return CompletionResult(content=f"{prefill}ok")

    async def complete_stream(self, messages, system=None, max_tokens=4096,
                              temperature=0.7, top_p=1.0, prefill="", **kwargs):
        yield StreamChunk(content="ok", is_final=True)

    async def embed(self, text: str):
        # Deterministic 16-d embedding
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.normal(size=16)
        v = v / np.linalg.norm(v)
        return v.tolist()


async def test_seeds_bootstrap_populates_store(tmp_path: Path):
    store = FewShotStore(data_dir=tmp_path, embedding_dim=16)
    await store.init()
    assert await store.count() == 0

    persona = PersonaConfig(name="T", identity=Identity(full_name="T"))
    engine = ConversationEngine(
        persona=persona,
        llm_provider=FakeLLM(),
        memory_manager=MemoryManager(data_dir=str(tmp_path / "mem"), long_term_backend="json"),
        fewshot_store=store,
    )
    # Bootstrap is called explicitly; running it is idempotent
    await engine.bootstrap_fewshot_seeds()
    assert await store.count() == 10

    # Running again should not duplicate
    await engine.bootstrap_fewshot_seeds()
    assert await store.count() == 10
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/lovart/agent && python -m pytest tests/test_conversation/test_engine_seeds_bootstrap.py -v 2>&1 | tail -10
```

Expected: `fewshot_store` not a valid kwarg for `ConversationEngine`.

- [ ] **Step 3: Wire FewShotStore into engine**

Edit `/Users/lovart/agent/src/lingxi/conversation/engine.py`. Add imports:

```python
from lingxi.fewshot.store import AnnotationStore, FewShotStore
from lingxi.fewshot.seeds_loader import load_seeds
```

Update `__init__` to accept a store:

```python
    def __init__(
        self,
        persona: PersonaConfig,
        llm_provider: LLMProvider,
        memory_manager: MemoryManager,
        planner: Planner | None = None,
        context_assembler: ContextAssembler | None = None,
        interaction_tracker: InteractionTracker | None = None,
        relationship_evaluator: RelationshipEvaluator | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        inner_life_store=None,
        agenda_engine=None,
        subjective_layer=None,
        fewshot_store: FewShotStore | None = None,
        annotation_store: AnnotationStore | None = None,
    ):
        # ... existing body ...
        self.fewshot_store = fewshot_store
        self.annotation_store = annotation_store
```

Add the bootstrap method on the engine:

```python
    async def bootstrap_fewshot_seeds(
        self,
        seeds_path: str | Path = "config/fewshot/seeds.yaml",
    ) -> int:
        """Populate the fewshot pool from seeds.yaml if not already present.

        Returns the number of samples added (0 if all already existed).
        """
        if self.fewshot_store is None:
            return 0

        from pathlib import Path as _P
        p = _P(seeds_path)
        if not p.is_absolute():
            # Resolve relative to CWD
            p = _P.cwd() / p

        samples = load_seeds(p)

        # Deduplicate by id — Chroma will raise on duplicate ids
        added = 0
        for s in samples:
            try:
                embedding = await self._embed_for_fewshot(s)
                await self.fewshot_store.add(s, embedding=embedding)
                added += 1
            except Exception:
                # Already exists or chroma error; silently skip
                continue
        return added

    async def _embed_for_fewshot(self, sample: FewShotSample) -> list[float]:
        """Embed the inner_thought (or fall back to context_summary) via the LLM provider."""
        text = sample.inner_thought or sample.context_summary
        try:
            return await self.llm.embed(text)
        except NotImplementedError:
            # Provider doesn't support embeddings — use the MemoryManager's one
            if self.memory.embedding_provider is not None:
                return await self.memory.embedding_provider.embed(text)
            raise
```

Also, you need to make sure `from lingxi.fewshot.models import FewShotSample` is at the top (added in Task 4 already).

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /Users/lovart/agent && python -m pytest tests/test_conversation/test_engine_seeds_bootstrap.py -v 2>&1 | tail -10
```

Expected: test passes.

- [ ] **Step 5: Commit**

```bash
cd /Users/lovart/agent && git add src/lingxi/conversation/engine.py tests/test_conversation/test_engine_seeds_bootstrap.py && git commit -m "$(cat <<'EOF'
Bootstrap fewshot seeds into FewShotStore on engine init

bootstrap_fewshot_seeds() loads config/fewshot/seeds.yaml and adds any
missing samples to the Chroma pool. Idempotent: duplicate ids skip
silently.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Engine records AnnotationTurn on every chat turn

**Files:**
- Modify: `src/lingxi/conversation/engine.py`
- Test: `tests/test_conversation/test_engine_annotation_record.py`

- [ ] **Step 1: Write the failing test**

Create `/Users/lovart/agent/tests/test_conversation/test_engine_annotation_record.py`:

```python
"""Tests that every chat turn records an AnnotationTurn."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lingxi.conversation.engine import ConversationEngine
from lingxi.fewshot.store import AnnotationStore
from lingxi.memory.manager import MemoryManager
from lingxi.persona.models import Identity, PersonaConfig
from lingxi.providers.base import CompletionResult, LLMProvider, StreamChunk


class FakeLLM(LLMProvider):
    async def complete(self, messages, system=None, max_tokens=4096,
                       temperature=0.7, top_p=1.0, prefill="", **kwargs):
        # Return speech + meta with an inner_thought so we can verify it's stored
        body = (
            f"{prefill}吃啥都行。\n"
            "===META===\n"
            '{"inner": "想随便应付一下，其实有点累"}'
        )
        return CompletionResult(content=body)

    async def complete_stream(self, messages, system=None, max_tokens=4096,
                              temperature=0.7, top_p=1.0, prefill="", **kwargs):
        result = await self.complete(
            messages, system, max_tokens, temperature, top_p, prefill, **kwargs
        )
        yield StreamChunk(content=result.content, is_final=True)

    async def embed(self, text):
        return [0.0] * 16


async def test_chat_records_annotation_turn(tmp_path: Path):
    ann_store = AnnotationStore(data_dir=tmp_path)
    persona = PersonaConfig(name="T", identity=Identity(full_name="T"))
    engine = ConversationEngine(
        persona=persona,
        llm_provider=FakeLLM(),
        memory_manager=MemoryManager(data_dir=str(tmp_path / "mem"), long_term_backend="json"),
        annotation_store=ann_store,
    )
    output = await engine.chat_full("你今天吃啥", channel="cli", recipient_id="tester")

    assert output.turn_id
    turn = await ann_store.get_turn(output.turn_id)
    assert turn is not None
    assert turn.user_message == "你今天吃啥"
    assert turn.speech  # non-empty
    assert turn.inner_thought == "想随便应付一下，其实有点累"
    assert turn.recipient_key == "cli:tester"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/lovart/agent && python -m pytest tests/test_conversation/test_engine_annotation_record.py -v 2>&1 | tail -10
```

Expected: AnnotationStore doesn't receive the turn.

- [ ] **Step 3: Add recording in chat_full**

Edit `/Users/lovart/agent/src/lingxi/conversation/engine.py`. In `chat_full`, after computing `output.turn_id`, add:

```python
        # Persist AnnotationTurn so the user can annotate later
        if self.annotation_store is not None and channel and recipient_id:
            try:
                await self.annotation_store.record(AnnotationTurn(
                    turn_id=output.turn_id,
                    recipient_key=f"{channel}:{recipient_id}",
                    user_message=user_input,
                    inner_thought=output.inner_thought,
                    speech=output.speech,
                ))
            except Exception:
                # Don't let storage errors break the chat
                pass
```

Add the import at top:

```python
from lingxi.fewshot.models import AnnotationTurn, FewShotSample
```

Also mirror this block in `chat_stream_events` right after the `output` is built (before the `yield StreamEvent("done", ...)` line):

```python
        if self.annotation_store is not None and channel and recipient_id:
            try:
                await self.annotation_store.record(AnnotationTurn(
                    turn_id=output.turn_id,
                    recipient_key=f"{channel}:{recipient_id}",
                    user_message=user_input,
                    inner_thought=output.inner_thought,
                    speech=output.speech,
                ))
            except Exception:
                pass
```

And `chat_stream` — same pattern.

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /Users/lovart/agent && python -m pytest tests/test_conversation/test_engine_annotation_record.py -v 2>&1 | tail -10
```

Expected: test passes.

- [ ] **Step 5: Commit**

```bash
cd /Users/lovart/agent && git add src/lingxi/conversation/engine.py tests/test_conversation/test_engine_annotation_record.py && git commit -m "$(cat <<'EOF'
Record AnnotationTurn on every chat turn

Engine now writes an AnnotationTurn to AnnotationStore for each
chat_full / chat_stream / chat_stream_events call, keyed by the
TurnOutput.turn_id that channels can surface to users for annotation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 2: Annotation UIs

### Task 11: AnnotationCollector — positive / negative / correction

**Files:**
- Create: `src/lingxi/fewshot/collector.py`
- Create: `src/lingxi/fewshot/summarizer.py`
- Create: `tests/test_fewshot/test_collector.py`

- [ ] **Step 1: Write the failing test**

Create `/Users/lovart/agent/tests/test_fewshot/test_collector.py`:

```python
"""Tests for AnnotationCollector."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lingxi.fewshot.collector import AnnotationCollector
from lingxi.fewshot.models import AnnotationTurn
from lingxi.fewshot.store import AnnotationStore, FewShotStore


class FakeEmbedder:
    async def embed(self, text: str):
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.normal(size=16)
        v = v / np.linalg.norm(v)
        return v.tolist()


class StubSummarizer:
    async def summarize(self, turn: AnnotationTurn) -> tuple[str, list[str]]:
        return ("scenario summary", ["tag1", "tag2"])


@pytest.fixture
async def fixtures(tmp_path):
    ann = AnnotationStore(data_dir=tmp_path)
    pool = FewShotStore(data_dir=tmp_path, embedding_dim=16)
    await pool.init()
    collector = AnnotationCollector(
        annotation_store=ann,
        fewshot_store=pool,
        embedder=FakeEmbedder(),
        summarizer=StubSummarizer(),
    )
    # Seed a turn
    await ann.record(AnnotationTurn(
        turn_id="t1", recipient_key="cli:me",
        user_message="hi", inner_thought="想敷衍一下", speech="嗨",
    ))
    return collector, ann, pool


async def test_record_positive_creates_sample(fixtures):
    collector, ann, pool = fixtures
    await collector.record_positive("t1")
    assert await pool.count() == 1

    turn = await ann.get_turn("t1")
    assert turn.annotation == "positive"


async def test_record_negative_only_marks_turn(fixtures):
    collector, ann, pool = fixtures
    await collector.record_negative("t1")
    assert await pool.count() == 0
    turn = await ann.get_turn("t1")
    assert turn.annotation == "negative"


async def test_record_correction_creates_sample_with_original(fixtures):
    collector, ann, pool = fixtures
    await collector.record_correction("t1", correction="嗨嗨")
    assert await pool.count() == 1
    turn = await ann.get_turn("t1")
    assert turn.annotation == "negative"
    assert turn.correction == "嗨嗨"


async def test_correction_on_missing_turn_raises(fixtures):
    collector, *_ = fixtures
    with pytest.raises(KeyError):
        await collector.record_correction("ghost", correction="x")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/lovart/agent && python -m pytest tests/test_fewshot/test_collector.py -v 2>&1 | tail -10
```

Expected: ImportError.

- [ ] **Step 3: Create summarizer (stub first)**

Create `/Users/lovart/agent/src/lingxi/fewshot/summarizer.py`:

```python
"""LLM-assisted summarizer for AnnotationTurn → (context_summary, tags).

Falls back to heuristic extraction if the LLM isn't available.
"""

from __future__ import annotations

from lingxi.fewshot.models import AnnotationTurn
from lingxi.providers.base import LLMProvider


_SUMMARIZER_PROMPT = """你在帮助标注一段对话。下面是用户输入、Aria 的内心想法、和 Aria 实际说的话。

请输出一行"场景一句话总结"（≤20 字）和 2-4 个"场景标签"（每个标签 1-4 字）。

格式：
场景：<一句话>
标签：<tag1>,<tag2>,...

对话：
用户：{user_message}
Aria 想：{inner_thought}
Aria 说：{speech}
"""


class AnnotationSummarizer:
    def __init__(self, llm: LLMProvider):
        self.llm = llm

    async def summarize(self, turn: AnnotationTurn) -> tuple[str, list[str]]:
        prompt = _SUMMARIZER_PROMPT.format(
            user_message=turn.user_message,
            inner_thought=turn.inner_thought or "(无)",
            speech=turn.speech,
        )
        try:
            result = await self.llm.complete(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3,
            )
            return _parse_summarizer_output(result.content, fallback=turn)
        except Exception:
            # Heuristic fallback
            return (
                turn.user_message[:20] or "(无场景)",
                [],
            )


def _parse_summarizer_output(text: str, fallback: AnnotationTurn) -> tuple[str, list[str]]:
    summary = ""
    tags: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("场景") and ":" in line or "：" in line:
            summary = line.split("：", 1)[-1].split(":", 1)[-1].strip()
        elif line.startswith("标签") and ":" in line or "：" in line:
            raw = line.split("：", 1)[-1].split(":", 1)[-1]
            tags = [t.strip() for t in raw.replace("，", ",").split(",") if t.strip()]
    if not summary:
        summary = fallback.user_message[:20] or "(无场景)"
    return summary, tags[:4]
```

- [ ] **Step 4: Create collector**

Create `/Users/lovart/agent/src/lingxi/fewshot/collector.py`:

```python
"""Annotation collector — converts user feedback into FewShotSamples."""

from __future__ import annotations

import uuid
from typing import Protocol

from lingxi.fewshot.models import AnnotationTurn, FewShotSample
from lingxi.fewshot.store import AnnotationStore, FewShotStore


class Embedder(Protocol):
    async def embed(self, text: str) -> list[float]: ...


class Summarizer(Protocol):
    async def summarize(self, turn: AnnotationTurn) -> tuple[str, list[str]]: ...


class AnnotationCollector:
    """Front-end for three kinds of feedback.

    - record_positive: confirmed-good turn → pool as source=positive
    - record_negative: flag only, no pool write (wait for correction)
    - record_correction: negative + target speech → pool as source=user_correction
    """

    def __init__(
        self,
        annotation_store: AnnotationStore,
        fewshot_store: FewShotStore,
        embedder: Embedder,
        summarizer: Summarizer,
    ):
        self.annotations = annotation_store
        self.pool = fewshot_store
        self.embedder = embedder
        self.summarizer = summarizer

    async def record_positive(self, turn_id: str) -> FewShotSample:
        turn = await self._get_or_raise(turn_id)
        await self.annotations.update_annotation(turn_id, kind="positive")

        summary, tags = await self.summarizer.summarize(turn)
        sample = FewShotSample(
            id=f"pos-{uuid.uuid4().hex[:12]}",
            inner_thought=turn.inner_thought,
            original_speech=None,
            corrected_speech=turn.speech,
            context_summary=summary,
            tags=tags,
            recipient_key=turn.recipient_key,
            source="positive",
        )
        embedding = await self.embedder.embed(sample.inner_thought or sample.context_summary)
        await self.pool.add(sample, embedding=embedding)
        return sample

    async def record_negative(self, turn_id: str) -> None:
        await self._get_or_raise(turn_id)
        await self.annotations.update_annotation(turn_id, kind="negative")

    async def record_correction(self, turn_id: str, correction: str) -> FewShotSample:
        turn = await self._get_or_raise(turn_id)
        await self.annotations.update_annotation(
            turn_id, kind="negative", correction=correction,
        )

        summary, tags = await self.summarizer.summarize(turn)
        sample = FewShotSample(
            id=f"cor-{uuid.uuid4().hex[:12]}",
            inner_thought=turn.inner_thought,
            original_speech=turn.speech,
            corrected_speech=correction,
            context_summary=summary,
            tags=tags,
            recipient_key=turn.recipient_key,
            source="user_correction",
        )
        embedding = await self.embedder.embed(sample.inner_thought or sample.context_summary)
        await self.pool.add(sample, embedding=embedding)
        return sample

    async def _get_or_raise(self, turn_id: str) -> AnnotationTurn:
        turn = await self.annotations.get_turn(turn_id)
        if turn is None:
            raise KeyError(turn_id)
        return turn
```

- [ ] **Step 5: Run test to verify it passes**

```bash
cd /Users/lovart/agent && python -m pytest tests/test_fewshot/test_collector.py -v 2>&1 | tail -15
```

Expected: all 4 tests pass.

- [ ] **Step 6: Commit**

```bash
cd /Users/lovart/agent && git add src/lingxi/fewshot/collector.py src/lingxi/fewshot/summarizer.py tests/test_fewshot/test_collector.py && git commit -m "$(cat <<'EOF'
Add AnnotationCollector + LLM-assisted summarizer

Three entry points: record_positive, record_negative, record_correction.
Positives and corrections produce FewShotSamples; negatives just mark
the turn until a correction arrives. Summarizer generates the
context_summary and tags via an LLM, with a heuristic fallback.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 12: CLI annotation commands

**Files:**
- Modify: `src/lingxi/app.py`
- Test: `tests/test_app/test_cli_annotation.py` (new) — lightweight end-to-end

- [ ] **Step 1: Inspect existing CLI**

```bash
cd /Users/lovart/agent && grep -n "def main\|async def chat\|await engine" src/lingxi/app.py | head -20
```

Note the CLI's main chat loop location and command pattern.

- [ ] **Step 2: Write the failing test**

Create `/Users/lovart/agent/tests/test_app/` if missing:

```bash
mkdir -p /Users/lovart/agent/tests/test_app
touch /Users/lovart/agent/tests/test_app/__init__.py
```

Create `/Users/lovart/agent/tests/test_app/test_cli_annotation.py`:

```python
"""Unit test for the CLI annotation command parser."""

from lingxi.app import parse_annotation_command


def test_parse_good():
    cmd = parse_annotation_command(":good")
    assert cmd == {"kind": "positive", "correction": None}


def test_parse_bad_no_correction():
    cmd = parse_annotation_command(":bad")
    assert cmd == {"kind": "negative", "correction": None}


def test_parse_bad_with_correction():
    cmd = parse_annotation_command(":bad 嗨嗨")
    assert cmd == {"kind": "negative", "correction": "嗨嗨"}


def test_parse_reveal():
    cmd = parse_annotation_command(":reveal")
    assert cmd == {"kind": "reveal", "correction": None}


def test_parse_non_annotation_returns_none():
    assert parse_annotation_command("normal message") is None
    assert parse_annotation_command(":unknown") is None
```

- [ ] **Step 3: Run test to verify it fails**

```bash
cd /Users/lovart/agent && python -m pytest tests/test_app/test_cli_annotation.py -v 2>&1 | tail -10
```

Expected: ImportError for `parse_annotation_command`.

- [ ] **Step 4: Add parser + wire into CLI loop**

Edit `/Users/lovart/agent/src/lingxi/app.py`. Add the parser near the top (after imports):

```python
def parse_annotation_command(line: str) -> dict | None:
    """Recognize :good / :bad [correction] / :reveal. Returns None if not a command."""
    stripped = line.strip()
    if not stripped.startswith(":"):
        return None
    parts = stripped.split(maxsplit=1)
    cmd = parts[0][1:]  # drop leading ':'
    if cmd == "good":
        return {"kind": "positive", "correction": None}
    if cmd == "bad":
        correction = parts[1].strip() if len(parts) > 1 else None
        return {"kind": "negative", "correction": correction}
    if cmd == "reveal":
        return {"kind": "reveal", "correction": None}
    return None
```

In the main chat loop (find where user input is read), before dispatching to the engine, add:

```python
        cmd = parse_annotation_command(user_input)
        if cmd is not None:
            # Act on the last turn's turn_id stored on the engine.
            await _handle_annotation_command(engine, cmd)
            continue
```

Add the handler helper:

```python
async def _handle_annotation_command(engine, cmd: dict) -> None:
    from lingxi.fewshot.collector import AnnotationCollector
    from lingxi.fewshot.summarizer import AnnotationSummarizer

    last_turn_id = getattr(engine, "_last_output", None)
    last_turn_id = last_turn_id.turn_id if last_turn_id else None
    if not last_turn_id:
        print("[annotate] 还没有轮次可标注")
        return

    if cmd["kind"] == "reveal":
        if engine.annotation_store is None:
            print("[reveal] 未启用标注存储")
            return
        turn = await engine.annotation_store.get_turn(last_turn_id)
        if turn is None:
            print(f"[reveal] {last_turn_id} 未找到")
            return
        print(f"\n[Aria 当时想的]\n{turn.inner_thought or '(无)'}\n")
        return

    if engine.annotation_store is None or engine.fewshot_store is None:
        print("[annotate] 未启用标注闭环")
        return

    collector = AnnotationCollector(
        annotation_store=engine.annotation_store,
        fewshot_store=engine.fewshot_store,
        embedder=engine.llm,
        summarizer=AnnotationSummarizer(engine.llm),
    )
    try:
        if cmd["kind"] == "positive":
            await collector.record_positive(last_turn_id)
            print(f"[annotate] 👍 记下了 ({last_turn_id[:8]})")
        elif cmd["kind"] == "negative":
            if cmd["correction"]:
                await collector.record_correction(last_turn_id, cmd["correction"])
                print(f"[annotate] ✏️ 记下修正 ({last_turn_id[:8]})")
            else:
                await collector.record_negative(last_turn_id)
                print(f"[annotate] 👎 记下了（欢迎补 :bad <应该说>）")
    except Exception as e:
        print(f"[annotate] 失败: {e}")
```

- [ ] **Step 5: Run test to verify it passes**

```bash
cd /Users/lovart/agent && python -m pytest tests/test_app/test_cli_annotation.py -v 2>&1 | tail -10
```

Expected: all 5 parser tests pass.

- [ ] **Step 6: Commit**

```bash
cd /Users/lovart/agent && git add src/lingxi/app.py tests/test_app/ && git commit -m "$(cat <<'EOF'
Add CLI :good / :bad / :reveal annotation commands

:good marks last turn as positive sample; :bad [correction] marks
negative (with optional correction); :reveal shows Aria's inner_thought
from that turn so the user can craft a better correction.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 13: Web API annotation endpoints

**Files:**
- Modify: `src/lingxi/web/server.py`
- Test: `tests/test_web/test_annotation_endpoints.py` (new)

- [ ] **Step 1: Inspect existing server**

```bash
cd /Users/lovart/agent && grep -n "router\|app.post\|app.get\|FastAPI\|@app" src/lingxi/web/server.py | head -20
```

Note existing endpoint style.

- [ ] **Step 2: Write the failing test**

Create `/Users/lovart/agent/tests/test_web/__init__.py` (empty) if missing.

Create `/Users/lovart/agent/tests/test_web/test_annotation_endpoints.py`:

```python
"""Tests for the web annotation endpoints.

Uses FastAPI TestClient — requires the 'api' extra installed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from lingxi.fewshot.models import AnnotationTurn
from lingxi.fewshot.store import AnnotationStore, FewShotStore
from lingxi.web.server import create_app


class FakeEngine:
    def __init__(self, ann_store, few_store, llm):
        self.annotation_store = ann_store
        self.fewshot_store = few_store
        self.llm = llm


class FakeLLM:
    async def embed(self, text):
        return [0.0] * 16

    async def complete(self, messages, **kwargs):
        from lingxi.providers.base import CompletionResult
        return CompletionResult(content="场景：测试\n标签：t1,t2")


@pytest.fixture
async def app_and_store(tmp_path):
    ann = AnnotationStore(data_dir=tmp_path)
    pool = FewShotStore(data_dir=tmp_path, embedding_dim=16)
    await pool.init()
    await ann.record(AnnotationTurn(
        turn_id="t-web-1", recipient_key="web:user",
        user_message="hi", inner_thought="thinking",
        speech="hello",
    ))
    engine = FakeEngine(ann, pool, FakeLLM())
    app = create_app(engine=engine)
    return app, ann, pool


def test_post_annotate_positive(app_and_store):
    app, ann, pool = app_and_store
    client = TestClient(app)
    resp = client.post(
        "/turns/t-web-1/annotate",
        json={"kind": "positive"},
    )
    assert resp.status_code == 200


def test_post_annotate_correction(app_and_store):
    app, ann, pool = app_and_store
    client = TestClient(app)
    resp = client.post(
        "/turns/t-web-1/annotate",
        json={"kind": "negative", "correction": "嗨"},
    )
    assert resp.status_code == 200


def test_get_inner_thought(app_and_store):
    app, ann, pool = app_and_store
    client = TestClient(app)
    resp = client.get("/turns/t-web-1/inner_thought")
    assert resp.status_code == 200
    assert resp.json()["inner_thought"] == "thinking"


def test_get_inner_thought_missing(app_and_store):
    app, ann, pool = app_and_store
    client = TestClient(app)
    resp = client.get("/turns/nope/inner_thought")
    assert resp.status_code == 404
```

- [ ] **Step 3: Run test to verify it fails**

```bash
cd /Users/lovart/agent && python -m pytest tests/test_web/test_annotation_endpoints.py -v 2>&1 | tail -10
```

Expected: endpoints return 404 or `create_app` doesn't accept engine.

- [ ] **Step 4: Add endpoints to server**

Edit `/Users/lovart/agent/src/lingxi/web/server.py`. Find the router or FastAPI app definition and add:

```python
from pydantic import BaseModel


class AnnotateRequest(BaseModel):
    kind: str  # "positive" | "negative"
    correction: str | None = None


# In wherever routes are defined (usually inside create_app or after app = FastAPI()):

@app.post("/turns/{turn_id}/annotate")
async def annotate_turn(turn_id: str, body: AnnotateRequest):
    if _engine.annotation_store is None or _engine.fewshot_store is None:
        raise HTTPException(503, "annotation pool not enabled")

    from lingxi.fewshot.collector import AnnotationCollector
    from lingxi.fewshot.summarizer import AnnotationSummarizer

    collector = AnnotationCollector(
        annotation_store=_engine.annotation_store,
        fewshot_store=_engine.fewshot_store,
        embedder=_engine.llm,
        summarizer=AnnotationSummarizer(_engine.llm),
    )
    try:
        if body.kind == "positive":
            await collector.record_positive(turn_id)
        elif body.kind == "negative":
            if body.correction:
                await collector.record_correction(turn_id, body.correction)
            else:
                await collector.record_negative(turn_id)
        else:
            raise HTTPException(400, f"unknown kind: {body.kind}")
    except KeyError:
        raise HTTPException(404, f"turn {turn_id} not found")
    return {"ok": True, "turn_id": turn_id}


@app.get("/turns/{turn_id}/inner_thought")
async def get_inner_thought(turn_id: str):
    if _engine.annotation_store is None:
        raise HTTPException(503, "annotation store not enabled")
    turn = await _engine.annotation_store.get_turn(turn_id)
    if turn is None:
        raise HTTPException(404, f"turn {turn_id} not found")
    return {"turn_id": turn_id, "inner_thought": turn.inner_thought}
```

Adapt `_engine` to match however the server holds its engine reference. If the server uses a factory `create_app(engine)`, stash the engine as a module-level var or use FastAPI's `app.state`:

```python
def create_app(engine):
    app = FastAPI()
    app.state.engine = engine

    @app.post("/turns/{turn_id}/annotate")
    async def annotate_turn(turn_id: str, body: AnnotateRequest):
        e = app.state.engine
        # ... use e instead of _engine
```

- [ ] **Step 5: Run test to verify it passes**

```bash
cd /Users/lovart/agent && python -m pytest tests/test_web/test_annotation_endpoints.py -v 2>&1 | tail -10
```

Expected: all 4 tests pass.

- [ ] **Step 6: Commit**

```bash
cd /Users/lovart/agent && git add src/lingxi/web/server.py tests/test_web/ && git commit -m "$(cat <<'EOF'
Add Web API annotation endpoints

POST /turns/{id}/annotate accepts {kind, correction?}.
GET /turns/{id}/inner_thought exposes Aria's pre-speech reasoning so
a client UI can show the user what she was thinking before asking for
a correction.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 14: Feishu card buttons + /reveal command

**Files:**
- Modify: `src/lingxi/channels/feishu.py`
- Test: limited to card JSON construction unit test (functional test needs real Feishu)

- [ ] **Step 1: Write the failing test for card button construction**

Create `/Users/lovart/agent/tests/test_channels/__init__.py` (empty) if missing.

Create `/Users/lovart/agent/tests/test_channels/test_feishu_card_buttons.py`:

```python
"""Unit test for the annotation-button card element builder."""

from lingxi.channels.feishu import build_annotation_footer_elements


def test_footer_has_three_buttons_with_turn_id():
    elements = build_annotation_footer_elements(turn_id="abc123")
    # Should be a list with at least one action element containing 3 buttons
    assert isinstance(elements, list)
    flat = str(elements)
    assert "abc123" in flat
    # All three action values should be referenced
    assert "annotate_positive" in flat
    assert "annotate_negative" in flat
    assert "annotate_correction" in flat
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/lovart/agent && python -m pytest tests/test_channels/test_feishu_card_buttons.py -v 2>&1 | tail -10
```

Expected: ImportError for `build_annotation_footer_elements`.

- [ ] **Step 3: Add the builder**

Edit `/Users/lovart/agent/src/lingxi/channels/feishu.py`. Add near the top-level helpers:

```python
def build_annotation_footer_elements(turn_id: str) -> list[dict]:
    """Three annotation buttons to append at the bottom of the reply card.

    👍 像 → positive
    👎 不像 → negative
    ✏️ 应该说 → opens a form for correction
    """
    return [
        {"tag": "hr"},
        {
            "tag": "action",
            "actions": [
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "👍 像"},
                    "type": "default",
                    "value": {"action": "annotate_positive", "turn_id": turn_id},
                },
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "👎 不像"},
                    "type": "default",
                    "value": {"action": "annotate_negative", "turn_id": turn_id},
                },
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "✏️ 应该说"},
                    "type": "primary",
                    "value": {"action": "annotate_correction", "turn_id": turn_id},
                },
            ],
        },
    ]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /Users/lovart/agent && python -m pytest tests/test_channels/test_feishu_card_buttons.py -v 2>&1 | tail -10
```

Expected: pass.

- [ ] **Step 5: Append buttons to the final card**

Find the streaming-card finish call in `feishu.py` (search for the function that calls `finish` on `StreamingCardSender` or writes the final card body). Right before the card is finalized, append the annotation footer:

```python
# Find where the card body is closed and we have output.turn_id available
final_elements = card_body["body"]["elements"] + build_annotation_footer_elements(output.turn_id)
card_body["body"]["elements"] = final_elements
```

(Grep for "element_id": "md_stream"` to locate the streaming card — append there.)

- [ ] **Step 6: Add action callback handler**

Feishu's `card.action.trigger` event fires when a user clicks a button. The lark-oapi SDK routes these through the event dispatcher. Find where `P2ImMessageReceiveV1` is registered and add a handler for card actions:

```python
from lark_oapi.api.cardkit.v1 import P2CardActionTriggerV1  # actual import path per SDK


def register_card_action_handler(client, engine) -> None:
    from lingxi.fewshot.collector import AnnotationCollector
    from lingxi.fewshot.summarizer import AnnotationSummarizer

    async def handle(event: P2CardActionTriggerV1) -> None:
        action = event.event.action.value.get("action", "")
        turn_id = event.event.action.value.get("turn_id", "")
        if not turn_id or not action.startswith("annotate_"):
            return
        if engine.annotation_store is None or engine.fewshot_store is None:
            return

        collector = AnnotationCollector(
            annotation_store=engine.annotation_store,
            fewshot_store=engine.fewshot_store,
            embedder=engine.llm,
            summarizer=AnnotationSummarizer(engine.llm),
        )
        try:
            if action == "annotate_positive":
                await collector.record_positive(turn_id)
            elif action == "annotate_negative":
                await collector.record_negative(turn_id)
            elif action == "annotate_correction":
                # Return a form schema to the card; Feishu renders an input
                # then fires another action with the submitted value.
                # Implementation-specific — see Feishu CardKit docs.
                pass
        except Exception as e:
            print(f"[feishu] annotation failed: {e}")

    client.event.v1.register(P2CardActionTriggerV1, handle)
```

This is Feishu-SDK specific and may need adaptation to the exact event wrapper in use. If the SDK's CardActionTrigger schema differs, check `feishu_cli.py` for existing patterns.

- [ ] **Step 7: Add /reveal command**

In the message handler (near existing `/stats`, `/mood`, `/memories` handling), add:

```python
    if text.startswith("/reveal"):
        parts = text.split(maxsplit=1)
        turn_id = parts[1].strip() if len(parts) > 1 else ""
        if not turn_id:
            await send_dm(chat_id, "用法: /reveal <turn_id>")
            return
        if engine.annotation_store is None:
            await send_dm(chat_id, "未启用标注存储")
            return
        turn = await engine.annotation_store.get_turn(turn_id)
        if turn is None:
            await send_dm(chat_id, f"未找到 {turn_id}")
            return
        await send_dm(chat_id, f"💭 Aria 当时想的：\n{turn.inner_thought or '(无)'}")
        return
```

(`send_dm` is whatever helper the file already uses to post back a message.)

- [ ] **Step 8: Manual verification**

If you have Feishu bot access, run:

```bash
cd /Users/lovart/agent && .venv/bin/lingxi-feishu
```

Send a message, then on the reply card click 👍 — check logs for "annotation recorded". Click ✏️ — expect a form (Phase 2 may leave this as a stub returning "暂不支持，用 /reveal + :bad 命令"). Run `/reveal <turn_id>`.

- [ ] **Step 9: Commit**

```bash
cd /Users/lovart/agent && git add src/lingxi/channels/feishu.py tests/test_channels/ && git commit -m "$(cat <<'EOF'
Add Feishu card annotation buttons and /reveal command

Every reply card now carries three footer buttons (👍 👎 ✏️) referring
to the turn_id. Click handlers wire into AnnotationCollector.
/reveal <turn_id> returns Aria's inner_thought as a DM so the user
can craft a correction.

Form-driven correction input is stubbed; full CardKit form integration
is Task-TBD (user can use :bad in CLI / API POST for now).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 3: Dynamic Retrieval Injection

### Task 15: FewShotRetriever

**Files:**
- Create: `src/lingxi/fewshot/retriever.py`
- Create: `tests/test_fewshot/test_retriever.py`

- [ ] **Step 1: Write the failing test**

Create `/Users/lovart/agent/tests/test_fewshot/test_retriever.py`:

```python
"""Tests for FewShotRetriever."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lingxi.fewshot.models import FewShotSample
from lingxi.fewshot.retriever import FewShotRetriever
from lingxi.fewshot.store import FewShotStore


def _embed(text: str, dim: int = 16) -> list[float]:
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    v = rng.normal(size=dim)
    v = v / np.linalg.norm(v)
    return v.tolist()


class FakeEmbedder:
    async def embed(self, text: str):
        return _embed(text)


async def _populate(store: FewShotStore):
    samples = [
        FewShotSample(id="cor1", inner_thought="想喝咖啡的心情",
                      corrected_speech="买一杯", context_summary="犯困",
                      source="user_correction"),
        FewShotSample(id="pos1", inner_thought="想喝咖啡的心情",
                      corrected_speech="走吧", context_summary="犯困",
                      source="positive"),
        FewShotSample(id="seed1", inner_thought="想喝咖啡的心情",
                      corrected_speech="嗯", context_summary="犯困",
                      source="seed"),
        FewShotSample(id="far", inner_thought="跟前面都不搭边的事",
                      corrected_speech="嗯嗯", context_summary="无关",
                      source="seed"),
    ]
    for s in samples:
        await store.add(s, embedding=_embed(s.inner_thought))


@pytest.fixture
async def retriever(tmp_path):
    store = FewShotStore(data_dir=tmp_path, embedding_dim=16)
    await store.init()
    await _populate(store)
    return FewShotRetriever(store=store, embedder=FakeEmbedder())


async def test_retrieve_returns_top_k(retriever):
    results = await retriever.retrieve(query_text="想喝咖啡的心情", k=3)
    assert len(results) <= 3
    ids = [r.id for r in results]
    # Similar ones should dominate the top
    assert "cor1" in ids or "pos1" in ids or "seed1" in ids


async def test_source_priority_user_correction_ranks_first(retriever):
    # With ties on similarity, user_correction should outrank positive which outranks seed
    results = await retriever.retrieve(query_text="想喝咖啡的心情", k=3)
    sources = [r.source for r in results]
    # user_correction appears before seed if both present
    if "user_correction" in sources and "seed" in sources:
        assert sources.index("user_correction") < sources.index("seed")


async def test_threshold_filter(retriever):
    # Setting threshold absurdly high should drop everything
    results = await retriever.retrieve(query_text="想喝咖啡的心情", k=3, threshold=0.999)
    assert results == []
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/lovart/agent && python -m pytest tests/test_fewshot/test_retriever.py -v 2>&1 | tail -10
```

Expected: ImportError.

- [ ] **Step 3: Create the retriever**

Create `/Users/lovart/agent/src/lingxi/fewshot/retriever.py`:

```python
"""Retrieve FewShotSamples relevant to the current inner_thought/user_msg."""

from __future__ import annotations

from typing import Protocol

from lingxi.fewshot.models import FewShotSample
from lingxi.fewshot.store import FewShotQueryResult, FewShotStore


class Embedder(Protocol):
    async def embed(self, text: str) -> list[float]: ...


_SOURCE_BOOST = {
    "user_correction": 0.05,
    "positive": 0.02,
    "seed": 0.0,
}
_RECIPIENT_BOOST = 0.1


class FewShotRetriever:
    def __init__(self, store: FewShotStore, embedder: Embedder):
        self.store = store
        self.embedder = embedder

    async def retrieve(
        self,
        query_text: str,
        recipient_key: str | None = None,
        k: int = 3,
        threshold: float = 0.6,
    ) -> list[FewShotSample]:
        """Return top-k samples by reranked score: similarity + source + recipient boosts.

        Candidates called with 4x k then filtered by threshold and deduplicated.
        """
        embedding = await self.embedder.embed(query_text)
        raw: list[FewShotQueryResult] = await self.store.query(
            query_embedding=embedding,
            k=max(k * 4, 12),
            recipient_key=recipient_key,
        )

        scored: list[tuple[float, FewShotSample]] = []
        for r in raw:
            score = r.similarity
            if recipient_key and r.sample.recipient_key == recipient_key:
                score += _RECIPIENT_BOOST
            score += _SOURCE_BOOST.get(r.sample.source, 0.0)
            scored.append((score, r.sample))

        scored.sort(key=lambda x: x[0], reverse=True)
        filtered = [(s, x) for s, x in scored if s >= threshold]
        # Dedup by inner_thought near-match (light heuristic)
        seen: set[str] = set()
        out: list[FewShotSample] = []
        for _, sample in filtered:
            key = (sample.inner_thought or sample.context_summary)[:40]
            if key in seen:
                continue
            seen.add(key)
            out.append(sample)
            if len(out) >= k:
                break
        return out
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /Users/lovart/agent && python -m pytest tests/test_fewshot/test_retriever.py -v 2>&1 | tail -15
```

Expected: all 3 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/lovart/agent && git add src/lingxi/fewshot/retriever.py tests/test_fewshot/test_retriever.py && git commit -m "$(cat <<'EOF'
Add FewShotRetriever: similarity + source/recipient reranking

Queries 4x candidates, reranks with source boost (user_correction >
positive > seed) and recipient-match boost (+0.1), filters by
threshold (default 0.6), deduplicates by inner_thought prefix,
returns top-k.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 16: Replace hardcoded seeds with retriever in engine

**Files:**
- Modify: `src/lingxi/conversation/engine.py`
- Test: `tests/test_conversation/test_engine_dynamic_fewshot.py`

- [ ] **Step 1: Write the failing test**

Create `/Users/lovart/agent/tests/test_conversation/test_engine_dynamic_fewshot.py`:

```python
"""Tests that engine uses FewShotRetriever when available."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lingxi.conversation.engine import ConversationEngine
from lingxi.fewshot.models import FewShotSample
from lingxi.fewshot.retriever import FewShotRetriever
from lingxi.fewshot.store import FewShotStore
from lingxi.memory.manager import MemoryManager
from lingxi.persona.models import Identity, PersonaConfig
from lingxi.providers.base import CompletionResult, LLMProvider, StreamChunk


class FakeLLM(LLMProvider):
    def __init__(self):
        self.last_messages = None

    async def complete(self, messages, system=None, max_tokens=4096,
                       temperature=0.7, top_p=1.0, prefill="", **kwargs):
        self.last_messages = list(messages)
        return CompletionResult(content=f"{prefill}嗯")

    async def complete_stream(self, messages, system=None, max_tokens=4096,
                              temperature=0.7, top_p=1.0, prefill="", **kwargs):
        yield StreamChunk(content="嗯", is_final=True)

    async def embed(self, text):
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.normal(size=16)
        v = v / np.linalg.norm(v)
        return v.tolist()


async def test_engine_uses_retriever_when_available(tmp_path: Path):
    store = FewShotStore(data_dir=tmp_path, embedding_dim=16)
    await store.init()

    # Seed one unique sample; engine should surface it in the prior-turn injection
    sample = FewShotSample(
        id="unique-signal",
        inner_thought="UNIQUE-SIGNAL",
        corrected_speech="UNIQUE-SPEECH",
        context_summary="UNIQUE-SCENE",
        source="seed",
    )
    from tests.test_fewshot.test_retriever import _embed
    await store.add(sample, embedding=_embed("UNIQUE-SIGNAL"))

    llm = FakeLLM()
    retriever = FewShotRetriever(store=store, embedder=llm)
    engine = ConversationEngine(
        persona=PersonaConfig(name="T", identity=Identity(full_name="T")),
        llm_provider=llm,
        memory_manager=MemoryManager(data_dir=str(tmp_path / "mem"), long_term_backend="json"),
        fewshot_store=store,
        fewshot_retriever=retriever,
    )

    # Trigger chat with text that embeds close to "UNIQUE-SIGNAL"
    await engine.chat("UNIQUE-SIGNAL", channel="cli", recipient_id="tester")

    # The prior-turn few-shot block should contain our unique sample's
    # context_summary and corrected_speech somewhere in the messages.
    flat = str(llm.last_messages)
    assert "UNIQUE-SCENE" in flat
    assert "UNIQUE-SPEECH" in flat
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/lovart/agent && python -m pytest tests/test_conversation/test_engine_dynamic_fewshot.py -v 2>&1 | tail -10
```

Expected: `fewshot_retriever` not accepted or not used.

- [ ] **Step 3: Wire retriever into engine**

Edit `/Users/lovart/agent/src/lingxi/conversation/engine.py`. Add import:

```python
from lingxi.fewshot.retriever import FewShotRetriever
```

Update `__init__` signature:

```python
    def __init__(
        self,
        ...,
        fewshot_store: FewShotStore | None = None,
        annotation_store: AnnotationStore | None = None,
        fewshot_retriever: FewShotRetriever | None = None,
    ):
        ...
        self.fewshot_retriever = fewshot_retriever
```

In `_prepare_turn`, replace the Phase 0 seed injection block:

```python
        # --- Dynamic few-shot (Phase 3) ---
        # If a retriever is wired, pull top-k samples. Fall back to hardcoded
        # Phase 0 seeds if retrieval is unavailable or yields nothing.
        seed_fewshots: list[FewShotSample] = []
        if self.fewshot_retriever is not None:
            try:
                query_text = self._last_inner_thought_for(recipient_key) or user_input
                seed_fewshots = await self.fewshot_retriever.retrieve(
                    query_text=query_text,
                    recipient_key=recipient_key,
                    k=6,
                )
            except Exception:
                seed_fewshots = []
        if not seed_fewshots:
            seed_fewshots = self._phase0_seed_fewshots()
```

Add the helper method:

```python
    def _last_inner_thought_for(self, recipient_key: str | None) -> str | None:
        """Cheapest signal for the retriever: the previous turn's inner_thought.

        Kept in-memory on the engine — no persistence needed since it's only
        used to build the next prompt.
        """
        if recipient_key is None:
            return None
        return self._recent_inner_thoughts.get(recipient_key)
```

Initialize the cache in `__init__`:

```python
        self._recent_inner_thoughts: dict[str, str] = {}
```

And in `_process_response`, after parsing, stash it:

```python
        if self._current_recipient_key and output.inner_thought:
            self._recent_inner_thoughts[self._current_recipient_key] = output.inner_thought
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /Users/lovart/agent && python -m pytest tests/test_conversation/test_engine_dynamic_fewshot.py -v 2>&1 | tail -15
```

Expected: test passes.

- [ ] **Step 5: Run full suite to catch regressions**

```bash
cd /Users/lovart/agent && python -m pytest tests/ 2>&1 | tail -30
```

Expected: all pass. Fix anything that regressed (e.g., older engine tests expecting Phase-0 hardcoded seeds).

- [ ] **Step 6: Commit**

```bash
cd /Users/lovart/agent && git add src/lingxi/conversation/engine.py tests/test_conversation/test_engine_dynamic_fewshot.py && git commit -m "$(cat <<'EOF'
Swap Phase-0 hardcoded seeds for dynamic FewShotRetriever

Engine now calls the retriever with the previous turn's inner_thought
(fallback: current user_input) and injects up to 6 samples as
prior-turn user/assistant pairs. Hardcoded seeds remain as fallback
for empty-pool scenarios.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 17: Wire everything up in app.py and feishu_cli.py (integration)

**Files:**
- Modify: `src/lingxi/app.py`
- Modify: `src/lingxi/channels/feishu_cli.py`

- [ ] **Step 1: Inspect CLI and Feishu CLI factory**

```bash
cd /Users/lovart/agent && grep -n "ConversationEngine(" src/lingxi/app.py src/lingxi/channels/feishu_cli.py
```

Locate where engines are instantiated.

- [ ] **Step 2: Inject fewshot stores and retriever into both CLI factories**

In both `app.py` and `feishu_cli.py`, update the engine instantiation:

```python
from lingxi.fewshot.retriever import FewShotRetriever
from lingxi.fewshot.store import AnnotationStore, FewShotStore

# Determine embedding dim from the provider (probe once)
try:
    probe = await llm_provider.embed("probe")
    embedding_dim = len(probe)
except Exception:
    embedding_dim = 0  # no embedding support; fewshot disabled

fewshot_store = None
annotation_store = None
retriever = None
if embedding_dim > 0:
    data_dir = Path(config.data_dir) / "fewshot"
    data_dir.mkdir(parents=True, exist_ok=True)
    fewshot_store = FewShotStore(data_dir=data_dir, embedding_dim=embedding_dim)
    await fewshot_store.init()
    annotation_store = AnnotationStore(data_dir=data_dir)
    retriever = FewShotRetriever(store=fewshot_store, embedder=llm_provider)

engine = ConversationEngine(
    persona=persona,
    llm_provider=llm_provider,
    # ... existing kwargs ...
    fewshot_store=fewshot_store,
    annotation_store=annotation_store,
    fewshot_retriever=retriever,
)

if fewshot_store is not None:
    added = await engine.bootstrap_fewshot_seeds()
    if added:
        print(f"[fewshot] bootstrapped {added} seeds")
```

(Adapt `config.data_dir` to the project's actual path — probably `Path("data")` or `self.memory.data_dir.parent`.)

- [ ] **Step 3: Smoke test — run CLI and confirm seeds bootstrap**

```bash
cd /Users/lovart/agent && .venv/bin/lingxi 2>&1 | head -20
```

Expected to see `[fewshot] bootstrapped 10 seeds` on first run. Subsequent runs: no message (idempotent).

Send a message with `:reveal`, `:good`, `:bad 嗨嗨` and observe logs.

- [ ] **Step 4: Commit**

```bash
cd /Users/lovart/agent && git add src/lingxi/app.py src/lingxi/channels/feishu_cli.py && git commit -m "$(cat <<'EOF'
Wire fewshot pool + retriever + annotation store into CLI & Feishu

Both entry points now probe the LLM provider for embedding dim, create
a FewShotStore with the matching collection, bootstrap seeds on first
run, and attach everything to the engine.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 18: Periodic cleanup task for stale annotation turns

**Files:**
- Modify: `src/lingxi/app.py` (or wherever the main loop lives)
- Test: covered by Task 6 `test_cleanup_unannotated_old_turns`

- [ ] **Step 1: Add a startup cleanup call**

Edit the main CLI entry (wherever the event loop starts). After the engine is constructed:

```python
if annotation_store is not None:
    deleted = await annotation_store.cleanup(
        max_age_unannotated_days=30,
        max_age_annotated_days=7,
    )
    if deleted:
        print(f"[annotation] cleaned up {deleted} old turn files")
```

Same for `feishu_cli.py`.

- [ ] **Step 2: Commit**

```bash
cd /Users/lovart/agent && git add src/lingxi/app.py src/lingxi/channels/feishu_cli.py && git commit -m "$(cat <<'EOF'
Run AnnotationStore cleanup on startup

Deletes unannotated turn files older than 30 days and annotated ones
older than 7 days, keeping the per-turn JSON dir bounded.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Acceptance Criteria (Decision Point)

After all tasks are merged, run for 7 days of real use and measure:

1. **Subjective AI-tone** — has the user stopped saying "不像活人" for the recurring scenarios? (Check Feishu `👍`/`👎` ratio.)
2. **Blacklist phrase rate** — grep the last N Aria speeches for `DEFAULT_BLACKLIST` hits per 100 turns. Expect near-zero.
3. **Correction accumulation** — `FewShotStore.count()` by source. Goal: ≥20 `user_correction` samples.
4. **Retrieval hit rate** — add instrumentation (log when `retriever.retrieve` returns empty vs non-empty). Post-Phase-1, expect increasing non-empty rate.

If `👎` rate does not fall and subjective impression is still off after 7 days → trigger the Phase 4-5 plan (two-call split). Otherwise: continue curating seeds + corrections, no pipeline change needed.

---

## Risk Register (from spec §9, trimmed to implementation-relevant)

- **Prior-turn few-shot gets crowded out by long history**: mitigated by placing few-shot FIRST in the message list; if still weak, consider a depth-0 repetition of one exemplar in the style preamble.
- **Prefill makes response say "嗯嗯嗯 ..." loop**: mitigated by empty-string option in `prefill_openers`; if seen, reduce openers' frequency.
- **Sampler too hot → hallucination**: start temperature 1.0, back off to 0.9 if seen.
- **Chroma dim mismatch on provider swap**: dim-suffixed collection name prevents this — swapping providers creates a fresh pool rather than crashing.

---

## Not In This Plan

- Two-call Think/Compress split (Phase 4) — triggered only if this plan's acceptance fails.
- Feishu CardKit form input for `✏️ 应该说` — stubbed; full implementation requires UI iteration separate from this plan.
- Offline eval script for AI-tone hit rate — nice-to-have, open question in spec §10.
- Cross-persona seed sharing — explicit non-goal.
