"""Export annotated samples to 火山方舟 SFT JSONL format.

方舟 SFT 期望 OpenAI chat 格式：每行一个 {"messages": [...]}。

数据源：data/fewshot/fewshot/samples.jsonl（user_correction + positive，
跳过 seed —— seed 是冷启动的硬编码模板，不是真实标注）。

使用方式：
    .venv/bin/python tools/export_sft_dataset.py [persona_yaml] [out_path]

默认：
    persona = config/personas/example_persona.yaml
    out     = data/sft_dataset.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def build_system_prompt(persona_path: Path) -> str:
    """Distill persona into a compact system prompt for training."""
    import yaml

    if not persona_path.exists():
        return "你是 Aria。"

    p = yaml.safe_load(persona_path.read_text(encoding="utf-8"))
    name = p.get("name", "Aria")
    identity = p.get("identity", {})
    full_name = identity.get("full_name", name)
    age = identity.get("age")
    occupation = identity.get("occupation", "")
    background = (identity.get("background") or "").strip()

    style = p.get("speaking_style", {})
    tone = style.get("tone", "")
    vocab = style.get("vocabulary_level", "")

    traits = p.get("personality", {}).get("traits", [])
    top = sorted(traits, key=lambda t: t.get("intensity", 0), reverse=True)[:3]
    trait_names = "、".join(t["trait"] for t in top)

    parts = [f"你是 {full_name}（{name}）。"]
    if age:
        parts.append(f"年龄 {age} 岁。")
    if occupation:
        parts.append(f"职业：{occupation}。")
    if background:
        parts.append(f"背景：{background}")
    if tone or vocab:
        parts.append(f"语感：{tone}{('，' + vocab) if vocab else ''}。")
    if trait_names:
        parts.append(f"核心性格：{trait_names}。")
    parts.append("在 IM 里聊天用口语短句，不写小作文，不用 `——` 抒情停顿。")
    return "\n".join(parts)


def main() -> int:
    args = sys.argv[1:]
    persona_path = Path(args[0]) if args else Path("config/personas/example_persona.yaml")
    out_path = Path(args[1]) if len(args) > 1 else Path("data/sft_dataset.jsonl")

    samples_path = Path("data/fewshot/fewshot/samples.jsonl")
    if not samples_path.exists():
        print(f"✗ {samples_path} 不存在")
        return 1

    system_prompt = build_system_prompt(persona_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = 0
    n_kept = 0
    n_correction = 0
    n_positive = 0

    with samples_path.open(encoding="utf-8") as fin, out_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            try:
                s = json.loads(line)
            except json.JSONDecodeError:
                continue
            src = s.get("source")
            if src not in ("user_correction", "positive"):
                continue

            user_side = (s.get("context_summary") or "").strip()
            assistant_side = (s.get("corrected_speech") or "").strip()
            if not user_side or not assistant_side:
                continue

            record = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_side},
                    {"role": "assistant", "content": assistant_side},
                ]
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_kept += 1
            if src == "user_correction":
                n_correction += 1
            else:
                n_positive += 1

    print(f"✓ 导出 {n_kept} 条 ({n_correction} corrections + {n_positive} positives)")
    print(f"  源样本: {n_total} 条 (跳过 seed)")
    print(f"  写入:   {out_path}")

    if n_kept < 50:
        print(
            f"\n⚠️  只有 {n_kept} 条，方舟 SFT 一般 ≥50 起步，"
            "建议攒到 80-100 条再开训"
        )
    elif n_kept < 200:
        print(f"\n→ {n_kept} 条够跑一个 LoRA SFT 试试，迭代两轮")
    else:
        print(f"\n→ {n_kept} 条够正经训了")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
