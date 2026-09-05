"""Pick facts that cover different things, not the top N of one thing.

Selecting the highest-scoring N along a single axis returns a cluster. The
opener block asks for the eight best facts about him and, on 2026-08-31, got
eight slots holding three subjects: the autograph at the Chengdu show three
times over, the handwritten letter twice, the trip to see Liyuu twice. She
held twenty-eight distinct facts at the time and could see three of them. It
shows in what she says — she circles the same thing.

Write-side dedup does not help with this and is not supposed to. It stops the
same sentence being stored twice. Eight true facts about one afternoon are
eight true facts; they simply should not take the whole block.
"""

from __future__ import annotations


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


async def select_diverse(facts: list, k: int, embedder, *,
                         threshold: float = 0.62) -> list:
    """Up to `k` facts, in the order given, skipping same-subject repeats.

    Input order is the caller's ranking and is preserved — the best fact is
    always kept, and later ones only lose to something already chosen. If
    skipping leaves fewer than `k`, the block is topped up from what was
    passed over: a half-empty block is worse than a repetitive one.

    Fail-safe: without an embedder, or on any embedding error, this returns
    the first `k` unchanged. Degrading to the old behaviour costs variety;
    raising would cost the turn.
    """
    if not facts or k <= 0:
        return []
    if embedder is None or len(facts) <= 1:
        return facts[:k]
    try:
        vectors = [await embedder.embed(f.content) for f in facts]
    except Exception as e:
        print(f"[facts] diversify unavailable: {e}", flush=True)
        return facts[:k]

    picked: list[int] = []
    for i in range(len(facts)):
        if any(_cosine(vectors[i], vectors[j]) >= threshold for j in picked):
            continue
        picked.append(i)
        if len(picked) == k:
            break
    for i in range(len(facts)):
        if len(picked) == k:
            break
        if i not in picked:
            picked.append(i)
    return [facts[i] for i in sorted(picked)]
