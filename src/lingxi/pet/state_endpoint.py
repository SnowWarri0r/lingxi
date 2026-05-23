"""HTTP endpoint exposing Aria's current state for the desktop pet.

The pet runs as a separate process and polls `/pet/state` every few
seconds. This keeps pet crashes from affecting the IM agent — and lets
multiple pet UIs (e.g. menubar + sprite window) share one state source.

The endpoint runs in a daemon thread inside the feishu agent so we don't
need a separate service. Read-only — no mutation of agent state.
"""

from __future__ import annotations

import threading
from datetime import datetime

from fastapi import FastAPI
import uvicorn

from lingxi.inner_life.models import derive_engagement_mode
from lingxi.persona.models import EmotionState
from lingxi.pet.sprite_mapper import pick_sprite


def classify_emotion_family(emotion: EmotionState | None) -> str:
    """Return the dominant emotion family name (or 'NEUTRAL').

    Picks the family whose top dimension is highest, with a threshold so
    minor baseline values don't flip the pet around. Mirrors the priority
    logic used by derive_engagement_mode but without the gating thresholds —
    we want the *visible* family even if it's not strong enough to change
    engagement mode.
    """
    if emotion is None:
        return "NEUTRAL"

    families = {
        "FLUSTERED": EmotionState.FLUSTERED_DIMS,
        "HEAVY": EmotionState.HEAVY_DIMS,
        "PROVOKED": EmotionState.PROVOKED_DIMS,
        "HIGH_ENERGY": EmotionState.HIGH_ENERGY_DIMS,
        "LOW_ENERGY": EmotionState.LOW_ENERGY_DIMS,
    }

    best_family = "NEUTRAL"
    # 0.25 is below typical baseline (平静 0.5) for any single dim — using
    # the relative max instead means baseline 平静=0.5 would always trigger
    # something. So gate on a meaningful family activation.
    best_score = 0.3
    for family, dims in families.items():
        score = max(
            (v for n, v in emotion.dimensions.items() if n in dims),
            default=0.0,
        )
        if score > best_score:
            best_score = score
            best_family = family
    return best_family


def build_pet_state_app(engine) -> FastAPI:
    """FastAPI app exposing /pet/state and /pet/health.

    `engine` is the running ConversationEngine — we read `_emotion_state`
    and `inner_life_store` from it. No writes.
    """
    app = FastAPI(title="Lingxi Pet State", version="0.1.0")

    @app.get("/pet/health")
    def health():
        return {"status": "ok"}

    @app.get("/pet/state")
    async def state():
        emotion = engine._emotion_state
        inner_state = None
        if engine.inner_life_store is not None:
            try:
                inner_state = await engine.inner_life_store.load_state()
            except Exception:
                inner_state = None

        family = classify_emotion_family(emotion)
        mode = derive_engagement_mode(inner_state, emotion).value

        activity = inner_state.current_activity if inner_state else None
        activity_kind = activity.kind.value if activity else None
        activity_name = activity.name if activity else None

        sprite = pick_sprite(
            engagement_mode=mode,
            emotion_family=family,
            activity_kind=activity_kind,
            hour=datetime.now().hour,
        )

        return {
            "sprite": sprite,
            "engagement_mode": mode,
            "emotion_family": family,
            "activity_kind": activity_kind,
            "activity_name": activity_name,
            "mood_narrative": engine._current_mood,
            "ts": datetime.now().isoformat(),
        }

    return app


def start_pet_endpoint_in_thread(
    engine, *, host: str = "127.0.0.1", port: int = 7891
) -> threading.Thread:
    """Start the pet state endpoint in a daemon thread.

    Daemon = dies with the parent process (feishu agent). Single-threaded
    uvicorn is plenty — pet polls maybe once every 2-5s.
    """
    app = build_pet_state_app(engine)

    def run():
        uvicorn.run(app, host=host, port=port, log_level="warning")

    t = threading.Thread(target=run, daemon=True, name="pet-state-endpoint")
    t.start()
    return t
