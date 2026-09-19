"""CLI entry point for the Feishu bot."""

from __future__ import annotations

import asyncio
import sys


def load_proactive_config(cfg: dict):
    """Build ProactiveConfig from the `proactive:` block of the YAML.

    Driven by the model's own fields rather than a hand-written list of keys.
    The list had gone stale: reengage_backoff and reengage_max_hours were
    never on it, so setting either in config/default.yaml did nothing and said
    nothing — the code default applied while the file claimed otherwise.
    """
    from lingxi.temporal.proactive import ProactiveConfig

    block = (cfg or {}).get("proactive") or {}
    kwargs = {k: v for k, v in block.items()
              if k in ProactiveConfig.model_fields}
    thresholds = kwargs.get("silence_thresholds")
    if isinstance(thresholds, dict):
        kwargs["silence_thresholds"] = {
            int(k): int(v) for k, v in thresholds.items()}
    return ProactiveConfig(**kwargs)


def main() -> None:
    """Start the Feishu bot with WebSocket long connection."""
    import os

    # Load .env before checking env vars so local dev doesn't need export
    from dotenv import load_dotenv
    load_dotenv()

    if not os.environ.get("FEISHU_APP_ID"):
        print("需要设置环境变量（在 .env 文件里或 shell export）:")
        print("  FEISHU_APP_ID=cli_xxxxx")
        print("  FEISHU_APP_SECRET=xxxxx")
        sys.exit(1)

    from lingxi.utils.logging import setup_logging

    setup_logging()

    # Parse args
    persona_path = None
    config_path = "config/default.yaml"
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] in ("--persona", "-p") and i + 1 < len(args):
            persona_path = args[i + 1]
            i += 2
        elif args[i] in ("--config", "-c") and i + 1 < len(args):
            config_path = args[i + 1]
            i += 2
        else:
            i += 1

    # Step 1: Create engine (async) - run to completion first
    from lingxi.app import create_engine
    from lingxi.utils.config import load_config, get_nested

    engine = asyncio.run(create_engine(persona_path=persona_path, config_path=config_path))

    # Load proactive config
    cfg = load_config(config_path)

    proactive_cfg = load_proactive_config(cfg)

    # Optional: desktop pet state endpoint. Runs in daemon thread on
    # localhost so the pet process can poll Aria's current state.
    pet_enabled = get_nested(cfg, "pet", "enabled", default=True)
    pet_port = int(get_nested(cfg, "pet", "port", default=7891))
    if pet_enabled:
        try:
            from lingxi.pet.state_endpoint import start_pet_endpoint_in_thread
            start_pet_endpoint_in_thread(engine, port=pet_port)
            print(f"[pet] state endpoint on http://127.0.0.1:{pet_port}/pet/state")
        except Exception as e:
            print(f"[pet] failed to start endpoint (non-fatal): {e}")

    # Step 2: Start bot (blocking, SDK manages its own event loop)
    from lingxi.channels.feishu import FeishuBot

    bot = FeishuBot(engine=engine, proactive_config=proactive_cfg)
    bot.start()


if __name__ == "__main__":
    main()
