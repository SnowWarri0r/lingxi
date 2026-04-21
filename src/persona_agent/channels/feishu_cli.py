"""CLI entry point for the Feishu bot."""

from __future__ import annotations

import asyncio
import sys


def main() -> None:
    """Start the Feishu bot with WebSocket long connection."""
    import os

    if not os.environ.get("FEISHU_APP_ID"):
        print("需要设置环境变量:")
        print("  export FEISHU_APP_ID=cli_xxxxx")
        print("  export FEISHU_APP_SECRET=xxxxx")
        sys.exit(1)

    from persona_agent.utils.logging import setup_logging

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
    from persona_agent.app import create_engine
    from persona_agent.utils.config import load_config, get_nested

    engine = asyncio.run(create_engine(persona_path=persona_path, config_path=config_path))

    # Load proactive config
    cfg = load_config(config_path)
    from persona_agent.temporal.proactive import ProactiveConfig

    proactive_cfg = ProactiveConfig(
        enabled=get_nested(cfg, "proactive", "enabled", default=True),
        check_interval_minutes=get_nested(cfg, "proactive", "check_interval_minutes", default=5),
        silence_thresholds={
            int(k): int(v)
            for k, v in get_nested(cfg, "proactive", "silence_thresholds", default={1: 72, 2: 24, 3: 6, 4: 3}).items()
        },
        cooldown_hours=get_nested(cfg, "proactive", "cooldown_hours", default=12.0),
        quiet_hours_start=get_nested(cfg, "proactive", "quiet_hours_start", default=23),
        quiet_hours_end=get_nested(cfg, "proactive", "quiet_hours_end", default=8),
    )

    # Step 2: Start bot (blocking, SDK manages its own event loop)
    from persona_agent.channels.feishu import FeishuBot

    bot = FeishuBot(engine=engine, proactive_config=proactive_cfg)
    bot.start()


if __name__ == "__main__":
    main()
