"""Application entry point: CLI-based conversation with a virtual persona."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from lingxi.auth.manager import AuthManager, AuthError
from lingxi.auth.models import AuthConfig, AuthMethod
from lingxi.auth.profile_store import ProfileStore
from lingxi.auth.external_sync import ExternalCredentialSync
from lingxi.conversation.engine import ConversationEngine
from lingxi.memory.manager import MemoryManager
from lingxi.persona.loader import load_persona
from lingxi.planning.planner import Planner
from lingxi.providers.registry import ProviderRegistry
from lingxi.utils.config import load_config, get_nested
from lingxi.utils.logging import setup_logging, get_logger


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


async def _handle_annotation_command(engine, cmd: dict) -> None:
    from lingxi.fewshot.collector import AnnotationCollector
    from lingxi.fewshot.summarizer import AnnotationSummarizer

    last_output = getattr(engine, "_last_output", None)
    last_turn_id = last_output.turn_id if last_output else None
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

    embedder = engine.memory.embedding_provider or (
        engine.fewshot_retriever.embedder if engine.fewshot_retriever else None
    )
    if embedder is None:
        print("[annotate] 没有可用的 embedding provider")
        return
    collector = AnnotationCollector(
        annotation_store=engine.annotation_store,
        fewshot_store=engine.fewshot_store,
        embedder=embedder,
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


def _build_auth_manager(config: dict) -> AuthManager:
    """Build AuthManager with ProfileStore, ExternalSync, and OAuth configs from YAML."""
    oauth_configs = get_nested(config, "auth", "providers", default=None)
    return AuthManager(
        profile_store=ProfileStore(),
        external_sync=ExternalCredentialSync(),
        extra_provider_configs=oauth_configs,
    )


async def create_engine(
    persona_path: str | None = None,
    config_path: str = "config/default.yaml",
    auth_manager: AuthManager | None = None,
) -> ConversationEngine:
    """Create and initialize a conversation engine."""
    config = load_config(config_path)

    # Load persona
    if persona_path is None:
        persona_path = "config/personas/example_persona.yaml"
    persona = load_persona(persona_path)

    # Set up auth and LLM provider
    ProviderRegistry.register_defaults()
    provider_name = get_nested(config, "llm", "provider", default="claude")
    model = get_nested(config, "llm", "model", default="claude-sonnet-4-20250514")

    auth_method_str = get_nested(config, "llm", "auth_method", default="oauth_pkce")
    auth_method = AuthMethod(auth_method_str)

    if auth_manager is None:
        auth_manager = _build_auth_manager(config)

    # Create provider with automatic auth resolution
    llm_provider = await ProviderRegistry.create_llm_with_auth(
        provider_name,
        auth_manager=auth_manager,
        auth_method=auth_method,
        model=model,
    )

    # Set up embedding provider FIRST so we know the dimension
    from lingxi.providers.embedding import create_embedding_provider

    emb_kind = get_nested(config, "embedding", "provider", default="local")
    emb_model = (
        os.environ.get("EMBEDDING_MODEL")
        or get_nested(config, "embedding", "model", default=None)
    )
    emb_device = get_nested(config, "embedding", "device", default="cpu")
    emb_api_key = get_nested(config, "embedding", "api_key", default=None)
    emb_base_url = get_nested(config, "embedding", "base_url", default=None)
    embedding_provider = create_embedding_provider(
        kind=emb_kind,
        model=emb_model,
        llm_provider=llm_provider,
        device=emb_device,
        api_key=emb_api_key,
        base_url=emb_base_url,
    )

    # Probe embedding dim so Chroma collection names are correct from the start
    embedding_dim: int | None = None
    if embedding_provider is not None:
        try:
            sample = await embedding_provider.embed("probe")
            embedding_dim = len(sample)
            print(f"[embedding] detected dimension: {embedding_dim}")
        except Exception as e:
            print(f"[embedding] probe failed, falling back to keyword search: {e}")
            embedding_provider = None

    # Create memory manager with the right embedding_dim
    data_dir = os.environ.get("MEMORY_DATA_DIR", "./data/memory")
    memory_config = config.get("memory", {})
    memory_manager = MemoryManager(
        data_dir=data_dir,
        max_short_term_turns=get_nested(memory_config, "short_term", "max_turns", default=30),
        max_long_term_entries=get_nested(memory_config, "long_term", "max_entries", default=100000),
        max_episodes=get_nested(memory_config, "episodic", "max_episodes", default=500),
        retrieval_top_k=get_nested(memory_config, "long_term", "retrieval_top_k", default=10),
        long_term_backend=get_nested(memory_config, "long_term", "backend", default="chroma"),
        embedding_dim=embedding_dim,
    )

    if embedding_provider is not None:
        memory_manager.set_embedding_provider(embedding_provider)

    # Create planner
    planner = None
    if get_nested(config, "planning", "enabled", default=True):
        planner = Planner(llm_provider, persona)

    # Create interaction tracker (time awareness)
    from lingxi.temporal.tracker import InteractionTracker

    interaction_tracker = InteractionTracker(data_dir)

    # Inner life: life simulator, agenda, subjective layer
    from lingxi.inner_life import (
        AgendaEngine,
        InnerLifeStore,
        SubjectiveLayer,
    )

    inner_life_store = InnerLifeStore(data_dir)
    agenda_engine = AgendaEngine(inner_life_store)
    subjective_layer = SubjectiveLayer(inner_life_store)

    # Fewshot pool: FewShotStore + AnnotationStore + FewShotRetriever
    from lingxi.fewshot.retriever import FewShotRetriever
    from lingxi.fewshot.store import AnnotationStore, FewShotStore

    fewshot_store = None
    annotation_store = None
    fewshot_retriever = None
    if embedding_dim is not None and embedding_dim > 0 and embedding_provider is not None:
        fewshot_dir = Path(data_dir).parent / "fewshot"
        fewshot_dir.mkdir(parents=True, exist_ok=True)
        fewshot_store = FewShotStore(data_dir=fewshot_dir, embedding_dim=embedding_dim)
        await fewshot_store.init()
        annotation_store = AnnotationStore(data_dir=fewshot_dir)
        fewshot_retriever = FewShotRetriever(store=fewshot_store, embedder=embedding_provider)

    # Create engine
    engine = ConversationEngine(
        persona=persona,
        llm_provider=llm_provider,
        memory_manager=memory_manager,
        planner=planner,
        inner_life_store=inner_life_store,
        agenda_engine=agenda_engine,
        subjective_layer=subjective_layer,
        interaction_tracker=interaction_tracker,
        fewshot_store=fewshot_store,
        annotation_store=annotation_store,
        fewshot_retriever=fewshot_retriever,
    )

    # Load persisted state
    await engine.load_state()

    # Bootstrap fewshot seeds (idempotent — skips already-stored samples)
    if fewshot_store is not None:
        added = await engine.bootstrap_fewshot_seeds()
        if added:
            print(f"[fewshot] bootstrapped {added} seeds")

    # Bootstrap biography retriever (embeds all life_events in memory)
    try:
        embedded = await engine.bootstrap_biography()
        if embedded:
            print(f"[biography] embedded {embedded} life events")
    except Exception as e:
        print(f"[biography] bootstrap failed (non-fatal): {e}")

    # Task 18: startup cleanup of stale annotation turns
    if annotation_store is not None:
        deleted = await annotation_store.cleanup(
            max_age_unannotated_days=30,
            max_age_annotated_days=7,
        )
        if deleted:
            print(f"[annotation] cleaned up {deleted} old turn files")

    return engine


async def cmd_login(
    provider: str | None = None,
    use_device_flow: bool = False,
    config_path: str = "config/default.yaml",
) -> None:
    """Interactive login for a provider."""
    config = load_config(config_path)
    auth = _build_auth_manager(config)

    if provider is None:
        available = auth.list_available_providers()
        if available:
            print(f"可用的 OAuth 提供商: {', '.join(available)}")
        else:
            print("未配置任何 OAuth 提供商。")
            print("请在 config/default.yaml 中配置 auth.providers，或使用 API Key。")
            return
        provider = available[0]
        print(f"默认使用: {provider}\n")

    if not auth.has_provider_config(provider):
        # Check env var as fallback
        env_var = AuthManager._env_var_name(provider)
        print(f"未找到 '{provider}' 的 OAuth 配置。")
        print(f"你可以:")
        print(f"  1. 设置环境变量 {env_var}")
        print(f"  2. 在 config/default.yaml 的 auth.providers 中配置 OAuth")
        return

    method = AuthMethod.OAUTH_DEVICE_FLOW if use_device_flow else None

    try:
        await auth.login(provider, method=method)
        print(f"已成功登录 {provider}。")
        print(f"Token 已缓存到 ~/.persona-agent/tokens/")
    except Exception as e:
        print(f"登录失败: {e}")
        sys.exit(1)


async def cmd_logout(provider: str | None = None) -> None:
    """Remove cached credentials."""
    auth = AuthManager()

    if provider is None:
        authenticated = auth.list_authenticated()
        if not authenticated:
            print("没有已登录的提供商。")
            return
        for info in authenticated:
            await auth.logout(info["provider"])
            print(f"已登出 {info['provider']}")
    else:
        await auth.logout(provider)
        print(f"已登出 {provider}。")


async def cmd_auth_status(config_path: str = "config/default.yaml") -> None:
    """Show authentication status."""
    config = load_config(config_path)
    auth = _build_auth_manager(config)
    authenticated = auth.list_authenticated()

    print(f"\n{'='*50}")
    print("  认证状态")
    print(f"{'='*50}")

    # Show configured OAuth providers
    available = auth.list_available_providers()
    if available:
        print(f"\n  可用的 OAuth 提供商: {', '.join(available)}")

    # Show profiles
    if authenticated:
        print(f"\n  凭证概览 ({len(authenticated)} 个):")
        for info in authenticated:
            status_parts = []
            if info["has_valid_credential"]:
                status_parts.append("有效")
            else:
                status_parts.append("无效/已过期")
            if info["cooled_down"]:
                status_parts.append(f"冷却中({info['failures']}次失败)")

            icon = "+" if info["has_valid_credential"] and not info["cooled_down"] else "!"
            source_tag = f" [{info['source']}]" if info["source"] != "manual" else ""
            print(
                f"    [{icon}] {info['provider']}:{info['label']} "
                f"({info['type']}{source_tag}) - {', '.join(status_parts)}"
            )
    else:
        print("\n  无已存储的凭证")

    # Show env vars
    env_keys = {
        "ANTHROPIC_API_KEY": "anthropic",
        "OPENAI_API_KEY": "openai",
    }
    env_found = []
    for env_var, name in env_keys.items():
        if os.environ.get(env_var):
            env_found.append(f"    [+] {name} (via {env_var})")
    if env_found:
        print("\n  环境变量:")
        for line in env_found:
            print(line)

    print()


async def run_cli() -> None:
    """Run the interactive CLI chat loop."""
    logger = get_logger("cli")

    # Parse args
    persona_path = None
    config_path = "config/default.yaml"
    use_device_flow = False

    args = sys.argv[1:]

    # Handle subcommands
    if args and args[0] == "login":
        provider = None
        for a in args[1:]:
            if a == "--device-auth":
                use_device_flow = True
            elif not a.startswith("-"):
                provider = a
        await cmd_login(provider, use_device_flow, config_path)
        return

    if args and args[0] == "logout":
        provider = args[1] if len(args) > 1 else None
        await cmd_logout(provider)
        return

    if args and args[0] == "auth-status":
        await cmd_auth_status(config_path)
        return

    # Parse flags
    i = 0
    while i < len(args):
        if args[i] in ("--persona", "-p") and i + 1 < len(args):
            persona_path = args[i + 1]
            i += 2
        elif args[i] in ("--config", "-c") and i + 1 < len(args):
            config_path = args[i + 1]
            i += 2
        elif args[i] in ("--help", "-h"):
            _print_help()
            return
        else:
            i += 1

    try:
        engine = await create_engine(persona_path, config_path)
    except AuthError as e:
        print(f"\n[认证错误] {e}\n")
        sys.exit(1)

    persona_name = engine.persona.name

    print(f"\n{'='*50}")
    print(f"  {persona_name} 已上线")
    print(f"  /quit    退出")
    print(f"  /stats   查看记忆状态")
    print(f"  /mood    查看当前心情")
    print(f"  /memories <query>  搜索记忆")
    print(f"  /entities  查看实体图谱")
    print(f"  /episodes  查看最近session摘要")
    print(f"{'='*50}\n")

    try:
        while True:
            try:
                user_input = input("你: ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            if user_input == "/quit":
                print(f"\n{persona_name}: 再见，期待下次见面……")
                break

            if user_input == "/stats":
                stats = engine.memory.get_stats()
                print(f"\n[记忆状态] 短期: {stats['short_term_turns']} 轮 | "
                      f"长期: {stats['long_term_entries']} 条 | "
                      f"情景: {stats['episodes']} 段")
                await engine.memory.entity_graph.load()
                ent_stats = engine.memory.entity_graph.stats()
                print(f"[实体图谱] {ent_stats['entity_count']} 个实体, "
                      f"{ent_stats['total_links']} 个事实链接\n")
                continue

            if user_input == "/mood":
                top = engine._emotion_state.top_k(k=5)
                print(f"\n[心情] {engine._current_mood}")
                for name, val in top:
                    bar = "█" * int(val * 20)
                    print(f"   {name:6s} {val:.2f} {bar}")
                print()
                continue

            if user_input.startswith("/memories"):
                query = user_input[len("/memories"):].strip() or "最近"
                ctx = await engine.memory.assemble_context(
                    query, long_term_limit=10, episode_limit=5,
                    recipient_key="cli:local",
                )
                print(f"\n[搜索: \"{query}\"]")
                print(f"长期记忆 ({len(ctx.long_term_facts)}):")
                for f in ctx.long_term_facts:
                    rec = (f.metadata or {}).get("recipient_key", "_global")
                    print(f"  • [{f.importance:.1f}] {f.content[:80]} ({rec})")
                print(f"情景记忆 ({len(ctx.relevant_episodes)}):")
                for ep in ctx.relevant_episodes:
                    print(f"  • [{ep.timestamp.strftime('%m-%d %H:%M')}] {ep.summary[:80]}")
                print()
                continue

            if user_input == "/entities":
                await engine.memory.entity_graph.load()
                ents = engine.memory.entity_graph.all_entities()
                ents.sort(key=lambda e: e.mention_count, reverse=True)
                print(f"\n[实体图谱] 共 {len(ents)} 个")
                for e in ents[:20]:
                    print(f"  • {e.name} ({e.type}) - 提及 {e.mention_count} 次, "
                          f"链接 {len(e.fact_ids)} 条事实")
                print()
                continue

            if user_input == "/episodes":
                eps = await engine.memory.episodic.get_recent(limit=10)
                print(f"\n[最近 {len(eps)} 个session摘要]")
                for ep in eps:
                    print(f"  • [{ep.timestamp.strftime('%Y-%m-%d %H:%M')}] "
                          f"({ep.emotional_tone}) {ep.summary[:100]}")
                print()
                continue

            cmd = parse_annotation_command(user_input)
            if cmd is not None:
                await _handle_annotation_command(engine, cmd)
                continue

            try:
                response = await engine.chat(
                    user_input, channel="cli", recipient_id="local"
                )
                print(f"\n{persona_name}: {response}\n")
                # After printing Aria's reply, append a small turn_id footer
                if hasattr(engine, "_last_output") and engine._last_output and engine._last_output.turn_id:
                    print(f"\033[2m[{engine._last_output.turn_id[:8]}]\033[0m")
            except Exception as e:
                logger.error("chat_error", error=str(e))
                print(f"\n[错误] {e}\n")

    finally:
        print("\n正在保存对话记忆……")
        result = await engine.end_session(channel="cli", recipient_id="local")
        print(f"已保存。提取了 {result.get('facts_stored', 0)} 条记忆。")


def _print_help() -> None:
    print("""
persona-agent - 虚拟人格对话代理

用法:
  persona-agent                              启动对话 (使用默认人设)
  persona-agent -p <persona.yaml>            使用指定人设
  persona-agent -c <config.yaml>             使用指定配置
  persona-agent login [provider]             浏览器 OAuth 登录 (PKCE)
  persona-agent login [provider] --device-auth  设备码登录 (适合远程/无头环境)
  persona-agent logout [provider]            登出并清除缓存 token
  persona-agent auth-status                  查看认证状态

认证方式 (按优先级):
  1. 缓存的 OAuth Token     persona-agent login openai
  2. 环境变量                ANTHROPIC_API_KEY=sk-ant-xxx
  3. 配置文件中的 api_key

配置 OAuth 提供商 (config/default.yaml):
  auth:
    providers:
      my_provider:
        client_id: "your-app-id"
        auth_url: "https://auth.example.com/authorize"
        token_url: "https://auth.example.com/token"
        device_auth_url: "https://auth.example.com/device/code"  # 可选
        scopes: ["openid", "offline_access"]

对话中命令:
  /quit    退出对话
  /stats   查看记忆状态
  /mood    查看当前心情
""")


def main() -> None:
    """Entry point."""
    setup_logging()
    asyncio.run(run_cli())


if __name__ == "__main__":
    main()
