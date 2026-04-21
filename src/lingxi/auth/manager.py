"""Authentication manager: unified entry point for all auth methods.

Supports profile-based multi-account auth, external CLI credential sync,
failure cooldown, and SecretRef resolution. Inspired by OpenClaw's architecture.
"""

from __future__ import annotations

import os

from lingxi.auth.device_flow import DeviceFlowAuth, DeviceFlowError, refresh_token
from lingxi.auth.external_sync import ExternalCredentialSync
from lingxi.auth.models import (
    AuthConfig,
    AuthMethod,
    AuthProfile,
    CredentialType,
    OAuthProviderConfig,
    PROVIDER_OAUTH_CONFIGS,
    TokenInfo,
)
from lingxi.auth.pkce_flow import PKCEFlowAuth, PKCEFlowError
from lingxi.auth.profile_store import ProfileStore
from lingxi.auth.token_store import TokenStore


class AuthManager:
    """Manages authentication across providers with multiple auth methods.

    Resolution order:
    1. Stored profiles (ordered: oauth > token > api_key, skipping cooled-down)
    2. External CLI credentials (Claude Code, Codex CLI)
    3. Environment variable API key
    4. Config-specified API key
    5. Interactive OAuth login (PKCE or Device Code)
    """

    def __init__(
        self,
        profile_store: ProfileStore | None = None,
        token_store: TokenStore | None = None,  # deprecated, kept for test compat
        external_sync: ExternalCredentialSync | None = None,
        auto_sync_external: bool = True,
        extra_provider_configs: dict[str, dict] | None = None,
    ):
        self._profile_store = profile_store or ProfileStore()
        self._external_sync = external_sync or ExternalCredentialSync()
        self._provider_configs: dict[str, OAuthProviderConfig] = dict(PROVIDER_OAUTH_CONFIGS)
        if extra_provider_configs:
            for name, cfg in extra_provider_configs.items():
                self._provider_configs[name] = OAuthProviderConfig.model_validate(cfg)

        if auto_sync_external:
            self._sync_external_credentials()

    def _sync_external_credentials(self) -> None:
        """Import credentials from Claude Code and Codex CLI."""
        external_profiles = self._external_sync.sync_all()
        for profile in external_profiles:
            existing = self._profile_store.get(profile.provider, profile.label)
            # Only overwrite if same source or no existing profile
            if existing is None or existing.source == profile.source:
                self._profile_store.upsert(profile)

    async def resolve_credentials(self, config: AuthConfig) -> str:
        """Resolve credentials for a provider. Returns an API key or access token."""
        provider = config.provider

        # 1. Try stored profiles (ordered, with cooldown awareness)
        profiles = self._profile_store.get_ordered_for_provider(provider)

        # If targeting a specific label, filter
        if config.profile_label:
            profiles = [p for p in profiles if p.label == config.profile_label]

        for profile in profiles:
            credential = await self._try_resolve_profile(profile, config)
            if credential is not None:
                self._profile_store.reset_cooldown(provider, profile.label)
                return credential

        # 2. Check environment variable
        env_key = self._get_env_key(config)
        if env_key:
            return env_key

        # 3. Check config-specified API key
        if config.api_key:
            return config.api_key

        # 4. Interactive OAuth login (if configured)
        if config.method in (AuthMethod.OAUTH_DEVICE_FLOW, AuthMethod.OAUTH_PKCE):
            merged = self._merge_with_provider_config(config)
            if merged.client_id and merged.auth_url and merged.token_url:
                token = await self._interactive_login(merged)
                # Save as a new profile
                self._profile_store.upsert(AuthProfile(
                    credential_type=CredentialType.OAUTH,
                    provider=provider,
                    label="default",
                    oauth_token=token,
                    source="manual",
                ))
                return token.access_token

        raise AuthError(
            f"未找到 '{provider}' 的凭证。\n"
            f"  方式1: 设置环境变量 {self._env_var_name(provider)}\n"
            f"  方式2: 运行 persona-agent login {provider}\n"
            f"  方式3: 在 config/default.yaml 的 auth.providers 中配置 OAuth\n"
            f"  方式4: 已安装 Claude Code / Codex CLI? 先登录那些工具，凭证会自动同步"
        )

    async def _try_resolve_profile(
        self, profile: AuthProfile, config: AuthConfig
    ) -> str | None:
        """Try to resolve a credential from a profile."""
        if profile.cooldown.is_cooled_down:
            return None

        if profile.credential_type in (CredentialType.API_KEY, CredentialType.TOKEN):
            return profile.resolve_secret()

        elif profile.credential_type == CredentialType.OAUTH:
            # Try current token
            val = profile.resolve_secret()
            if val:
                return val

            # Try refresh
            if profile.oauth_token and profile.oauth_token.refresh_token:
                refreshed = await self._try_refresh_profile(profile, config)
                if refreshed:
                    return refreshed

        return None

    async def _try_refresh_profile(
        self, profile: AuthProfile, config: AuthConfig
    ) -> str | None:
        """Try to refresh an expired OAuth token on a profile."""
        if not profile.oauth_token or not profile.oauth_token.refresh_token:
            return None

        merged = self._merge_with_provider_config(config)
        if not merged.token_url or not merged.client_id:
            return None

        try:
            new_token = await refresh_token(
                token_url=merged.token_url,
                client_id=merged.client_id,
                refresh_token=profile.oauth_token.refresh_token,
                client_secret=merged.client_secret,
            )
            profile.oauth_token = new_token
            self._profile_store.upsert(profile)
            return new_token.access_token
        except DeviceFlowError:
            return None

    async def login(
        self,
        provider: str,
        config: AuthConfig | None = None,
        method: AuthMethod | None = None,
    ) -> TokenInfo:
        """Interactive login for a provider."""
        if config is None:
            config = self._build_auth_config(provider, method)
        merged = self._merge_with_provider_config(config)
        token = await self._interactive_login(merged)

        # Save as profile
        self._profile_store.upsert(AuthProfile(
            credential_type=CredentialType.OAUTH,
            provider=provider,
            label="default",
            oauth_token=token,
            source="manual",
        ))
        return token

    async def logout(self, provider: str, label: str | None = None) -> None:
        """Remove cached credentials for a provider."""
        if label:
            self._profile_store.delete(provider, label)
        else:
            # Delete all profiles for this provider
            for p in self._profile_store.list_profiles(provider):
                self._profile_store.delete(provider, p.label)

    def get_cached_token(self, provider: str) -> TokenInfo | None:
        """Check if a valid cached OAuth token exists."""
        for p in self._profile_store.list_profiles(provider):
            if p.credential_type == CredentialType.OAUTH and p.oauth_token:
                if not p.oauth_token.is_expired:
                    return p.oauth_token
        return None

    def list_authenticated(self) -> list[dict]:
        """List all providers with stored profiles."""
        result = []
        seen_providers: set[str] = set()
        for profile in self._profile_store.list_profiles():
            provider = profile.provider
            if provider not in seen_providers:
                seen_providers.add(provider)
            has_valid = profile.resolve_secret() is not None
            result.append({
                "provider": profile.provider,
                "label": profile.label,
                "type": profile.credential_type.value,
                "source": profile.source,
                "has_valid_credential": has_valid,
                "cooled_down": profile.cooldown.is_cooled_down,
                "failures": profile.cooldown.consecutive_failures,
            })
        return result

    def has_provider_config(self, provider: str) -> bool:
        return provider in self._provider_configs

    def list_available_providers(self) -> list[str]:
        return list(self._provider_configs.keys())

    def record_auth_failure(self, provider: str, label: str = "default") -> None:
        """Record an auth failure for cooldown tracking."""
        self._profile_store.record_failure(provider, label)

    def reset_auth_cooldown(self, provider: str, label: str = "default") -> None:
        """Reset cooldown after successful use."""
        self._profile_store.reset_cooldown(provider, label)

    # --- Internal (unchanged from before) ---

    async def _interactive_login(self, config: AuthConfig) -> TokenInfo:
        if not config.client_id or not config.token_url:
            raise AuthError(
                f"OAuth 未配置提供商 '{config.provider}'。\n"
                f"请在 config/default.yaml 的 auth.providers 中配置 client_id, auth_url, token_url。"
            )

        if config.method == AuthMethod.OAUTH_DEVICE_FLOW:
            if not config.device_auth_url:
                raise AuthError(
                    f"Device Code 流程需要 device_auth_url，'{config.provider}' 未配置。"
                )
            return await self._run_device_flow(config)
        else:
            if not config.auth_url:
                raise AuthError(
                    f"PKCE 流程需要 auth_url，'{config.provider}' 未配置。"
                )
            return await self._run_pkce_flow(config)

    async def _run_pkce_flow(self, config: AuthConfig) -> TokenInfo:
        flow = PKCEFlowAuth(
            client_id=config.client_id,
            auth_url=config.auth_url,
            token_url=config.token_url,
            scopes=config.scopes,
            client_secret=config.client_secret,
            callback_port=config.callback_port,
            extra_auth_params=config.extra_auth_params,
        )
        return await flow.login_interactive()

    async def _run_device_flow(self, config: AuthConfig) -> TokenInfo:
        flow = DeviceFlowAuth(
            client_id=config.client_id,
            device_auth_url=config.device_auth_url,
            token_url=config.token_url,
            scopes=config.scopes,
            client_secret=config.client_secret,
        )
        return await flow.login_interactive()

    def _merge_with_provider_config(self, config: AuthConfig) -> AuthConfig:
        known = self._provider_configs.get(config.provider)
        if not known:
            return config
        data = config.model_dump()
        known_data = known.model_dump()
        for key, value in known_data.items():
            if key in data and data[key] is None:
                data[key] = value
            elif key in data and isinstance(data[key], list) and not data[key]:
                data[key] = value
            elif key in data and isinstance(data[key], dict) and not data[key]:
                data[key] = value
        return AuthConfig.model_validate(data)

    def _build_auth_config(self, provider: str, method: AuthMethod | None = None) -> AuthConfig:
        if method is None:
            known = self._provider_configs.get(provider)
            if known and known.auth_url:
                method = AuthMethod.OAUTH_PKCE
            elif known and known.device_auth_url:
                method = AuthMethod.OAUTH_DEVICE_FLOW
            else:
                method = AuthMethod.OAUTH_PKCE
        return AuthConfig(provider=provider, method=method)

    @staticmethod
    def _get_env_key(config: AuthConfig) -> str | None:
        if config.api_key_env_var:
            val = os.environ.get(config.api_key_env_var)
            if val:
                return val
        env_name = AuthManager._env_var_name(config.provider)
        return os.environ.get(env_name)

    @staticmethod
    def _env_var_name(provider: str) -> str:
        mapping = {
            "anthropic": "ANTHROPIC_API_KEY",
            "claude": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
        }
        return mapping.get(provider, f"{provider.upper()}_API_KEY")


class AuthError(Exception):
    """Authentication error."""
