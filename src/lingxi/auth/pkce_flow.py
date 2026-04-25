"""OAuth 2.0 Authorization Code Flow with PKCE (RFC 7636).

This is the primary flow used by OpenAI Codex CLI:
1. CLI starts a local HTTP server on a callback port
2. Opens browser to the authorization URL with a PKCE challenge
3. User authenticates in browser
4. Browser redirects to localhost with an authorization code
5. CLI exchanges the code + verifier for tokens
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import secrets
import urllib.parse
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread
from datetime import datetime, timedelta

import httpx

from lingxi.auth.models import AuthConfig, OAuthProviderConfig, TokenInfo


class PKCEFlowError(Exception):
    """Errors during PKCE flow authentication."""


def _generate_code_verifier() -> str:
    """Generate a cryptographically random code verifier (43-128 chars, base64url)."""
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode("ascii")


def _generate_code_challenge(verifier: str) -> str:
    """Generate S256 code challenge from verifier."""
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


class _CallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler that captures the OAuth callback."""

    auth_code: str | None = None
    error: str | None = None

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)

        if "code" in params:
            _CallbackHandler.auth_code = params["code"][0]
            self._respond(
                200,
                "<html><body><h2>登录成功！</h2>"
                "<p>你可以关闭此页面，回到终端继续操作。</p></body></html>",
            )
        elif "error" in params:
            _CallbackHandler.error = params.get("error_description", params["error"])[0]
            self._respond(
                400,
                f"<html><body><h2>登录失败</h2><p>{_CallbackHandler.error}</p></body></html>",
            )
        else:
            self._respond(400, "<html><body><h2>未知回调</h2></body></html>")

    def _respond(self, status: int, body: str):
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def log_message(self, format, *args):
        pass  # Suppress request logs


class PKCEFlowAuth:
    """Handles OAuth 2.0 Authorization Code Flow with PKCE."""

    def __init__(
        self,
        client_id: str,
        auth_url: str,
        token_url: str,
        scopes: list[str] | None = None,
        client_secret: str | None = None,
        callback_port: int = 1455,
        extra_auth_params: dict[str, str] | None = None,
    ):
        self.client_id = client_id
        self.auth_url = auth_url
        self.token_url = token_url
        self.scopes = scopes or []
        self.client_secret = client_secret
        self.callback_port = callback_port
        self.extra_auth_params = extra_auth_params or {}

    @property
    def redirect_uri(self) -> str:
        return f"http://localhost:{self.callback_port}/auth/callback"

    async def login_interactive(self) -> TokenInfo:
        """Run the full interactive PKCE login flow."""
        # Generate PKCE pair
        code_verifier = _generate_code_verifier()
        code_challenge = _generate_code_challenge(code_verifier)

        # Build authorization URL
        state = secrets.token_urlsafe(16)
        params: dict[str, str] = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "state": state,
        }
        if self.scopes:
            params["scope"] = " ".join(self.scopes)
        params.update(self.extra_auth_params)

        authorize_url = f"{self.auth_url}?{urllib.parse.urlencode(params)}"

        # Reset handler state
        _CallbackHandler.auth_code = None
        _CallbackHandler.error = None

        # Start local callback server
        server = HTTPServer(("127.0.0.1", self.callback_port), _CallbackHandler)
        server_thread = Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        try:
            print(f"\n{'='*50}")
            print(f"  正在打开浏览器进行登录...")
            print(f"  如果浏览器没有自动打开，请手动访问：")
            print(f"  {authorize_url[:80]}...")
            print(f"{'='*50}\n")

            webbrowser.open(authorize_url)

            # Wait for callback (timeout 5 minutes)
            for _ in range(300):
                if _CallbackHandler.auth_code or _CallbackHandler.error:
                    break
                await asyncio.sleep(1)

            if _CallbackHandler.error:
                raise PKCEFlowError(f"Authorization failed: {_CallbackHandler.error}")

            if not _CallbackHandler.auth_code:
                raise PKCEFlowError("Authorization timed out (5 minutes).")

            # Exchange code for tokens
            token = await self._exchange_code(
                code=_CallbackHandler.auth_code,
                code_verifier=code_verifier,
            )

            print("  ✓ 登录成功！\n")
            return token

        finally:
            server.shutdown()

    async def _exchange_code(self, code: str, code_verifier: str) -> TokenInfo:
        """Exchange authorization code for tokens."""
        data: dict[str, str] = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.redirect_uri,
            "client_id": self.client_id,
            "code_verifier": code_verifier,
        }
        if self.client_secret:
            data["client_secret"] = self.client_secret

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.token_url,
                data=data,
                headers={"Accept": "application/json"},
            )

            if response.status_code != 200:
                raise PKCEFlowError(
                    f"Token exchange failed: {response.status_code} {response.text}"
                )

            body = response.json()
            expires_at = None
            if "expires_in" in body:
                expires_at = datetime.now() + timedelta(seconds=body["expires_in"])

            return TokenInfo(
                access_token=body["access_token"],
                refresh_token=body.get("refresh_token"),
                token_type=body.get("token_type", "Bearer"),
                expires_at=expires_at,
                scope=body.get("scope"),
                id_token=body.get("id_token"),
            )


def create_pkce_flow_from_config(config: AuthConfig) -> PKCEFlowAuth:
    """Create a PKCEFlowAuth instance from an AuthConfig."""
    if not config.client_id or not config.auth_url or not config.token_url:
        raise PKCEFlowError(
            f"PKCE OAuth not configured for '{config.provider}'. "
            f"Need client_id, auth_url, and token_url in config."
        )
    return PKCEFlowAuth(
        client_id=config.client_id,
        auth_url=config.auth_url,
        token_url=config.token_url,
        scopes=config.scopes,
        client_secret=config.client_secret,
        callback_port=config.callback_port,
        extra_auth_params=config.extra_auth_params,
    )
