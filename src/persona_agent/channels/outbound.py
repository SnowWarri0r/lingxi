"""Abstract outbound channel interface for proactive messaging."""

from __future__ import annotations

from abc import ABC, abstractmethod


class OutboundChannel(ABC):
    """Abstract interface for pushing messages to a recipient."""

    @property
    @abstractmethod
    def channel_name(self) -> str:
        """Identifier for this channel type (e.g., 'feishu', 'web', 'cli')."""

    @abstractmethod
    async def send_message(self, recipient_id: str, text: str) -> None:
        """Send a proactive message to a specific recipient."""


class ChannelRegistry:
    """Maps channel names to OutboundChannel instances."""

    def __init__(self) -> None:
        self._channels: dict[str, OutboundChannel] = {}

    def register(self, channel: OutboundChannel) -> None:
        self._channels[channel.channel_name] = channel

    def unregister(self, name: str) -> None:
        self._channels.pop(name, None)

    def get(self, name: str) -> OutboundChannel | None:
        return self._channels.get(name)

    def all_channels(self) -> list[OutboundChannel]:
        return list(self._channels.values())

    def __contains__(self, name: str) -> bool:
        return name in self._channels
