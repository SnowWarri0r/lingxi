"""Abstract outbound channel interface for proactive messaging."""

from __future__ import annotations

from abc import ABC, abstractmethod


class RecipientUnreachable(Exception):
    """The channel says this recipient cannot receive anything, and will not
    until something changes on their side — the bot was removed from the chat,
    they left the organisation, the group was dissolved.

    Distinct from an ordinary send failure because retrying cannot help: a
    caller that retries on a timer spends a composed message every tick on a
    recipient who will never see one. Whoever hears from them again is the
    proof they are reachable.
    """


class OutboundChannel(ABC):
    """Abstract interface for pushing messages to a recipient."""

    @property
    @abstractmethod
    def channel_name(self) -> str:
        """Identifier for this channel type (e.g., 'feishu', 'web', 'cli')."""

    @abstractmethod
    async def send_message(
        self,
        recipient_id: str,
        text: str,
        turn_id: str | None = None,
    ) -> None:
        """Send a proactive message to a specific recipient.

        If `turn_id` is provided, the channel may attach annotation UI
        (👍/👎/✏️) so the user can rate the proactive message.

        Returning means it was delivered. A failure must raise — the caller
        counts a return as a message she sent, remembers having said it, and
        will later tell him how many of them he left unanswered. Raise
        RecipientUnreachable when retrying cannot succeed.
        """

    async def send_sticker(self, recipient_id: str, file_path: str) -> None:
        """Send a 表情包 image, if this channel can.

        Not abstract: a channel with no image support is a channel where the
        sticker is simply skipped, which is better than every such channel
        having to write the same no-op.
        """
        return None


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
