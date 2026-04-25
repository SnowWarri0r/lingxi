"""Chinese time/date formatting helpers."""

from __future__ import annotations

from datetime import datetime, timedelta

_WEEKDAYS = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]


def weekday_cn(dt: datetime) -> str:
    return _WEEKDAYS[dt.weekday()]


def format_timedelta_cn(delta: timedelta) -> str:
    """Format a timedelta as natural Chinese, e.g., '3天2小时'."""
    total_seconds = int(delta.total_seconds())
    if total_seconds < 0:
        return "0秒"

    days = total_seconds // 86400
    hours = (total_seconds % 86400) // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    parts: list[str] = []
    if days > 0:
        parts.append(f"{days}天")
    if hours > 0:
        parts.append(f"{hours}小时")
    if minutes > 0 and days == 0:  # Don't show minutes if we have days
        parts.append(f"{minutes}分钟")
    if not parts:
        parts.append(f"{seconds}秒")

    return "".join(parts)


def format_datetime_cn(dt: datetime) -> str:
    """Format a datetime as '2026-04-13 22:30（星期一）'."""
    return f"{dt.strftime('%Y-%m-%d %H:%M')}（{weekday_cn(dt)}）"
