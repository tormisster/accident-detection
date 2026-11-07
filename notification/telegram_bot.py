"""Asynchronous Telegram Bot notification module for real-time accident alerts.

Supports automatic multi-recipient & group broadcasting:
Anyone who opens https://t.me/Accident_accident_bot and presses 'START', or any
Telegram Group/Channel where the bot is added, will automatically receive alerts.
"""

from __future__ import annotations

from datetime import datetime
import html
import logging
import threading
from typing import List, Optional, Set, Tuple

import cv2
import numpy as np
import requests

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Dispatches non-blocking Telegram photo alerts to all subscribed chats/groups."""

    DEFAULT_BOT_TOKEN: str = "9233730263:AAEkjMzC-pK2PO7ekTGgN4rmVYtqK3hHqHo"
    DEFAULT_CHAT_IDS: Tuple[str, ...] = ("6819938549", "9191991912", "-5505596792")

    def __init__(
        self,
        bot_token: str,
        chat_id: str | List[str],
        enabled: bool = True,
        timeout_seconds: float = 10.0,
        auto_discover_subscribers: bool = True,
    ) -> None:
        """Initialize the TelegramNotifier."""
        clean_token = bot_token.strip()
        if not clean_token or clean_token == "YOUR_TELEGRAM_BOT_TOKEN":
            clean_token = self.DEFAULT_BOT_TOKEN
        self.bot_token = clean_token
        self.enabled = enabled
        self.timeout_seconds = float(timeout_seconds)
        self.auto_discover_subscribers = auto_discover_subscribers

        self.subscribers: Set[str] = set(self.DEFAULT_CHAT_IDS)
        if isinstance(chat_id, (list, tuple, set)):
            for cid in chat_id:
                s = str(cid).strip()
                if s and s != "YOUR_TELEGRAM_CHAT_ID":
                    self.subscribers.add(s)
        else:
            for part in str(chat_id).split(","):
                s = part.strip()
                if s and s != "YOUR_TELEGRAM_CHAT_ID":
                    self.subscribers.add(s)

        self.send_photo_url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
        self.get_updates_url = f"https://api.telegram.org/bot{self.bot_token}/getUpdates"
        self._active_threads: List[threading.Thread] = []

    @property
    def is_configured(self) -> bool:
        """Return True if notifications are enabled and valid credentials are set."""
        return (
            self.enabled
            and bool(self.bot_token)
            and self.bot_token != "YOUR_TELEGRAM_BOT_TOKEN"
            and (bool(self.subscribers) or self.auto_discover_subscribers)
        )

    def refresh_subscribers(self) -> Set[str]:
        """Discover any new users or groups that messaged/started the bot via getUpdates."""
        if not self.auto_discover_subscribers or not self.bot_token:
            return self.subscribers
        try:
            resp = requests.get(self.get_updates_url, timeout=5.0)
            data = resp.json()
            if data.get("ok") and data.get("result"):
                for update in data["result"]:
                    msg = (
                        update.get("message")
                        or update.get("edited_message")
                        or update.get("channel_post")
                        or update.get("my_chat_member", {})
                    )
                    chat = msg.get("chat")
                    if chat and "id" in chat:
                        self.subscribers.add(str(chat["id"]))
        except requests.RequestException:
            pass
        return self.subscribers

    @staticmethod
    def format_html_caption(
        timestamp: str,
        num_motorcycles: int,
        num_persons: int,
        max_confidence: float,
        low_light_active: bool = False,
        location_tag: Optional[str] = None,
    ) -> str:
        """Format the alert metadata into an HTML caption for Telegram."""
        safe_ts = html.escape(timestamp)
        lines = [
            "🚨 <b>MOTORCYCLE ACCIDENT DETECTED</b> 🚨",
            "",
            f"🕒 <b>Timestamp:</b> <code>{safe_ts}</code>",
            f"🏍️ <b>Accident Motorcycles:</b> <code>{num_motorcycles}</code>",
            f"🤕 <b>Injured Persons:</b> <code>{num_persons}</code>",
            f"🎯 <b>Confidence Score:</b> <code>{max_confidence * 100:.1f}% ({max_confidence:.3f})</code>",
            f"🌙 <b>Low-Light Enhancement:</b> <code>{'Active' if low_light_active else 'Bypassed'}</code>",
        ]
        if location_tag:
            lines.append(f"📍 <b>Source:</b> <code>{html.escape(location_tag)}</code>")
        return "\n".join(lines)

    def _send_photo_worker(
        self,
        image_bytes: bytes,
        caption: str,
    ) -> None:
        """Worker executed in a background thread to broadcast the alert to all subscribers."""
        if not self.is_configured:
            logger.warning(
                "TelegramNotifier triggered, but credentials are not configured. "
                "Skipping HTTP call."
            )
            return

        target_chat_ids = set(self.refresh_subscribers())
        if not target_chat_ids:
            logger.warning("No Telegram subscribers found yet. Press START in the bot first.")
            return

        for cid in target_chat_ids:
            files = {
                "photo": ("accident_snapshot.jpg", image_bytes, "image/jpeg"),
            }
            data = {
                "chat_id": cid,
                "caption": caption,
                "parse_mode": "HTML",
            }
            try:
                response = requests.post(
                    self.send_photo_url,
                    data=data,
                    files=files,
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                logger.info(
                    "Telegram alert dispatched to chat_id=%s (HTTP %d).",
                    cid,
                    response.status_code,
                )
            except requests.RequestException as exc:
                logger.error("Failed to send Telegram alert to chat_id=%s: %s", cid, exc)

    def send_accident_alert(
        self,
        frame_bgr: np.ndarray,
        num_motorcycles: int,
        num_persons: int,
        max_confidence: float,
        timestamp: Optional[str] = None,
        low_light_active: bool = False,
        location_tag: Optional[str] = None,
    ) -> Optional[threading.Thread]:
        """Encode the annotated frame and asynchronously send a Telegram photo alert."""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        success, encoded_img = cv2.imencode(
            ".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92]
        )
        if not success:
            logger.error("Failed to JPEG-encode frame for Telegram notification.")
            return None

        caption = self.format_html_caption(
            timestamp=timestamp,
            num_motorcycles=num_motorcycles,
            num_persons=num_persons,
            max_confidence=max_confidence,
            low_light_active=low_light_active,
            location_tag=location_tag,
        )

        worker_thread = threading.Thread(
            target=self._send_photo_worker,
            args=(encoded_img.tobytes(), caption),
            daemon=True,
            name="TelegramNotifierWorker",
        )
        worker_thread.start()
        self._active_threads = [t for t in self._active_threads if t.is_alive()]
        self._active_threads.append(worker_thread)
        return worker_thread

    def flush(self, timeout: float = 10.0) -> None:
        """Wait for any in-flight asynchronous Telegram alert threads to finish."""
        for t in list(self._active_threads):
            if t.is_alive():
                t.join(timeout=timeout)
        self._active_threads.clear()
