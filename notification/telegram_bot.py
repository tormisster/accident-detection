"""Asynchronous Telegram Bot notification module for real-time accident alerts.

Supports automatic multi-recipient & group broadcasting in parallel:
Anyone who opens https://t.me/Accident_accident_bot and presses 'START', or any
Telegram Group/Channel where the bot is added, will automatically receive alerts.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import html
import logging
from pathlib import Path
import threading
from typing import List, Optional, Set, Tuple

import cv2
import numpy as np
import requests

logger = logging.getLogger(__name__)

THAILAND_TZ = timezone(timedelta(hours=7))


def now_thailand_str() -> str:
    """Return current date and time in Thailand timezone (UTC+7) formatted as YYYY-MM-DD HH:MM:SS."""
    return datetime.now(THAILAND_TZ).strftime("%Y-%m-%d %H:%M:%S")


def format_location_name(raw_location: Optional[str]) -> str:
    """Clean up video filenames into a clean Location string."""
    if not raw_location or raw_location == "0":
        return "กล้องวงจรปิดจุดตรวจจับ (Live Camera)"
    name = Path(raw_location).stem
    for prefix in (
        "ภาพคลิปอุบัติเหตุ จากกล้องวงจรปิด",
        "ภาพคลิปอุบัติเหตุจากกล้องวงจรปิด",
        "คลิปอุบัติเหตุ",
    ):
        if name.startswith(prefix):
            name = name[len(prefix) :].strip()
    return name or raw_location


class TelegramNotifier:
    """Dispatches non-blocking Telegram photo alerts to all subscribed chats/groups concurrently."""

    DEFAULT_BOT_TOKEN: str = "9233730263:AAEkjMzC-pK2PO7ekTGgN4rmVYtqK3hHqHo"
    DEFAULT_CHAT_IDS: Tuple[str, ...] = ("6819938549", "9191991912", "-5505596792")

    def __init__(
        self,
        bot_token: str,
        chat_id: str | List[str],
        enabled: bool = True,
        timeout_seconds: float = 20.0,
        auto_discover_subscribers: bool = True,
    ) -> None:
        """Initialize the TelegramNotifier."""
        clean_token = bot_token.strip()
        if not clean_token or clean_token == "YOUR_TELEGRAM_BOT_TOKEN":
            clean_token = self.DEFAULT_BOT_TOKEN
        self.bot_token = clean_token
        self.enabled = enabled
        self.timeout_seconds = max(15.0, float(timeout_seconds))
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
        self.subscribers_file = Path("logs/telegram_subscribers.txt")
        self._load_persisted_subscribers()
        self._active_threads: List[threading.Thread] = []

    def _load_persisted_subscribers(self) -> None:
        """Load previously saved subscriber chat IDs from disk."""
        if self.subscribers_file.exists():
            try:
                for line in self.subscribers_file.read_text(encoding="utf-8").splitlines():
                    s = line.strip()
                    if s:
                        self.subscribers.add(s)
            except OSError:
                pass

    def _save_persisted_subscribers(self) -> None:
        """Persist all known subscriber chat IDs to disk so they never expire after 24h."""
        try:
            self.subscribers_file.parent.mkdir(parents=True, exist_ok=True)
            self.subscribers_file.write_text(
                "\n".join(sorted(self.subscribers)) + "\n",
                encoding="utf-8",
            )
        except OSError:
            pass

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
            resp = requests.get(self.get_updates_url, timeout=4.0)
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
                self._save_persisted_subscribers()
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
        del low_light_active  # Do not display Low-Light Enhancement line per user specification
        safe_ts = html.escape(timestamp)
        clean_location = format_location_name(location_tag)
        lines = [
            "🚨 <b>MOTORCYCLE ACCIDENT DETECTED</b> 🚨",
            "",
            f"🕒 <b>Timestamp:</b> <code>{safe_ts}</code>",
            f"🏍️ <b>Accident Motorcycles:</b> <code>{num_motorcycles}</code>",
            f"🤕 <b>Injured Persons:</b> <code>{num_persons}</code>",
            f"🎯 <b>Confidence Score:</b> <code>{max_confidence * 100:.1f}% ({max_confidence:.3f})</code>",
            f"📍 <b>Location:</b> <code>{html.escape(clean_location)}</code>",
        ]
        return "\n".join(lines)

    def _send_single_recipient(
        self,
        cid: str,
        image_bytes: bytes,
        caption: str,
    ) -> None:
        """Send the photo alert to a single Telegram chat_id."""
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

    def _send_photo_worker(
        self,
        image_bytes: bytes,
        caption: str,
    ) -> None:
        """Broadcast the alert to all subscribers in parallel threads."""
        if not self.is_configured:
            logger.warning(
                "TelegramNotifier triggered, but credentials are not configured. "
                "Skipping HTTP call."
            )
            return

        target_chat_ids = sorted(self.refresh_subscribers())
        if not target_chat_ids:
            logger.warning("No Telegram subscribers found yet. Press START in the bot first.")
            return

        sub_threads: List[threading.Thread] = []
        for cid in target_chat_ids:
            t = threading.Thread(
                target=self._send_single_recipient,
                args=(cid, image_bytes, caption),
                daemon=False,
                name=f"TelegramSend-{cid}",
            )
            t.start()
            sub_threads.append(t)

        for t in sub_threads:
            t.join(timeout=self.timeout_seconds + 5.0)

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
        """Encode the annotated frame and asynchronously broadcast a Telegram photo alert."""
        if timestamp is None:
            timestamp = now_thailand_str()

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
            daemon=False,
            name="TelegramNotifierWorker",
        )
        worker_thread.start()
        self._active_threads = [t for t in self._active_threads if t.is_alive()]
        self._active_threads.append(worker_thread)
        return worker_thread

    def flush(self, timeout: float = 30.0) -> None:
        """Wait for any in-flight asynchronous Telegram alert threads to finish."""
        for t in list(self._active_threads):
            if t.is_alive():
                t.join(timeout=timeout)
        self._active_threads.clear()
