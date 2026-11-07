"""Asynchronous Telegram Bot notification module for real-time accident alerts."""

from __future__ import annotations

from datetime import datetime
import html
import logging
import threading
from typing import Optional

import cv2
import numpy as np
import requests

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Dispatches non-blocking Telegram photo alerts using background daemon threads."""

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        enabled: bool = True,
        timeout_seconds: float = 10.0,
    ) -> None:
        """Initialize the TelegramNotifier.

        Args:
            bot_token: Telegram Bot API token obtained from @BotFather.
            chat_id: Target Telegram user, group, or channel chat ID.
            enabled: Master switch to enable/disable network dispatch.
            timeout_seconds: HTTP request timeout in seconds.
        """
        self.bot_token = bot_token.strip()
        self.chat_id = str(chat_id).strip()
        self.enabled = enabled
        self.timeout_seconds = float(timeout_seconds)
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
        self._active_threads: list[threading.Thread] = []

    @property
    def is_configured(self) -> bool:
        """Return True if notifications are enabled and valid credentials are set."""
        return (
            self.enabled
            and bool(self.bot_token)
            and bool(self.chat_id)
            and self.bot_token != "YOUR_TELEGRAM_BOT_TOKEN"
            and self.chat_id != "YOUR_TELEGRAM_CHAT_ID"
        )

    @staticmethod
    def format_html_caption(
        timestamp: str,
        num_motorcycles: int,
        num_persons: int,
        max_confidence: float,
        low_light_active: bool = False,
        location_tag: Optional[str] = None,
    ) -> str:
        """Format the alert metadata into an HTML caption for Telegram.

        Args:
            timestamp: Formatted timestamp string of the confirmed accident event.
            num_motorcycles: Count of detected accident motorcycles (`motorcycle_accident`).
            num_persons: Count of detected injured persons (`person_accident`).
            max_confidence: Peak detection confidence score (0.0 - 1.0).
            low_light_active: Whether adaptive low-light enhancement was applied.
            location_tag: Optional camera or stream identifier.

        Returns:
            HTML-formatted string compatible with Telegram `parse_mode="HTML"`.
        """
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
        """Worker executed in a background thread to perform the HTTP POST request."""
        if not self.is_configured:
            logger.warning(
                "TelegramNotifier triggered, but credentials are not configured. "
                "Skipping HTTP call."
            )
            return

        files = {
            "photo": ("accident_snapshot.jpg", image_bytes, "image/jpeg"),
        }
        data = {
            "chat_id": self.chat_id,
            "caption": caption,
            "parse_mode": "HTML",
        }

        try:
            response = requests.post(
                self.api_url,
                data=data,
                files=files,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            logger.info("Telegram alert dispatched successfully (HTTP %d).", response.status_code)
        except requests.RequestException as exc:
            logger.error("Failed to send Telegram alert: %s", exc)

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
        """Encode the annotated frame and asynchronously send a Telegram photo alert.

        Args:
            frame_bgr: Annotated BGR image to attach to the alert.
            num_motorcycles: Number of accident motorcycles in the frame.
            num_persons: Number of injured persons in the frame.
            max_confidence: Highest detection confidence score among accident objects.
            timestamp: Optional ISO/human-readable timestamp (defaults to current local time).
            low_light_active: Whether the frame underwent low-light enhancement.
            location_tag: Optional source identifier.

        Returns:
            The started `threading.Thread` instance, or None if encoding failed.
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Encode image in memory prior to spawning the thread so frame_bgr can be mutated safely
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
