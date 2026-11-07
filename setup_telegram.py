"""Interactive & Automatic Telegram Bot Connector for Accident Detection System.

Usage:
    ./.venv/bin/python setup_telegram.py --token "<BOT_TOKEN>"

Features:
1. Validates your Bot Token via Telegram getMe API.
2. Automatically opens your new Bot inside the macOS Telegram app so you can click 'START'.
3. Automatically detects your `chat_id` from `getUpdates` (no manual lookup needed).
4. Saves `bot_token` and `chat_id` directly into `config.yaml`.
5. Sends a real test accident snapshot + HTML caption to your Telegram immediately.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, Optional

import cv2
import requests
import yaml

from notification.telegram_bot import TelegramNotifier


def verify_bot_token(token: str) -> Dict[str, Any]:
    """Verify the bot token with Telegram API and return bot info."""
    url = f"https://api.telegram.org/bot{token}/getMe"
    resp = requests.get(url, timeout=10)
    data = resp.json()
    if not resp.ok or not data.get("ok"):
        raise ValueError(f"Invalid Bot Token: {data.get('description', resp.text)}")
    return data["result"]


def auto_detect_chat_id(token: str, bot_username: str, wait_seconds: int = 90) -> Optional[str]:
    """Wait for the user to press Start or send a message to the bot, then return chat_id."""
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    print(f"\n👉 กำลังเปิดบอท @{bot_username} ในแอป Telegram...")
    print(f"👉 กรุณากดปุ่ม 'START' (หรือพิมพ์ข้อความอะไรก็ได้ 1 ครั้ง) ในแชท @{bot_username}")

    try:
        subprocess.run(
            ["open", f"tg://resolve?domain={bot_username}"],
            check=False,
        )
    except Exception:
        pass

    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=10)
            data = resp.json()
            if data.get("ok") and data.get("result"):
                for update in reversed(data["result"]):
                    msg = (
                        update.get("message")
                        or update.get("edited_message")
                        or update.get("channel_post")
                        or update.get("my_chat_member", {})
                    )
                    chat = msg.get("chat")
                    if chat and "id" in chat:
                        return str(chat["id"])
        except requests.RequestException:
            pass
        time.sleep(2.0)
        print(".", end="", flush=True)

    print()
    return None


def update_config_yaml(token: str, chat_id: str, config_path: Path = Path("config.yaml")) -> None:
    """Write bot_token and chat_id into config.yaml."""
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if "telegram" not in cfg:
        cfg["telegram"] = {}
    cfg["telegram"]["enabled"] = True
    cfg["telegram"]["bot_token"] = token.strip()
    cfg["telegram"]["chat_id"] = str(chat_id).strip()

    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-connect Telegram Bot to config.yaml")
    parser.add_argument("--token", type=str, default=None, help="Telegram Bot Token from @BotFather")
    parser.add_argument("--chat-id", type=str, default=None, help="Optional Telegram Chat ID")
    args = parser.parse_args()

    token = args.token
    if not token:
        print("============================================================")
        print("🤖 ระบบเชื่อมต่อ Telegram Bot อัตโนมัติ (Telegram Auto-Setup)")
        print("============================================================")
        print("กำลังเปิด @BotFather ในแอป Telegram ให้คุณ...")
        try:
            subprocess.run(["open", "tg://resolve?domain=BotFather"], check=False)
        except Exception:
            pass
        print("1. ในแชท @BotFather ให้พิมพ์ /newbot (หรือ /mybots หากมีบอทอยู่แล้ว)")
        print("2. คัดลอก HTTP API Token (เช่น 123456789:AAH...) มาวางด้านล่างนี้:")
        token = input("\n🔑 วาง Bot Token ของคุณที่นี่: ").strip()

    if not token:
        print("❌ ไม่ได้รับ Bot Token ยกเลิกการตั้งค่า")
        sys.exit(1)

    bot_info = verify_bot_token(token)
    bot_username = bot_info.get("username", "")
    print(f"✅ เชื่อมต่อบอทสำเร็จ: @{bot_username} ({bot_info.get('first_name')})")

    chat_id = args.chat_id
    if not chat_id:
        chat_id = auto_detect_chat_id(token, bot_username)

    if not chat_id:
        print("❌ ยังไม่พบข้อความในบอท กรุณากด Start ในแชทบอทแล้วรันใหม่อีกครั้ง")
        sys.exit(1)

    print(f"\n✅ ค้นพบ Chat ID อัตโนมัติ: {chat_id}")
    update_config_yaml(token, chat_id)
    print("✅ บันทึกค่า bot_token และ chat_id ลงใน config.yaml เรียบร้อยแล้ว!")

    # Send a real test accident snapshot
    snapshots = sorted(Path("logs/snapshots").glob("accident_frame_*.jpg"))
    notifier = TelegramNotifier(bot_token=token, chat_id=chat_id, enabled=True)
    if snapshots:
        img = cv2.imread(str(snapshots[-1]))
    else:
        img = None

    if img is not None:
        print(f"📤 กำลังส่งภาพทดสอบแจ้งเตือนอุบัติเหตุ ({snapshots[-1].name}) ไปยัง Telegram...")
        thread = notifier.send_accident_alert(
            frame_bgr=img,
            num_motorcycles=1,
            num_persons=1,
            max_confidence=0.766,
            location_tag="สี่แยกบ่อขิง อ.พนัสนิคม จ.ชลบุรี (Test Alert)",
        )
        if thread is not None:
            thread.join(timeout=15.0)
        print("🎉 ส่งแจ้งเตือนเข้า Telegram เรียบร้อยแล้ว! เช็คดูในแอป Telegram ได้เลยครับ")


if __name__ == "__main__":
    main()
