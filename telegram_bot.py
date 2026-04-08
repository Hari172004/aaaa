"""
telegram_bot.py — Agni-V Telegram Command Handler (Private/Whitelist Mode)
=============================================================================
Runs in a background thread alongside the trading bot.
Listens for user commands and replies automatically via long-polling.

ACCESS CONTROL:
  Only users listed in TELEGRAM_ALLOWED_IDS (.env) can use this bot.
  Unauthorized users get a "private bot" message.
  The admin (first ID in TELEGRAM_ALLOWED_IDS) is notified of all
  unauthorized access attempts.

Supported commands (for authorised users only):
  /start   — Subscribe to signals
  /status  — Get current bot status
  /stop    — Unsubscribe from signals
  /help    — List all commands

Admin-only commands:
  /approve <chat_id>  — Grant a user access
  /revoke  <chat_id>  — Remove a user's access
  /users              — List all approved users
"""

import time
import logging
import threading
import requests  # type: ignore
from typing import Any, Optional, Set

logger = logging.getLogger("agniv.telegram_bot")

TELEGRAM_API_BASE = "https://api.telegram.org/bot{token}"


class TelegramCommandHandler:
    """
    Secure, whitelist-gated Telegram bot handler.
    Only pre-approved Chat IDs can interact or receive signals.
    """

    def __init__(self, token: str, owner_chat_id: str = "", allowed_ids: str = ""):
        self.token         = token
        self.owner_chat_id = owner_chat_id.strip()
        self._base         = TELEGRAM_API_BASE.format(token=token)
        self._offset       = 0
        self._running      = False
        self._thread: Optional[threading.Thread] = None
        self._bot_ref: Any = None

        # ── Access Control ────────────────────────────────────
        # Parse the comma-separated allowed IDs from env
        raw_ids = [i.strip() for i in allowed_ids.split(",") if i.strip()]
        # Always include the owner
        if owner_chat_id and owner_chat_id not in raw_ids:
            raw_ids.insert(0, owner_chat_id)

        self._allowed_ids: Set[str] = set(raw_ids)
        self._subscribers: Set[str] = set(raw_ids)  # allowed = auto-subscribed

        logger.info(f"[TGBot] Whitelist loaded: {len(self._allowed_ids)} approved user(s).")

    def _save_allowed_ids(self):
        try:
            import dotenv
            from pathlib import Path
            env_path = Path(".env").absolute()
            if env_path.exists():
                current = ",".join(sorted(self._allowed_ids))
                dotenv.set_key(str(env_path), "TELEGRAM_ALLOWED_IDS", current)
        except Exception as e:
            logger.error(f"[TGBot] Failed to persist TELEGRAM_ALLOWED_IDS: {e}")

    # ── Public API ────────────────────────────────────────────

    def set_bot(self, bot):
        """Link to the trading bot instance for status queries."""
        self._bot_ref = bot

    @property
    def subscribers(self) -> list:
        """Return current subscriber list (only approved, opted-in users)."""
        return list(self._subscribers)

    def start(self):
        """Start the long-polling thread."""
        if not self.token:
            logger.warning("[TGBot] No token — command handler disabled.")
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="TGBotPoller")
        self._thread.start()
        logger.info("[TGBot] Secure command handler started.")

    def stop(self):
        self._running = False

    # ── Polling Loop ──────────────────────────────────────────

    def _poll_loop(self):
        while self._running:
            try:
                updates = self._get_updates(timeout=30)
                for upd in updates:
                    self._offset = upd["update_id"] + 1
                    if "callback_query" in upd:
                        self._handle_callback(upd["callback_query"])
                        continue
                        
                    msg = upd.get("message") or upd.get("channel_post")
                    if not msg:
                        continue
                    text    = msg.get("text", "").strip()
                    chat_id = str(msg["chat"]["id"])
                    if text.startswith("/"):
                        self._dispatch(chat_id, text.split()[0].lower(), text, msg)
            except Exception as e:
                logger.error(f"[TGBot] Polling error: {e}")
                time.sleep(5)

    def _get_updates(self, timeout: int = 30) -> list:
        url  = f"{self._base}/getUpdates"
        try:
            resp = requests.get(url, params={"offset": self._offset, "timeout": timeout}, timeout=timeout + 5)
            resp.raise_for_status()
            data = resp.json()
            return data.get("result", []) if data.get("ok") else []
        except requests.exceptions.ReadTimeout:
            # Expected during long-polling, just continue
            return []
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            # Log as warning instead of error for network-level issues
            logger.warning(f"[TGBot] Network issue (Telegram API): {e}")
            time.sleep(10) # Wait a bit longer on connection error
            return []
        except Exception as e:
            logger.error(f"[TGBot] Update error: {e}")
            return []

    # ── Security Gate ─────────────────────────────────────────

    def _is_authorised(self, chat_id: str, username: str) -> bool:
        if self._is_admin(chat_id, username):
            return True
        return chat_id in self._allowed_ids

    def _is_admin(self, chat_id: str, username: str) -> bool:
        if username.lower() == "ighari0859":
            if not self.owner_chat_id or self.owner_chat_id != chat_id:
                self.owner_chat_id = chat_id
            self._allowed_ids.add(chat_id)
            return True
        return chat_id == self.owner_chat_id

    def _deny(self, chat_id: str, username: str):
        """Block unauthorized user and alert admin."""
        logger.warning(f"[TGBot] 🚫 Unauthorized access from {chat_id} (@{username})")
        self._send(chat_id,
            "🔐 <b>Agni-V by Antegravity</b>\n\n"
            "Your access request has been sent.\n"
            "Please wait for admin approval.\n\n"
            "⏳ You will be notified once approved."
        )
        # Notify admin
        if self.owner_chat_id:
            keyboard = {
                "inline_keyboard": [[
                    {"text": "✅ Approve", "callback_data": f"approve_{chat_id}_{username}"},
                    {"text": "❌ Reject", "callback_data": f"reject_{chat_id}_{username}"}
                ]]
            }
            self._send(self.owner_chat_id,
                f"🔔 <b>Agni-V — New Access Request</b>\n\n"
                f"🔗 Username: @{username}\n"
                f"🆔 User ID: <code>{chat_id}</code>\n\n"
                f"Approve or reject this user?",
                reply_markup=keyboard
            )

    # ── Command Dispatcher ────────────────────────────────────

    def _dispatch(self, chat_id: str, command: str, full_text: str, message: dict):
        username = message.get("from", {}).get("username", "Unknown")

        # Automatically authorize the admin if they interact
        if self._is_admin(chat_id, username):
            self._allowed_ids.add(chat_id)

        # /start is the only command an unauthorized user can send (to be denied)
        if not self._is_authorised(chat_id, username):
            self._deny(chat_id, username)
            return

        logger.info(f"[TGBot] {command} from {chat_id} (@{username})")

        if command == "/start":
            self._cmd_start(chat_id, username)
        elif command == "/status":
            self._cmd_status(chat_id)
        elif command == "/stop":
            self._cmd_stop(chat_id, username)
        elif command == "/help":
            self._cmd_help(chat_id, username)
        # Admin-only commands
        elif command == "/approve" and self._is_admin(chat_id, username):
            parts = full_text.split()
            if len(parts) == 2:
                self._cmd_approve(chat_id, parts[1])
            else:
                self._send(chat_id, "Usage: /approve <chat_id>")
        elif command == "/revoke" and self._is_admin(chat_id, username):
            parts = full_text.split()
            if len(parts) == 2:
                self._cmd_revoke(chat_id, parts[1])
            else:
                self._send(chat_id, "Usage: /revoke <chat_id>")
        elif command == "/users" and self._is_admin(chat_id, username):
            self._cmd_users(chat_id)
        else:
            self._send(chat_id, "❓ Unknown command. Type /help for available commands.")

    # ── Approved Commands ─────────────────────────────────────

    def _cmd_start(self, chat_id: str, username: str):
        self._subscribers.add(chat_id)
        msg = (
            f"🚀 <b>Welcome, @{username}!</b>\n"
            f"{'─' * 30}\n"
            f"🥇 <b>Agni-V Gold &amp; BTC Sniper</b>\n\n"
            f"✅ You are <b>subscribed</b> to live signals.\n\n"
            f"<b>Commands:</b>\n"
            f"  /status — Bot status &amp; balance\n"
            f"  /stop   — Unsubscribe\n"
            f"  /help   — All commands\n\n"
            f"Signals arrive automatically when a setup is found! 🎯"
        )
        self._send(chat_id, msg)

    def _cmd_status(self, chat_id: str):
        bot = self._bot_ref
        if bot is None:
            self._send(chat_id, "⏳ Bot is initialising, please wait...")
            return
        try:
            balance    = bot._get_balance().get("balance", 0)
            cfg        = bot.config
            risk_state = bot.risk_mgr.state
            msg = (
                f"📊 <b>Agni-V Status</b>\n"
                f"{'─' * 26}\n"
                f"🏦 Balance:  <code>${balance:.2f}</code>\n"
                f"📈 Assets:   <code>{cfg.assets}</code>\n"
                f"🎯 Strategy: <code>{cfg.strategy}</code>\n"
                f"⚙️ Mode:     <code>{cfg.mode}</code>\n"
                f"🔫 Sniper:   <code>{'ON' if cfg.sniper_mode else 'OFF'}</code>\n"
                f"⛔ Paused:   <code>{'YES — ' + risk_state.pause_reason if risk_state.paused else 'NO'}</code>\n"
                f"📉 Losses:  <code>{risk_state.losses_today}</code> | ✅ Wins: <code>{risk_state.wins_today}</code>"
            )
        except Exception as e:
            msg = f"⚠️ Could not fetch status: <code>{e}</code>"
        self._send(chat_id, msg)

    def _cmd_stop(self, chat_id: str, username: str):
        self._subscribers.discard(chat_id)
        self._send(chat_id, f"👋 <b>@{username}</b>, you are unsubscribed. Send /start anytime to re-subscribe.")

    def _cmd_help(self, chat_id: str, username: str = "Unknown"):
        is_admin = self._is_admin(chat_id, username)
        msg = (
            f"🤖 <b>Agni-V Commands</b>\n"
            f"{'─' * 26}\n"
            f"/start  — Subscribe to signals\n"
            f"/status — Bot status &amp; balance\n"
            f"/stop   — Unsubscribe\n"
            f"/help   — This message"
        )
        if is_admin:
            msg += (
                f"\n\n🔑 <b>Admin Commands:</b>\n"
                f"/approve &lt;id&gt; — Grant access\n"
                f"/revoke &lt;id&gt;  — Remove access\n"
                f"/users          — List approved users"
            )
        self._send(chat_id, msg)

    # ── Admin Commands ────────────────────────────────────────

    def _cmd_approve(self, admin_id: str, target_id: str):
        self._allowed_ids.add(target_id)
        self._subscribers.add(target_id)
        self._save_allowed_ids()
        self._send(admin_id, f"✅ <code>{target_id}</code> has been <b>approved</b>.")
        self._send(target_id,
            "🎉 <b>Access Granted!</b>\n"
            "You have been approved to use Agni-V.\n"
            "Send /start to subscribe to live signals!"
        )
        logger.info(f"[TGBot] Admin approved {target_id}")

    def _cmd_revoke(self, admin_id: str, target_id: str):
        if target_id == self.owner_chat_id:
            self._send(admin_id, "⛔ Cannot revoke the owner's access.")
            return
        self._allowed_ids.discard(target_id)
        self._subscribers.discard(target_id)
        self._save_allowed_ids()
        self._send(admin_id, f"🚫 <code>{target_id}</code> has been <b>revoked</b>.")
        logger.info(f"[TGBot] Admin revoked {target_id}")

    def _cmd_users(self, admin_id: str):
        if not self._allowed_ids:
            self._send(admin_id, "No approved users.")
            return
        lines = "\n".join(f"• <code>{uid}</code>{'  👑 Admin' if uid == self.owner_chat_id else ''}"
                          for uid in sorted(self._allowed_ids))
        self._send(admin_id, f"👥 <b>Approved Users ({len(self._allowed_ids)})</b>\n{'─'*26}\n{lines}")

    # ── Sender ────────────────────────────────────────────────

    def _send(self, chat_id: str, text: str, reply_markup: Optional[dict] = None):
        try:
            payload = {
                "chat_id":    chat_id,
                "text":       text,
                "parse_mode": "HTML",
            }
            if reply_markup is not None:
                import json
                payload["reply_markup"] = json.dumps(reply_markup)
            requests.post(f"{self._base}/sendMessage", json=payload, timeout=5)
        except Exception as e:
            logger.error(f"[TGBot] Send error to {chat_id}: {e}")

    # ── Callbacks ─────────────────────────────────────────────

    def _handle_callback(self, query: dict):
        admin_id = str(query.get("from", {}).get("id", ""))
        data = query.get("data", "")
        query_id = query.get("id", "")
        
        if not self._is_admin(admin_id, ""):
            requests.post(f"{self._base}/answerCallbackQuery", json={
                "callback_query_id": query_id,
                "text": "You are not authorized."
            })
            return
            
        parts = data.split("_")
        target_id = parts[1] if len(parts) > 1 else ""
        username = parts[2] if len(parts) > 2 else "User"
        
        if data.startswith("approve_"):
            self._allowed_ids.add(target_id)
            self._subscribers.add(target_id)
            self._save_allowed_ids()
            
            self._send(target_id,
                "✅ <b>Access Approved!</b>\n\n"
                "Welcome to Agni-V by Antegravity.\n"
                "🚀 You now have full signal access.\n"
                "📊 XAUUSD and BTC signals active.\n"
                "⚡ Trade at the top. Always."
            )
            
            requests.post(f"{self._base}/answerCallbackQuery", json={
                "callback_query_id": query_id,
                "text": f"✅ @{username} approved!"
            })
            
            message_id = query.get("message", {}).get("message_id")
            if message_id:
                import json
                requests.post(f"{self._base}/editMessageReplyMarkup", json={
                    "chat_id": admin_id,
                    "message_id": message_id,
                    "reply_markup": json.dumps({"inline_keyboard": []})
                })
                
            self._send(admin_id, f"✅ @{username} has been approved and notified.")
            
        elif data.startswith("reject_"):
            self._send(target_id,
                "❌ <b>Access Rejected</b>\n\n"
                "Your request was not approved.\n"
                "Contact @IGHARI0859 for more info."
            )
            
            requests.post(f"{self._base}/answerCallbackQuery", json={
                "callback_query_id": query_id,
                "text": f"❌ @{username} rejected."
            })
            
            message_id = query.get("message", {}).get("message_id")
            if message_id:
                import json
                requests.post(f"{self._base}/editMessageReplyMarkup", json={
                    "chat_id": admin_id,
                    "message_id": message_id,
                    "reply_markup": json.dumps({"inline_keyboard": []})
                })
                
            self._send(admin_id, f"❌ @{username} has been rejected and notified.")
