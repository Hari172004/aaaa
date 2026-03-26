"""
btc_alerts.py — BTC Telegram Alerts
===================================
Handles all Bitcoin-specific notifications.
"""

import logging
from datetime import datetime

logger = logging.getLogger("agniv.btc_alerts")

class BTCAlerts:
    """Sends BTC-specific alerts to Telegram."""

    def __init__(self, alert_manager):
        self.alerts = alert_manager
        self._sent_alerts = {}

    def signal_detected(self, strategy: str, direction: str, price: float, reason: str):
        self.signal_alert("BTCUSD", direction, strategy, reason, entry=price)

    def signal_alert(self, symbol: str, signal: str, strategy: str, reason: str,
                     entry: float = 0.0, sl: float = 0.0, tp: float = 0.0):
        emoji = "🟢" if signal == "BUY" else "🔴"
        msg = (
            f"₿ <b>Agni-V BTC Signal</b>\n"
            f"{'─' * 30}\n"
            f"{emoji} <b>{symbol} {strategy} — {signal}</b>\n\n"
            f"📝 <i>{reason}</i>\n"
        )
        if entry > 0:
            msg += (
                f"\n💰 Entry: <code>{entry:.2f}</code>\n"
                f"🛑 SL:    <code>{sl:.2f}</code>\n"
                f"🎯 TP:    <code>{tp:.2f}</code>\n"
            )
        msg += f"\n🕐 {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC"
        self.alerts.send_telegram(msg)

    def whale_alert(self, amount: float, source: str):
        """Whale alert with 60-min cooldown."""
        alert_key = "whale_alert"
        if not self._should_send(alert_key, cooldown_mins=60):
            return

        msg = (
            f"🐋 <b>Whale Move Detected</b>\n"
            f"Amount: <code>{amount:.2f} BTC</code> sent to <code>{source}</code>\n"
            f"⚠️ Caution recommended for new BTC trades."
        )
        self.alerts.send_telegram(msg, is_alert=True)
        logger.info(f"[BTCAlerts] {msg.replace(chr(10), ' | ')}")
        self._mark_sent(alert_key)

    def extreme_sentiment_alert(self, score: float, label: str):
        """Sentiment alert with 60-min cooldown."""
        alert_key = "sentiment_alert"
        if not self._should_send(alert_key, cooldown_mins=60):
            return

        msg = (
            f"🚨 <b>BTC Sentiment Alert</b>\n"
            f"Mood: <code>{label}</code> (Score: {score:.2f})\n"
            f"⚠️ Market extremes detected. Reducing trade sizes."
        )
        self.alerts.send_telegram(msg, is_alert=True)
        logger.info(f"[BTCAlerts] {msg.replace(chr(10), ' | ')}")
        self._mark_sent(alert_key)

    # ── Internal Helpers ──────────────────────────────────────────────────

    def _should_send(self, key: str, cooldown_mins: int = 60) -> bool:
        """Check if an alert of this type was sent recently."""
        last_time = self._sent_alerts.get(f"{key}_time")
        if not last_time:
            return True
        elapsed = (datetime.utcnow() - last_time).total_seconds() / 60
        return elapsed >= cooldown_mins

    def _mark_sent(self, key: str):
        """Mark an alert as sent with current timestamp."""
        self._sent_alerts[f"{key}_time"] = datetime.utcnow()
