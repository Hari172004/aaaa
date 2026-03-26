"""
gold_alerts.py -- Full Telegram alert system for Gold (XAUUSD)
Alert types: Signal, DXY warning, News pause, Kill Zone open,
             Fundamental score, Spread warning, ETF flow, Geopolitical, Daily report.
"""

import logging
from datetime import datetime, timezone, date

logger = logging.getLogger("agniv.gold_alerts")

EMOJI = {
    "gold":         "🥇",
    "dxy":          "💵",
    "news":         "🚨",
    "lightning":    "⚡",
    "chart":        "📊",
    "warning":      "⚠️",
    "etf":          "📈",
    "globe":        "🌍",
    "report":       "📋",
    "buy":          "🟢",
    "sell":         "🔴",
    "hold":         "⏸️",
}


class GoldAlerts:
    def __init__(self, alert_manager):
        """
        alert_manager: existing AlertManager instance from logger.py.
        Wraps it to send Gold-specific Telegram messages.
        """
        self.alerts = alert_manager
        self._sent_alerts = {}  # Tracks sent alerts by key and date

    # ── 1. Trade Signal ───────────────────────────────────────────────────

    def signal_alert(self, symbol: str, signal: str, strategy: str, reason: str,
                     entry: float = 0.0, sl: float = 0.0, tp: float = 0.0):
        """Fire on every Gold BUY/SELL signal."""
        direction_emoji = EMOJI["buy"] if signal == "BUY" else EMOJI["sell"]
        action_line     = f"{direction_emoji} <b>{symbol} {strategy} — {signal}</b>"
        msg = (
            f"🥇 <b>Agni-V Gold Signal</b>\n"
            f"{'─' * 30}\n"
            f"{action_line}\n\n"
            f"📝 <i>{reason}</i>\n"
        )
        if entry > 0:
            msg += (
                f"\n💰 Entry: <code>{entry:.3f}</code>\n"
                f"🛑 SL:    <code>{sl:.3f}</code>\n"
                f"🎯 TP:    <code>{tp:.3f}</code>\n"
            )
        msg += f"\n🕐 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC"
        self._send(msg, is_signal=True)

    # ── 2. DXY Warning ────────────────────────────────────────────────────

    def dxy_warning(self, dxy_val: float, change_pct: float):
        """Fire when DXY spikes significantly — caution for Gold longs."""
        alert_key = "dxy_warning"
        if not self._should_send(alert_key, cooldown_mins=30):
            return

        severity = "Spiking" if change_pct > 0.3 else "Rising"
        msg = (
            f"{EMOJI['dxy']} <b>DXY {severity} — Gold Caution Mode Active</b>\n"
            f"DXY: <code>{dxy_val:.2f}</code> ({'+' if change_pct > 0 else ''}{change_pct:.2f}%)\n"
            f"⚠️ Gold long positions at risk. Review open trades."
        )
        self._send(msg, alert=True)
        self._mark_sent(alert_key)

    # ── 3. News Pause ─────────────────────────────────────────────────────

    def news_pause_alert(self, event_name: str, minutes_to: int = 30):
        """Fire when high-impact news is approaching."""
        msg = (
            f"{EMOJI['news']} <b>High Impact News — Gold Bot Paused for Safety</b>\n"
            f"Event: <code>{event_name}</code>\n"
            f"Pausing gold trading for <code>{minutes_to}</code> minutes."
        )
        self._send(msg, alert=True)

    def news_resume_alert(self, event_name: str):
        """Fire when news blackout window ends."""
        msg = (
            f"✅ <b>Gold Trading Resumed</b> — <code>{event_name}</code> data released.\n"
            f"Agni-V scanning gold markets again."
        )
        self._send(msg)

    # ── 4. Kill Zone Open ─────────────────────────────────────────────────

    def session_alert(self, session: str, is_killzone: bool):
        """Fire when London or NY Kill Zone opens."""
        if not is_killzone:
            return
            
        today = date.today().isoformat()
        alert_key = f"session_{session}_{today}"
        
        # Deduplicate: only send once per session per day
        if self._sent_alerts.get(alert_key):
            return

        session_display = "London" if session == "LONDON" else "New York"
        msg = (
            f"{EMOJI['lightning']} <b>{session_display} Kill Zone Open</b>\n"
            f"Agni-V scanning gold opportunities.\n"
            f"🕐 <code>{datetime.now(timezone.utc).strftime('%H:%M')}</code> UTC"
        )
        self._send(msg, alert=True)
        self._sent_alerts[alert_key] = True

    # ── 5. Fundamental Alert ──────────────────────────────────────────────

    def fundamental_alert(self, score: float, bias: str, dxy: float, us10y: float, vix: float):
        """Fire when macro bias shifts — deduplicated by bias state."""
        if bias == "NEUTRAL":
            return
            
        alert_key = f"fund_bias_{bias}"
        # Only send if the bias is DIFFERENT from the last sent bias TODAY
        if self._sent_alerts.get("last_fund_bias") == bias:
            return

        indicator_emoji = EMOJI["chart"] if bias == "BULLISH" else EMOJI["warning"]
        msg = (
            f"{indicator_emoji} <b>Gold Fundamental Score: {bias}</b>\n"
            f"{'─' * 28}\n"
            f"Score: <code>{score:+.0f}</code> | DXY: <code>{dxy:.2f}</code> | US10Y: <code>{us10y:.2f}%</code> | VIX: <code>{vix:.1f}</code>\n"
        )
        if bias == "BEARISH":
            msg += "⬇️ Reducing gold position size automatically."
        else:
            msg += "⬆️ Gold macro conditions bullish — normal sizing."
            
        self._send(msg)
        self._sent_alerts["last_fund_bias"] = bias

    # ── 6. Spread Warning ─────────────────────────────────────────────────

    def spread_alert(self, spread_points: float, threshold: float = 3.0):
        """Fire when gold spread is too wide — 30 min cooldown."""
        alert_key = "spread_warning"
        if not self._should_send(alert_key, cooldown_mins=30):
            return

        msg = (
            f"{EMOJI['warning']} *Gold Spread Too High — Paused*\n"
            f"Current spread: `{spread_points:.2f}` pts (`{spread_points * 10:.0f}` pips)\n"
            f"Threshold: `{threshold:.2f}` pts. Waiting for normal spread to resume."
        )
        self._send(msg, alert=True)
        self._mark_sent(alert_key)

    # ── Internal Helpers ──────────────────────────────────────────────────

    def _should_send(self, key: str, cooldown_mins: int = 60) -> bool:
        """Check if an alert of this type was sent recently."""
        last_time = self._sent_alerts.get(f"{key}_time")
        if not last_time:
            return True
        elapsed = (datetime.now() - last_time).total_seconds() / 60
        return elapsed >= cooldown_mins

    def _mark_sent(self, key: str):
        """Mark an alert as sent with current timestamp."""
        self._sent_alerts[f"{key}_time"] = datetime.now()

    # ── 7. ETF Flow Alert ─────────────────────────────────────────────────

    def etf_flow_alert(self, flow_pct: float, direction: str):
        """Fire when GLD ETF shows significant inflow or outflow."""
        flow_label = "Inflow" if direction == "IN" else "Outflow"
        bias_label = "Gold Bullish Bias Active" if direction == "IN" else "Gold Risk — Smart Money Exiting"
        msg = (
            f"{EMOJI['etf']} *GLD ETF {flow_label} Detected*\n"
            f"Change: `{flow_pct:+.2f}%` today\n"
            f"📌 {bias_label}"
        )
        self._send(msg)

    # ── 8. Geopolitical Alert ─────────────────────────────────────────────

    def geopolitical_alert(self, event: str, severity: str = "HIGH"):
        """Fire when geopolitical risk triggers safe-haven demand for gold."""
        msg = (
            f"{EMOJI['globe']} *Geopolitical Risk Detected [{severity}]*\n"
            f"Event: _{event}_\n"
            f"🛡️ Gold Safe Haven Mode Active. Bias shifted Bullish."
        )
        self._send(msg, alert=True)

    # ── 9. Daily Report ───────────────────────────────────────────────────

    def daily_report(self, trades: int, pnl: float, win_rate: float,
                     fund_bias: str, sentiment: str, session_summary: str):
        """End-of-day summary report."""
        pnl_emoji = "📈" if pnl >= 0 else "📉"
        msg = (
            f"{EMOJI['gold']} <b>Agni-V Gold Daily Report {EMOJI['gold']}</b>\n"
            f"{'─' * 32}\n"
            f"📅 Date:       <code>{date.today().isoformat()}</code>\n"
            f"{pnl_emoji} PnL:        <code>{'+'if pnl>=0 else ''}{pnl:.2f}</code> USD\n"
            f"📊 Win Rate:   <code>{win_rate:.0f}%</code>\n"
            f"🔢 Trades:    <code>{trades}</code>\n"
            f"🌍 Fund Bias:  <code>{fund_bias}</code>\n"
            f"📰 Sentiment:  <code>{sentiment}</code>\n"
            f"🕐 Sessions:   <i>{session_summary}</i>"
        )
        self._send(msg)

    # ── Internal sender ───────────────────────────────────────────────────

    def _send(self, message: str, alert: bool = False, is_signal: bool = False):
        """Delegate to the AlertManager — supports both Telegram and email."""
        try:
            # Send to Telegram if it's a signal OR a specific alert (News/Session/Risk)
            if (is_signal or alert) and hasattr(self.alerts, "send_telegram"):
                self.alerts.send_telegram(message, is_alert=alert)
            
            # Always log to console/file
            clean_msg = message.replace('\n', ' | ')
            logger.info(f"[GoldAlerts] {clean_msg}")
        except Exception as e:
            logger.error(f"[GoldAlerts] Failed to send alert: {e}")
