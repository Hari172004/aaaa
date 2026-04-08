"""
trade_journal.py — Agni-V Trade Journal
========================================
Logs every trade with its trigger reason, signal strength, session,
and outcome. Lets you see WHICH signals actually make money over time.

Famous bots like Zignaly, 3Commas track this internally.
We expose it as a JSON file at data/trade_journal.json.

Usage:
    journal = TradeJournal()
    entry_id = journal.log_open(symbol, direction, reason, strength, entry, sl, tp)
    journal.log_close(entry_id, pnl, exit_reason)
    stats = journal.win_rate_by_trigger()
"""

import json
import logging
import pathlib
import uuid
from datetime import datetime, timezone

logger = logging.getLogger("agniv.journal")

_JOURNAL_FILE = pathlib.Path("data/trade_journal.json")


class TradeJournal:
    """
    Persistent trade journal with per-trigger win rate analysis.
    Thread-safe for a single process.
    """

    def __init__(self, journal_file: str = ""):
        self._file = pathlib.Path(journal_file) if journal_file else _JOURNAL_FILE
        self._file.parent.mkdir(parents=True, exist_ok=True)

    # ── Internal helpers ─────────────────────────────────────────────

    def _load(self) -> list:
        try:
            if self._file.exists():
                return json.loads(self._file.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"[Journal] Load error: {e}")
        return []

    def _save(self, data: list):
        try:
            self._file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"[Journal] Save error: {e}")

    # ── Public API ───────────────────────────────────────────────────

    def log_open(
        self,
        symbol: str,
        direction: str,
        reason: str,
        strength: float,
        entry: float,
        sl: float,
        tp: float,
        session: str = "",
        strategy: str = "",
    ) -> str:
        """
        Record a newly opened trade. Returns a unique entry_id to pass to log_close().
        'reason' should be the signal reason string from generate_signal()
        e.g. 'ICT Sweep+FVG, H1 Trend+, Vol Spike'
        """
        entry_id = str(uuid.uuid4())[:12]
        record = {
            "id":         entry_id,
            "symbol":     symbol,
            "direction":  direction,
            "reason":     reason,
            "strength":   round(strength, 3),
            "entry":      entry,
            "sl":         sl,
            "tp":         tp,
            "session":    session,
            "strategy":   strategy,
            "open_time":  datetime.now(timezone.utc).isoformat(),
            "close_time": None,
            "pnl":        None,
            "exit_reason": None,
            "win":        None,
        }
        data = self._load()
        data.append(record)
        self._save(data)
        logger.info(
            f"[Journal] 📝 Trade opened | {symbol} {direction} | "
            f"Strength={strength:.0%} | Reason={reason[:60]} | ID={entry_id}"
        )
        return entry_id

    def log_close(self, entry_id: str, pnl: float, exit_reason: str = ""):
        """Update the journal record when a trade closes."""
        data   = self._load()
        found  = False
        for rec in data:
            if rec.get("id") == entry_id:
                rec["close_time"]  = datetime.now(timezone.utc).isoformat()
                rec["pnl"]         = round(pnl, 2)
                rec["exit_reason"] = exit_reason
                rec["win"]         = pnl >= 0
                found = True
                break
        if found:
            self._save(data)
            emoji = "✅" if pnl >= 0 else "❌"
            logger.info(
                f"[Journal] {emoji} Trade closed | ID={entry_id} | "
                f"PnL=${pnl:+.2f} | Exit={exit_reason}"
            )
        else:
            logger.warning(f"[Journal] log_close: ID {entry_id} not found")

    # ── Analytics ───────────────────────────────────────────────────

    def win_rate_by_trigger(self) -> dict:
        """
        Returns win-rate breakdown per trigger keyword found in the 'reason' field.
        Example output:
        {
            "ICT Sweep+FVG": {"trades": 12, "wins": 9, "win_rate": 75.0},
            "BullByte Ultimate": {"trades": 8,  "wins": 7, "win_rate": 87.5},
            "EMA+RSI Cross":     {"trades": 20, "wins": 9, "win_rate": 45.0},
        }
        """
        TRIGGER_KEYWORDS = [
            "ICT Sweep+FVG",
            "BullByte Ultimate",
            "HA Breakout",
            "EMA+RSI Cross",
            "EMA+RSI Sustained",
            "H1 Trend+",
            "H1 Trend-",
            "Vol Spike",
            "PPO:Confirm",
        ]

        data   = [r for r in self._load() if r.get("win") is not None]
        result = {}

        for kw in TRIGGER_KEYWORDS:
            relevant = [r for r in data if kw in r.get("reason", "")]
            if not relevant:
                continue
            wins = sum(1 for r in relevant if r["win"])
            result[kw] = {
                "trades":   len(relevant),
                "wins":     wins,
                "losses":   len(relevant) - wins,
                "win_rate": round(wins / len(relevant) * 100, 1),
                "total_pnl": round(sum(r["pnl"] for r in relevant if r["pnl"] is not None), 2),
            }

        # Sort by win_rate descending
        return dict(sorted(result.items(), key=lambda x: x[1]["win_rate"], reverse=True))

    def summary(self) -> dict:
        """Overall journal summary statistics."""
        data   = self._load()
        closed = [r for r in data if r.get("win") is not None]
        total  = len(closed)
        if total == 0:
            return {"total_trades": 0, "win_rate": 0.0, "total_pnl": 0.0}

        wins     = sum(1 for r in closed if r["win"])
        total_pnl = sum(r["pnl"] for r in closed if r["pnl"] is not None)
        avg_win  = sum(r["pnl"] for r in closed if r.get("win") and r["pnl"] is not None) / max(wins, 1)
        avg_loss = sum(r["pnl"] for r in closed if not r.get("win") and r["pnl"] is not None) / max(total - wins, 1)

        return {
            "total_trades":   total,
            "wins":           wins,
            "losses":         total - wins,
            "win_rate":       round(wins / total * 100, 1),
            "total_pnl":      round(total_pnl, 2),
            "avg_win":        round(avg_win, 2),
            "avg_loss":       round(avg_loss, 2),
            "profit_factor":  round(abs(avg_win / avg_loss), 2) if avg_loss != 0 else 0.0,
            "best_triggers":  self.win_rate_by_trigger(),
        }

    def print_report(self):
        """Print a readable journal report to the terminal and log."""
        s = self.summary()
        lines = [
            "━" * 55,
            "  📊  AGNI-V TRADE JOURNAL REPORT",
            "━" * 55,
            f"  Total Trades : {s['total_trades']}",
            f"  Win Rate     : {s['win_rate']}%  ({s['wins']}W / {s['losses']}L)",
            f"  Total PnL    : ${s['total_pnl']:+.2f}",
            f"  Avg Win      : ${s.get('avg_win', 0):+.2f}",
            f"  Avg Loss     : ${s.get('avg_loss', 0):+.2f}",
            f"  Profit Factor: {s.get('profit_factor', 0):.2f}",
            "",
            "  Win Rate by Trigger:",
        ]
        for trigger, stats in s.get("best_triggers", {}).items():
            bar = "█" * int(stats["win_rate"] / 10)
            lines.append(
                f"  {trigger:<25} {stats['win_rate']:>5.1f}%  "
                f"[{bar:<10}]  ({stats['trades']} trades)"
            )
        lines.append("━" * 55)
        report = "\n".join(lines)
        logger.info("\n" + report)
        print(report)
        return report
