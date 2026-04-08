"""
risk_manager.py — Smart Risk Management Engine for Agni-V
==============================================================
Handles: position sizing, SL/TP, breakeven, consecutive-loss stops,
daily loss limits, and reduction of risk as drawdown increases.

PRO UPGRADES (v2):
- Anti-Martingale: size UP after wins, DOWN after losses
- Cooldown Circuit Breaker: 45-min pause after 2 losses (not full-day block)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("agniv.risk_manager")


@dataclass
class RiskState:
    consecutive_losses: int   = 0
    daily_loss: float         = 0.0
    daily_starting_balance: float = 0.0
    trade_count_today: int    = 0
    wins_today: int           = 0
    losses_today: int         = 0
    consecutive_wins: int     = 0    # ← Anti-Martingale tracker
    paused: bool              = False
    pause_reason: str         = ""
    cooldown_until: float     = 0.0  # ← Unix timestamp; 0 = no cooldown
    anti_martingale_mult: float = 1.0  # ← Current lot-size multiplier (0.5 – 1.5)


class RiskManager:
    """
    Shared risk engine used by all trading modes (Demo, Real, Funded).
    Funded mode layered ON TOP via FundedModeEngine.
    """

    def __init__(self,
                 max_risk_pct: float = 2.0,
                 max_daily_loss_pct: float = 5.0,
                 max_consecutive_losses: int = 3,
                 breakeven_at_r: float = 1.0,
                 cooldown_losses: int = 2,          # ← Trigger 45-min cooldown after N losses
                 cooldown_minutes: int = 45):        # ← How long the cooldown lasts
        self.max_risk_pct = max_risk_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_consecutive_losses = max_consecutive_losses
        self.breakeven_at_r = breakeven_at_r
        self.cooldown_losses = cooldown_losses
        self.cooldown_minutes = cooldown_minutes
        self.state = RiskState()

    def set_dynamic_safety(self, balance: float):
        """Adjust risk parameters based on account size."""
        if balance < 50:
            self.breakeven_at_r = 0.3  # Nano-Safety: Protect $10 accounts
            self.max_daily_loss_pct = 20.0 # Increase limit for tiny accounts to avoid instant lock
            logger.info(f"[RiskMgr] NANO balance ({balance}) detected. Breakeven=0.3R, MaxLoss=20%.")
        elif balance < 500:
            self.breakeven_at_r = 0.7 
            self.max_daily_loss_pct = 10.0
            logger.info(f"[RiskMgr] Low balance ({balance}) detected. Breakeven=0.7R, MaxLoss=10%.")
        else:
            self.breakeven_at_r = 1.0

    # ── Lot Size Calculation ───────────────────────────────────

    def calculate_lot_size(self, balance: float, sl_pips: float,
                           pip_value: float, symbol: str) -> float:
        """
        Calculate lot size so that max_risk_pct % of balance is at risk.
        pip_value = value in account currency per 1 pip on 1 standard lot.
        sl_pips   = distance to stop-loss in pips.
        """
        if sl_pips <= 0 or pip_value <= 0:
            logger.warning("[RiskMgr] Invalid SL or pip_value — using minimum lot 0.01")
            return 0.01

        risk_amount = balance * (self.max_risk_pct / 100)
        lot_size = risk_amount / (sl_pips * pip_value)
        lot_size = round(float(max(0.01, min(lot_size, 10.0))), 2)  # type: ignore[arg-type]  Clamp 0.01–10
        logger.debug(
            f"[RiskMgr] LotSize={lot_size} | Balance=${balance:.2f} "
            f"| Risk=${risk_amount:.2f} | SL={sl_pips}pips"
        )
        return lot_size

    def calculate_lot_size_adjusted(self, balance: float, sl_pips: float,
                                    pip_value: float, symbol: str) -> float:
        """
        Anti-Martingale variant: applies the current size multiplier.
        Use this instead of calculate_lot_size() for all live entries.
        """
        base = self.calculate_lot_size(balance, sl_pips, pip_value, symbol)
        adjusted = base * self.state.anti_martingale_mult
        adjusted = round(float(max(0.01, min(adjusted, 10.0))), 2)  # type: ignore[arg-type]
        if self.state.anti_martingale_mult != 1.0:
            logger.info(
                f"[RiskMgr] Anti-Martingale: Base lot={base} × "
                f"{self.state.anti_martingale_mult:.2f} = {adjusted}"
            )
        return adjusted

    # ── ATR-Based SL / TP ─────────────────────────────────────

    def calculate_sl_tp(self, entry_price: float, atr: float,
                        direction: str, sl_multiplier: float = 1.5,
                        tp_multiplier: float = 2.5) -> tuple[float, float]:
        """
        direction: 'BUY' or 'SELL'
        Returns (stop_loss_price, take_profit_price)
        """
        sl_distance = atr * sl_multiplier
        tp_distance = atr * tp_multiplier

        if direction.upper() == "BUY":
            sl = entry_price - sl_distance
            tp = entry_price + tp_distance
        else:
            sl = entry_price + sl_distance
            tp = entry_price - tp_distance

        sl = round(float(sl), 5)  # type: ignore[arg-type]
        tp = round(float(tp), 5)  # type: ignore[arg-type]
        logger.debug(f"[RiskMgr] ATR={atr:.5f} SL={sl} TP={tp} [{direction}]")
        return sl, tp

    # ── Trailing Stop Loss Logic ───────────────────────────────

    def should_update_sl(self, entry_price: float, current_price: float,
                         current_sl: float, initial_sl: float, direction: str,
                         override_breakeven_r: float = None) -> tuple[bool, float]:
        """
        Returns (should_move: bool, new_sl: float).
        Implements a Trailing Stop Loss. Activates when profit >= 1R (initial SL distance).
        Once active, it maintains exactly a 1R distance behind the current price.
        """
        # We don't trail if the initial SL is invalid or missing
        if not initial_sl or initial_sl == entry_price:
            return False, current_sl

        threshold = override_breakeven_r if override_breakeven_r is not None else self.breakeven_at_r

        if direction.upper() == "BUY":
            initial_risk = entry_price - initial_sl
            profit = current_price - entry_price

            # Activation at threshold * R profit
            if profit >= (initial_risk * threshold):
                # Trail by 1R distance behind current price
                new_sl = current_price - initial_risk
                new_sl = round(float(new_sl), 5) # type: ignore[arg-type]
                # Only move if it locks in more profit than current SL
                if new_sl > current_sl:
                    return True, new_sl

        else:  # SELL
            initial_risk = initial_sl - entry_price
            profit = entry_price - current_price

            if profit >= (initial_risk * threshold):
                new_sl = current_price + initial_risk
                new_sl = round(float(new_sl), 5) # type: ignore[arg-type]
                # Only move if it locks in more profit than current SL (lower is better for SL in SELL)
                if new_sl < current_sl:
                    return True, new_sl

        return False, current_sl

    # ── Consecutive Loss / Daily Loss Guards ───────────────────

    def check_can_trade(self, current_balance: float) -> tuple[bool, str]:
        """
        Base risk check — call BEFORE every trade.
        Returns (can_trade: bool, reason: str)
        Includes smart cooldown: 45-min pause after N consecutive losses
        (not a full-day block) so the bot resumes automatically.
        """
        s = self.state
        now = time.time()

        if s.paused:
            return False, f"Risk pause: {s.pause_reason}"

        # ── Cooldown Circuit Breaker ───────────────────────────────────────────────
        if s.cooldown_until > 0:
            if now < s.cooldown_until:
                remaining_mins = int((s.cooldown_until - now) / 60)
                return False, f"Cooldown active — resumes in {remaining_mins} min"
            else:
                # Cooldown expired — auto-resume
                s.cooldown_until = 0.0
                logger.info("[RiskMgr] ⏰ Cooldown lifted. Trading resumed automatically.")

        if s.consecutive_losses >= self.max_consecutive_losses:
            reason = f"Max consecutive losses ({self.max_consecutive_losses}) reached — pausing."
            self._pause(reason)
            return False, reason

        if s.daily_starting_balance > 0:
            daily_loss_pct = (s.daily_loss / s.daily_starting_balance) * 100
            if daily_loss_pct >= self.max_daily_loss_pct:
                reason = (
                    f"Daily max loss reached: {daily_loss_pct:.2f}% "
                    f"/ limit {self.max_daily_loss_pct}%"
                )
                self._pause(reason)
                return False, reason

        return True, "OK"

    def update_after_trade(self, pnl: float):
        """Update internal state after each trade closes. Applies Anti-Martingale sizing."""
        s = self.state
        s.trade_count_today += 1
        if pnl >= 0:
            s.consecutive_losses = 0
            s.wins_today         += 1
            s.consecutive_wins   += 1
            # Anti-Martingale: grow size by 10% after each win, cap at 1.5×
            s.anti_martingale_mult = min(1.0 + (s.consecutive_wins * 0.10), 1.50)
            # Reset streak after 3 wins (avoid over-sizing)
            if s.consecutive_wins >= 3:
                s.anti_martingale_mult = 1.0
                s.consecutive_wins = 0
                logger.info("[RiskMgr] Anti-Martingale: 3-win streak reset — size back to 1.0×")
            else:
                logger.info(f"[RiskMgr] Anti-Martingale: Win streak {s.consecutive_wins} — next lot ×{s.anti_martingale_mult:.2f}")
        else:
            s.consecutive_losses += 1
            s.consecutive_wins    = 0
            s.daily_loss         += abs(pnl)
            s.losses_today       += 1
            # Anti-Martingale: shrink size by 20% after each loss, floor at 0.5×
            s.anti_martingale_mult = max(s.anti_martingale_mult * 0.80, 0.50)
            logger.info(f"[RiskMgr] Anti-Martingale: Loss streak {s.consecutive_losses} — next lot ×{s.anti_martingale_mult:.2f}")
            # Cooldown Circuit Breaker: after N losses, pause for cooldown_minutes
            if s.consecutive_losses == self.cooldown_losses:
                cooldown_secs = self.cooldown_minutes * 60
                s.cooldown_until = time.time() + cooldown_secs
                logger.warning(
                    f"[RiskMgr] ⚡ Cooldown triggered after {self.cooldown_losses} losses! "
                    f"Trading paused for {self.cooldown_minutes} minutes."
                )
        logger.info(
            f"[RiskMgr] Trade result: PnL=${pnl:+.2f} | "
            f"ConsecLosses={s.consecutive_losses} | "
            f"DailyLoss=${s.daily_loss:.2f} | SizeMult={s.anti_martingale_mult:.2f}"
        )

    def on_new_day(self, current_balance: float):
        """Reset daily counters at the start of each session."""
        s = self.state
        s.consecutive_losses     = 0
        s.consecutive_wins       = 0
        s.daily_loss             = 0.0
        s.daily_starting_balance = current_balance
        s.trade_count_today      = 0
        s.wins_today             = 0
        s.losses_today           = 0
        s.cooldown_until         = 0.0  # lift any active cooldown on new day
        s.anti_martingale_mult   = 1.0  # reset position sizing on new day
        if s.paused:
            s.paused       = False
            s.pause_reason = ""
            logger.info("[RiskMgr] Daily reset — pause lifted.")

    def _pause(self, reason: str):
        self.state.paused = True
        self.state.pause_reason = reason
        logger.error(f"[RiskMgr] 🚨 TRADING PAUSED: {reason}")

    def resume(self):
        """Manually resume (e.g. from the app)."""
        self.state.paused = False
        self.state.pause_reason = ""
        logger.info("[RiskMgr] Trading resumed manually.")

    def stats(self) -> dict:
        s = self.state
        total = s.wins_today + s.losses_today
        now   = time.time()
        cooldown_remaining = max(0, int((s.cooldown_until - now) / 60)) if s.cooldown_until > now else 0
        return {
            "consecutive_losses":    s.consecutive_losses,
            "consecutive_wins":      s.consecutive_wins,
            "daily_loss":            round(float(s.daily_loss), 2),  # type: ignore[arg-type]
            "trade_count_today":     s.trade_count_today,
            "wins_today":            s.wins_today,
            "losses_today":          s.losses_today,
            "win_rate_today":        round(float((s.wins_today / total * 100) if total > 0 else 0.0), 1),  # type: ignore[arg-type]
            "paused":                s.paused,
            "pause_reason":          s.pause_reason,
            "cooldown_remaining_min": cooldown_remaining,
            "anti_martingale_mult":  round(s.anti_martingale_mult, 2),
        }
