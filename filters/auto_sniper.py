"""
auto_sniper.py — Auto-Sniper Detection Engine
==============================================
Detects when institutional / smart-money players ("snipers") are entering
the market and automatically activates Sniper Mode for that trade.

Detection Signals (need 2+ to trigger):
  1. Volume Spike     — current bar volume > 2.5× the 20-bar average
  2. Impulsive Candle — body > 65% of the bar range (decisive move)
  3. Liquidity Sweep  — price swept a recent swing high/low and reversed

When triggered, returns a dict with:
  - is_sniper_entry: bool
  - direction:       "BUY" | "SELL" | None
  - signals_fired:   list of which signals triggered
  - strength_boost:  float added to base signal strength
  - sl_multiplier:   tighter SL (0.8–1.0×ATR)
  - tp_multiplier:   wider TP (3.0–4.0×ATR)
  - reason:          human-readable explanation
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger("agniv.auto_sniper")


class AutoSniperDetector:
    """
    Real-time institutional footprint scanner.

    Usage:
        detector = AutoSniperDetector()
        result   = detector.scan(df_1m, df_5m)
        if result["is_sniper_entry"]:
            # override sniper settings for this trade
    """

    def __init__(self,
                 volume_spike_mult: float = 2.5,   # vol > N× average to trigger
                 body_ratio_min:    float = 0.65,  # candle body / range ratio
                 sweep_lookback:    int   = 20,    # bars to look for swing levels
                 min_signals:       int   = 2):    # signals needed to activate
        self.volume_spike_mult = volume_spike_mult
        self.body_ratio_min    = body_ratio_min
        self.sweep_lookback    = sweep_lookback
        self.min_signals       = min_signals

    # ── Public API ────────────────────────────────────────────────────────

    def scan(self, df_1m: pd.DataFrame, df_5m: pd.DataFrame = None) -> dict:
        """
        Scan for institutional sniper entry patterns on the most recent candle.

        Args:
            df_1m: 1-minute OHLCV DataFrame (needs at least 25 bars)
            df_5m: 5-minute OHLCV DataFrame (optional, improves sweep detection)

        Returns:
            dict with is_sniper_entry, direction, signals_fired, etc.
        """
        base = {
            "is_sniper_entry": False,
            "direction":       None,
            "signals_fired":   [],
            "strength_boost":  0.0,
            "sl_multiplier":   1.5,   # default — unchanged
            "tp_multiplier":   2.5,   # default — unchanged
            "reason":          "No sniper pattern detected",
        }

        if df_1m is None or len(df_1m) < 25:
            return base

        signals_fired = []
        direction_votes = {"BUY": 0, "SELL": 0}

        # ── Signal 1: Volume Spike ─────────────────────────────────────────
        vol_result = self._check_volume_spike(df_1m)
        if vol_result["fired"]:
            signals_fired.append(f"VolSpike×{vol_result['ratio']:.1f}")
            direction_votes[vol_result["direction"]] += 1

        # ── Signal 2: Impulsive Candle ─────────────────────────────────────
        imp_result = self._check_impulsive_candle(df_1m)
        if imp_result["fired"]:
            signals_fired.append(f"Impulse({imp_result['body_ratio']:.0%}body)")
            direction_votes[imp_result["direction"]] += 1

        # ── Signal 3: Liquidity Sweep ──────────────────────────────────────
        df_for_sweep = df_5m if (df_5m is not None and len(df_5m) >= 25) else df_1m
        sweep_result = self._check_liquidity_sweep(df_1m, df_for_sweep)
        if sweep_result["fired"]:
            signals_fired.append(f"LiqSweep({sweep_result['type']})")
            direction_votes[sweep_result["direction"]] += 1

        # ── Signal 4: Engulfing Volume Candle (bonus) ──────────────────────
        engulf_result = self._check_volume_engulf(df_1m)
        if engulf_result["fired"]:
            signals_fired.append(f"VolEngulf({engulf_result['direction']})")
            direction_votes[engulf_result["direction"]] += 2  # strong signal — double vote

        # ── Evaluate ──────────────────────────────────────────────────────
        total_signals = len(signals_fired)
        if total_signals < self.min_signals:
            return {**base, "signals_fired": signals_fired}

        # Determine dominant direction
        if direction_votes["BUY"] > direction_votes["SELL"]:
            sniper_direction = "BUY"
        elif direction_votes["SELL"] > direction_votes["BUY"]:
            sniper_direction = "SELL"
        else:
            # Tie — conflicting signals, not a clean sniper entry
            return {**base, "signals_fired": signals_fired,
                    "reason": "Signals conflicting — no clear sniper direction"}

        # Calculate strength bonus based on number of confirming signals
        strength_boost = min(total_signals * 0.06, 0.20)  # up to +0.20 bonus

        # Tighter SL and wider TP for sniper trades
        sl_mult = max(1.5 - (total_signals * 0.1), 0.8)  # 0.8–1.5×ATR
        tp_mult = min(2.5 + (total_signals * 0.3), 4.0)  # 2.5–4.0×ATR

        reason = (
            f"🎯 AUTO-SNIPER ACTIVATED | {sniper_direction} | "
            f"Signals: {', '.join(signals_fired)} | "
            f"SL={sl_mult:.1f}×ATR TP={tp_mult:.1f}×ATR"
        )
        logger.info(f"[AutoSniper] {reason}")

        return {
            "is_sniper_entry": True,
            "direction":       sniper_direction,
            "signals_fired":   signals_fired,
            "strength_boost":  strength_boost,
            "sl_multiplier":   sl_mult,
            "tp_multiplier":   tp_mult,
            "reason":          reason,
        }

    # ── Internal Signal Detectors ──────────────────────────────────────────

    def _check_volume_spike(self, df: pd.DataFrame) -> dict:
        """Signal 1: Current bar volume > N× 20-bar average."""
        fail = {"fired": False, "direction": None, "ratio": 0.0}
        try:
            if "volume" not in df.columns or len(df) < 21:
                return fail

            cur_vol  = float(df["volume"].iloc[-1])
            avg_vol  = float(df["volume"].iloc[-21:-1].mean())

            if avg_vol <= 0:
                return fail

            ratio = cur_vol / avg_vol
            if ratio >= self.volume_spike_mult:
                # Direction from the spike candle's close vs open
                last = df.iloc[-1]
                direction = "BUY" if float(last["close"]) >= float(last["open"]) else "SELL"
                return {"fired": True, "direction": direction, "ratio": ratio}
        except Exception:
            pass
        return fail

    def _check_impulsive_candle(self, df: pd.DataFrame) -> dict:
        """Signal 2: Body > 65% of total range (explosive institutional move)."""
        fail = {"fired": False, "direction": None, "body_ratio": 0.0}
        try:
            last = df.iloc[-1]
            o, h, l, c = float(last["open"]), float(last["high"]), float(last["low"]), float(last["close"])
            rng  = h - l
            body = abs(c - o)

            if rng <= 0:
                return fail

            ratio = body / rng
            if ratio >= self.body_ratio_min:
                direction = "BUY" if c > o else "SELL"
                return {"fired": True, "direction": direction, "body_ratio": ratio}
        except Exception:
            pass
        return fail

    def _check_liquidity_sweep(self, df_1m: pd.DataFrame, df_ref: pd.DataFrame) -> dict:
        """
        Signal 3: Price swept a recent swing high/low and reversed.
        Classic 'stop hunt' — snipers clear liquidity then reverse hard.
        """
        fail = {"fired": False, "direction": None, "type": ""}
        try:
            lookback = min(self.sweep_lookback, len(df_ref) - 2)
            if lookback < 5:
                return fail

            # Recent swing high/low from reference timeframe
            recent = df_ref.iloc[-(lookback + 1):-1]
            swing_high = float(recent["high"].max())
            swing_low  = float(recent["low"].min())

            # Current 1m candle
            last = df_1m.iloc[-1]
            h, l, c, o = float(last["high"]), float(last["low"]), float(last["close"]), float(last["open"])

            # Bullish sweep: wick below swing_low but closed ABOVE (buyers absorbed sells)
            if l < swing_low and c > swing_low and c > o:
                return {"fired": True, "direction": "BUY", "type": "BullishSweep"}

            # Bearish sweep: wick above swing_high but closed BELOW (sellers absorbed buys)
            if h > swing_high and c < swing_high and c < o:
                return {"fired": True, "direction": "SELL", "type": "BearishSweep"}

        except Exception:
            pass
        return fail

    def _check_volume_engulf(self, df: pd.DataFrame) -> dict:
        """
        Bonus signal: Large engulfing candle WITH volume spike.
        Most reliable — institutions often enter this way.
        """
        fail = {"fired": False, "direction": None}
        try:
            if len(df) < 3:
                return fail

            last = df.iloc[-1]
            prev = df.iloc[-2]

            o, h, l, c   = float(last["open"]), float(last["high"]), float(last["low"]), float(last["close"])
            po, ph, pl, pc = float(prev["open"]), float(prev["high"]), float(prev["low"]), float(prev["close"])

            # Bullish engulf: closes above prev open AND has volume spike
            bull_engulf = c > po and c > pc and o < pc
            bear_engulf = c < po and c < pc and o > pc

            # Volume confirmation: this candle > 1.8× previous candle volume
            if "volume" in df.columns:
                vol_ratio = float(last["volume"]) / max(float(prev["volume"]), 1)
                vol_ok = vol_ratio >= 1.8
            else:
                vol_ok = True  # no volume data, accept on pattern alone

            if bull_engulf and vol_ok:
                return {"fired": True, "direction": "BUY"}
            if bear_engulf and vol_ok:
                return {"fired": True, "direction": "SELL"}

        except Exception:
            pass
        return fail
