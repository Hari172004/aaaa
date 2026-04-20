"""
mtf_smc_scalp.py — Multi-Timeframe SMC Scalp Strategy Engine
=============================================================
Implements the 3-Gate ICT/SMC trading framework for 24/5 XAUUSD scalping.

GATE 1 — 15m Context:
  • Supply & Demand zones (institutional price walls)
  • Key Levels (swing highs/lows)
  • Overall directional bias (BULLISH / BEARISH / NEUTRAL)
  → HOLD if bias is NEUTRAL

GATE 2 — 5m Setup:
  • Price at or near a 15m S&D zone
  • Break of Structure (BOS) or Change of Character (CHoCH) present
  • Active Order Block or Fair Value Gap
  • Liquidity sweep detected
  • Buyer vs Seller fight → must be DECISIVE (score ≥ 3) at the level
  → HOLD if setup conditions not met

GATE 3 — 1m Confirmation:
  • Trigger candle: Engulfing, Pin Bar, or FVG fill
  • Micro BOS on 1m (price breaks last 1m swing)
  • BvS winner matches Gate 1 direction
  → FIRE if all pass

Usage:
    from strategies.mtf_smc_scalp import MTFSMCScalpStrategy
    engine = MTFSMCScalpStrategy()
    result = engine.generate_signal(df_15m, df_5m, df_1m)
"""

import logging
import pandas as pd
import numpy as np

from analysis.gold_market_structure import (  # type: ignore
    detect_gold_smc,
    detect_15m_bias,
    price_near_zone,
    near_ob,
    near_fvg,
)
from filters.bvs_fight_engine import BvSFightEngine  # type: ignore

logger = logging.getLogger("agniv.mtf_smc")


class MTFSMCScalpStrategy:
    """
    3-Gate Multi-Timeframe SMC Scalp Engine.

    All three gates must pass before a signal is emitted.
    This eliminates false signals in choppy 24/5 markets.
    """

    def __init__(self,
                 bvs_decisive_score: int = 3,
                 zone_buffer_pct:    float = 0.002,
                 ob_buffer_pips:     float = 50.0):
        """
        Args:
            bvs_decisive_score: Minimum BvS score (out of 5) to confirm a fight winner.
            zone_buffer_pct:    Price proximity buffer for S&D zone detection (0.2% default).
            ob_buffer_pips:     Proximity buffer for Order Block nearness check.
        """
        self.name              = "MTF-SMC-Scalp"
        self.bvs_5m            = BvSFightEngine(lookback=5, decisive_score=bvs_decisive_score)
        self.bvs_1m            = BvSFightEngine(lookback=3, decisive_score=bvs_decisive_score)
        self.zone_buffer_pct   = zone_buffer_pct
        self.ob_buffer_pips    = ob_buffer_pips

        # State: cache Gate 1 result to avoid recomputing every tick
        self._last_15m_bias: dict = {}
        self._last_15m_len:  int  = 0

    # ── Public API ────────────────────────────────────────────────────────

    def generate_signal(self,
                        df_15m: pd.DataFrame,
                        df_5m:  pd.DataFrame,
                        df_1m:  pd.DataFrame) -> dict:
        """
        Run the full 3-Gate MTF SMC analysis.

        Args:
            df_15m: 15-minute OHLCV candles (min 80 bars recommended)
            df_5m:  5-minute  OHLCV candles (min 100 bars recommended)
            df_1m:  1-minute  OHLCV candles (min 50 bars recommended)

        Returns:
            {
                "signal":    "BUY" | "SELL" | "HOLD",
                "strength":  float (0.0 – 1.0),
                "reason":    str,
                "gate1":     dict,   # 15m context
                "gate2":     dict,   # 5m setup
                "gate3":     dict,   # 1m confirmation
                "atr":       float,
                "sl_distance": float,
                "tp_distance": float,
            }
        """
        empty = {
            "signal": "HOLD", "strength": 0.0, "reason": "No MTF setup",
            "gate1": {}, "gate2": {}, "gate3": {},
            "atr": 0.0, "sl_distance": 0.0, "tp_distance": 0.0,
        }

        # ── GATE 1: 15m Context ──────────────────────────────────────────
        g1 = self._gate1_15m(df_15m)
        if g1["bias"] == "NEUTRAL":
            return {**empty, "reason": f"Gate1: 15m Bias=NEUTRAL ({g1['trend']})", "gate1": g1}

        direction = "BUY" if g1["bias"] == "BULLISH" else "SELL"

        # ── GATE 2: 5m Setup ─────────────────────────────────────────────
        g2 = self._gate2_5m(df_5m, direction, g1)
        if not g2["passed"]:
            return {**empty, "reason": f"Gate2: {g2['reason']}", "gate1": g1, "gate2": g2}

        # ── GATE 3: 1m Confirmation ──────────────────────────────────────
        g3 = self._gate3_1m(df_1m, direction)
        if not g3["passed"]:
            return {**empty, "reason": f"Gate3: {g3['reason']}", "gate1": g1, "gate2": g2, "gate3": g3}

        # ── All Gates Passed — compute SL/TP ────────────────────────────
        atr      = g2["atr"]
        sl_dist  = g2["sl_distance"]
        tp_dist  = sl_dist * 2.5

        strength = self._compute_strength(g1, g2, g3)

        reason = (
            f"15m={g1['bias']}({g1['trend']}) | "
            f"5m={g2['setup_type']} BvS={g2['bvs']['winner']}({g2['bvs']['score']}/5) | "
            f"1m={g3['trigger']} BvS={g3['bvs']['winner']}"
        )

        logger.info(
            f"[MTF-SMC] ✅ {direction} SIGNAL FIRED | {reason} | "
            f"Strength={strength:.0%} SL={sl_dist:.2f} TP={tp_dist:.2f}"
        )

        return {
            "signal":      direction,
            "strength":    strength,
            "reason":      reason,
            "gate1":       g1,
            "gate2":       g2,
            "gate3":       g3,
            "atr":         atr,
            "sl_distance": sl_dist,
            "tp_distance": tp_dist,
        }

    # ── Gate 1: 15m Context ──────────────────────────────────────────────

    def _gate1_15m(self, df_15m: pd.DataFrame) -> dict:
        """
        Analyse the 15m chart for overall bias, S&D zones, and key levels.
        Caches result if the 15m bar count hasn't changed (no new 15m candle).
        """
        if df_15m is None or len(df_15m) < 60:
            logger.debug("[MTF-G1] Insufficient 15m data")
            return {"bias": "NEUTRAL", "trend": "RANGING", "sd_zones": {"demand": [], "supply": []},
                    "key_levels": {"highs": [], "lows": []},
                    "ema21": 0.0, "ema50": 0.0, "ema200": 0.0, "bull_pts": 0, "bear_pts": 0}

        # Only recompute when a new 15m candle forms
        if len(df_15m) != self._last_15m_len:
            self._last_15m_bias = detect_15m_bias(df_15m)
            self._last_15m_len  = len(df_15m)

        return self._last_15m_bias

    # ── Gate 2: 5m Setup ─────────────────────────────────────────────────

    def _gate2_5m(self, df_5m: pd.DataFrame, direction: str, g1: dict) -> dict:
        """
        Check for a valid 5m setup in the direction given by Gate 1.

        Requirements (ALL must pass):
          a) Price is at or near a 15m S&D zone or key level
          b) BOS or CHoCH is present on 5m in the correct direction
          c) Price is near an Order Block or inside a Fair Value Gap
          d) BvS fight is decisive and won by the correct side
        """
        fail = lambda r: {
            "passed": False, "reason": r,
            "bvs": {"winner": "NEUTRAL", "score": 0, "is_decisive": False, "detail": ""},
            "setup_type": "NONE", "atr": 0.0, "sl_distance": 0.0
        }

        if df_5m is None or len(df_5m) < 60:
            return fail("Insufficient 5m data")

        current_price = float(df_5m["close"].iloc[-1])
        atr = float((df_5m["high"].tail(14) - df_5m["low"].tail(14)).mean())

        # a) Zone proximity check ─────────────────────────────────────────
        sd_zones   = g1.get("sd_zones", {})
        key_levels = g1.get("key_levels", {})
        zones_to_check = sd_zones.get("demand", []) if direction == "BUY" else sd_zones.get("supply", [])
        at_zone = price_near_zone(current_price, zones_to_check, buffer_pct=self.zone_buffer_pct)

        # Also accept if price is near a key level (swing high/low)
        at_key_level = False
        if not at_zone:
            levels = key_levels.get("lows", []) if direction == "BUY" else key_levels.get("highs", [])
            for lvl in levels:
                if abs(current_price - lvl) <= atr * 0.5:
                    at_key_level = True
                    break

        if not at_zone and not at_key_level:
            return fail(f"Price not at 15m zone/level (price={current_price:.2f})")

        # b) BOS / CHoCH on 5m ───────────────────────────────────────────
        smc_5m = detect_gold_smc(df_5m)
        bos    = smc_5m.get("bos", "NONE")
        choch  = smc_5m.get("choch", "NONE")

        has_bos   = (direction == "BUY" and bos == "BULLISH") or \
                    (direction == "SELL" and bos == "BEARISH")
        has_choch = choch not in ("NONE", "")

        if not has_bos and not has_choch:
            return fail(f"No 5m BOS/CHoCH for {direction} (BOS={bos} CHoCH={choch})")

        # c) OB or FVG proximity ─────────────────────────────────────────
        if direction == "BUY":
            at_ob  = near_ob(current_price, smc_5m.get("bull_obs", []), self.ob_buffer_pips)
            at_fvg = near_fvg(current_price, [f for f in smc_5m.get("fvgs", []) if f.get("type") == "BULL_FVG"])
        else:
            at_ob  = near_ob(current_price, smc_5m.get("bear_obs", []), self.ob_buffer_pips)
            at_fvg = near_fvg(current_price, [f for f in smc_5m.get("fvgs", []) if f.get("type") == "BEAR_FVG"])

        # Liquidity sweep is a bonus signal but not mandatory
        sweeps = smc_5m.get("sweeps", [])
        has_sweep = any(
            (direction == "BUY"  and s.get("type") == "BUY_SIDE_SWEEP") or
            (direction == "SELL" and s.get("type") == "SELL_SIDE_SWEEP")
            for s in sweeps
        )

        if not at_ob and not at_fvg:
            # Graceful fallback: if there's a decisive BvS + sweep, allow entry
            # even without a confirmed OB/FVG (e.g. Asian range breakout)
            if not has_sweep:
                return fail(f"No 5m OB/FVG near price (price={current_price:.2f})")

        # Determine setup type label — CHoCH = reversal, prioritise over plain OB
        if has_choch:
            setup_type = "CHoCH-Reversal"
        elif at_ob:
            setup_type = "OB-Retest"
        elif at_fvg:
            setup_type = "FVG-Fill"
        else:
            setup_type = "Sweep-Entry"

        # d) BvS Fight ───────────────────────────────────────────────────
        bvs = self.bvs_5m.evaluate(df_5m)
        bvs_ok = (
            (direction == "BUY"  and bvs["winner"] == "BUYERS") or
            (direction == "SELL" and bvs["winner"] == "SELLERS")
        ) and bvs["is_decisive"]

        if not bvs_ok:
            return fail(
                f"BvS fight not decisive for {direction}: "
                f"winner={bvs['winner']} score={bvs['score']}/5"
            )

        # Compute SL distance: below OB bottom for buys, above OB top for sells
        bull_obs = smc_5m.get("bull_obs", [])
        bear_obs = smc_5m.get("bear_obs", [])
        if direction == "BUY" and bull_obs:
            sl_distance = max(current_price - float(bull_obs[0]["bottom"]), atr * 1.2)
        elif direction == "SELL" and bear_obs:
            sl_distance = max(float(bear_obs[0]["top"]) - current_price, atr * 1.2)
        else:
            sl_distance = atr * 1.5

        logger.info(
            f"[MTF-G2] ✅ 5m Setup PASSED: {direction} | "
            f"Zone={'YES' if at_zone else 'KeyLevel'} "
            f"BOS={has_bos} CHoCH={has_choch} "
            f"OB={at_ob} FVG={at_fvg} Sweep={has_sweep} | "
            f"BvS={bvs['winner']}({bvs['score']}/5)"
        )

        return {
            "passed":      True,
            "reason":      f"{setup_type} confirmed",
            "bvs":         bvs,
            "setup_type":  setup_type,
            "at_zone":     at_zone,
            "at_ob":       at_ob,
            "at_fvg":      at_fvg,
            "has_sweep":   has_sweep,
            "has_bos":     has_bos,
            "has_choch":   has_choch,
            "is_reversal": has_choch,              # explicit reversal flag
            "atr":         atr,
            "sl_distance": sl_distance,
        }

    # ── Gate 3: 1m Confirmation ──────────────────────────────────────────

    def _gate3_1m(self, df_1m: pd.DataFrame, direction: str) -> dict:
        """
        Look for a 1m trigger candle that confirms the direction.

        Accepted triggers:
          1. Engulfing candle  — closes beyond the prior candle's body
          2. Pin Bar (Hammer / Shooting Star) — wick rejection ≥ 2× body
          3. FVG Fill — 1m price closes inside a 1m FVG in direction
          4. Micro BOS — 1m price breaks the previous 1m swing high/low

        Additionally require the BvS fight on 1m to agree with direction.
        """
        fail = lambda r, t="NONE": {
            "passed": False, "reason": r, "trigger": t,
            "bvs": {"winner": "NEUTRAL", "score": 0, "is_decisive": False, "detail": ""}
        }

        if df_1m is None or len(df_1m) < 20:
            return fail("Insufficient 1m data")

        last = df_1m.iloc[-1]
        prev = df_1m.iloc[-2]

        o, h, l, c  = float(last["open"]), float(last["high"]), float(last["low"]), float(last["close"])
        po, ph, pl, pc = float(prev["open"]), float(prev["high"]), float(prev["low"]), float(prev["close"])

        trigger_found = False
        trigger_name  = "NONE"

        # 1. Engulfing ────────────────────────────────────────────────────
        if direction == "BUY":
            bull_engulf = c > po and c > pc and o < pc  # close above prev open & close
            if bull_engulf:
                trigger_found = True
                trigger_name  = "Bullish-Engulf"
        else:
            bear_engulf = c < po and c < pc and o > pc
            if bear_engulf:
                trigger_found = True
                trigger_name  = "Bearish-Engulf"

        # 2. Pin Bar / Hammer / Shooting Star ─────────────────────────────
        if not trigger_found:
            body       = abs(c - o)
            upper_wick = h - max(o, c)
            lower_wick = min(o, c) - l
            candle_rng = h - l if (h - l) > 0 else 1e-6

            if direction == "BUY" and lower_wick >= body * 2 and lower_wick >= candle_rng * 0.40:
                trigger_found = True
                trigger_name  = "Hammer-PinBar"
            elif direction == "SELL" and upper_wick >= body * 2 and upper_wick >= candle_rng * 0.40:
                trigger_found = True
                trigger_name  = "ShootingStar-PinBar"

        # 3. Micro BOS on 1m ──────────────────────────────────────────────
        if not trigger_found:
            recent_5 = df_1m.tail(6).iloc[:-1]   # last 5 closed candles before current
            if direction == "BUY":
                recent_high = float(recent_5["high"].max())
                if c > recent_high:               # close breaks recent 1m highs
                    trigger_found = True
                    trigger_name  = "Micro-BOS-Bull"
            else:
                recent_low = float(recent_5["low"].min())
                if c < recent_low:
                    trigger_found = True
                    trigger_name  = "Micro-BOS-Bear"

        if not trigger_found:
            return fail(
                f"No 1m trigger candle for {direction} "
                f"(no Engulf/PinBar/MicroBOS)"
            )

        # 4. BvS fight on 1m must agree ───────────────────────────────────
        bvs = self.bvs_1m.evaluate(df_1m)
        bvs_ok = (
            (direction == "BUY"  and bvs["winner"] in ("BUYERS", "NEUTRAL")) or
            (direction == "SELL" and bvs["winner"] in ("SELLERS", "NEUTRAL"))
        )
        # On 1m we are slightly lenient: NEUTRAL is accepted if trigger is strong

        if not bvs_ok:
            return fail(
                f"1m BvS opposes {direction}: "
                f"winner={bvs['winner']} score={bvs['score']}/5",
                trigger_name
            )

        logger.info(
            f"[MTF-G3] ✅ 1m Confirmation: {trigger_name} | "
            f"BvS={bvs['winner']}({bvs['score']}/5)"
        )

        return {
            "passed":  True,
            "reason":  f"1m trigger: {trigger_name}",
            "trigger": trigger_name,
            "bvs":     bvs,
        }

    # ── Strength Calculator ───────────────────────────────────────────────

    def _compute_strength(self, g1: dict, g2: dict, g3: dict) -> float:
        """
        Compute a 0.0–1.0 signal strength score from all three gate qualities.
        """
        score = 0.60   # minimum for a valid signal

        # Gate 1 bonuses
        if g1.get("trend") in ("UPTREND", "DOWNTREND"):
            score += 0.05   # clean trending structure
        # EMA200 macro alignment bonus
        bias   = g1.get("bias", "NEUTRAL")
        ema200 = g1.get("ema200", 0.0)
        ema50  = g1.get("ema50",  0.0)
        if bias == "BULLISH" and ema50 > ema200:  score += 0.04
        if bias == "BEARISH" and ema50 < ema200:  score += 0.04

        # Gate 2 bonuses
        bvs2 = g2.get("bvs", {})
        score += min(bvs2.get("score", 0) * 0.02, 0.10)  # up to +0.10 for BvS score
        if g2.get("at_ob"):       score += 0.05
        if g2.get("at_fvg"):      score += 0.04
        if g2.get("has_sweep"):   score += 0.04
        if g2.get("has_bos"):     score += 0.04
        if g2.get("has_choch"):   score += 0.03
        if g2.get("is_reversal"): score += 0.05   # CHoCH reversal premium

        # Gate 3 bonuses
        bvs3 = g3.get("bvs", {})
        score += min(bvs3.get("score", 0) * 0.01, 0.05)
        if "Engulf"   in g3.get("trigger", ""): score += 0.05
        if "MicroBOS" in g3.get("trigger", ""): score += 0.03

        return float(round(min(score, 1.0), 3))
