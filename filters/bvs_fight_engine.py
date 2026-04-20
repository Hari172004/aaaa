"""
bvs_fight_engine.py — Buyer vs Seller (BvS) Fight Detector
============================================================
At any key price level (Order Block, FVG, S&D Zone), this engine
scores 5 battle signals to determine who is winning:

  1. Candle Body Pressure  — who closed in control?
  2. Wick Rejection        — who rejected the level?
  3. Volume Surge          — is institutional money joining the fight?
  4. EMA9 Slope            — which direction is momentum pushing?
  5. Delta Volume Proxy    — buying vs selling candle sequence

Score ≥ 3 → Fight is DECISIVE → Enter with the winner.

Used by:
  - MTF Gate 2 (5m Setup): BvS fight at the OB / FVG level
  - MTF Gate 3 (1m Confirmation): BvS fight must confirm direction
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("agniv.bvs_fight")


class BvSFightEngine:
    """
    Buyer vs Seller Fight Detector.

    Evaluates the last N candles to determine if buyers or sellers
    are winning the battle at a price level.
    """

    def __init__(self, lookback: int = 5, vol_surge_mult: float = 1.4,
                 ema_period: int = 9, decisive_score: int = 3):
        """
        Args:
            lookback:        Number of candles to analyse (last N candles).
            vol_surge_mult:  Volume must be this × average to count as a surge.
            ema_period:      EMA period for slope / momentum calculation.
            decisive_score:  Minimum score (out of 5) to call a winner.
        """
        self.lookback       = lookback
        self.vol_surge_mult = vol_surge_mult
        self.ema_period     = ema_period
        self.decisive_score = decisive_score

    # ── Public API ────────────────────────────────────────────────────────

    def evaluate(self, df: pd.DataFrame) -> dict:
        """
        Run the full 5-signal BvS analysis on the latest candle cluster.

        Returns:
            {
                "winner":        "BUYERS" | "SELLERS" | "NEUTRAL",
                "score":         int (0-5),
                "bull_score":    int,
                "bear_score":    int,
                "body_pressure": "BULL" | "BEAR" | "NONE",
                "wick_rejection":"BULL" | "BEAR" | "NONE",
                "vol_surge":     bool,
                "ema_slope":     "UP" | "DOWN" | "FLAT",
                "delta_seq":     "BULL" | "BEAR" | "NONE",
                "is_decisive":   bool,
                "detail":        str    # human readable summary
            }
        """
        empty = {
            "winner": "NEUTRAL", "score": 0, "bull_score": 0, "bear_score": 0,
            "body_pressure": "NONE", "wick_rejection": "NONE",
            "vol_surge": False, "ema_slope": "FLAT", "delta_seq": "NONE",
            "is_decisive": False, "detail": "Insufficient data"
        }

        min_bars = max(self.lookback, self.ema_period) + 5
        if df is None or len(df) < min_bars:
            return empty

        try:
            return self._score(df)
        except Exception as e:
            logger.error(f"[BvS] evaluate error: {e}")
            return empty

    # ── Internal Scoring ─────────────────────────────────────────────────

    def _score(self, df: pd.DataFrame) -> dict:
        last     = df.iloc[-1]
        recent   = df.tail(self.lookback)

        o  = float(last["open"])
        h  = float(last["high"])
        l  = float(last["low"])
        c  = float(last["close"])
        rng = h - l if (h - l) > 0 else 1e-6

        # ── Signal 1: Candle Body Pressure ───────────────────────────────
        # Who closed in control of the most recent candle?
        body       = abs(c - o)
        body_ratio = body / rng         # 0 = all wicks, 1 = all body

        if c > o and body_ratio >= 0.40:
            body_pressure = "BULL"      # green body dominates
        elif c < o and body_ratio >= 0.40:
            body_pressure = "BEAR"      # red body dominates
        else:
            body_pressure = "NONE"      # indecision doji

        # ── Signal 2: Wick Rejection ─────────────────────────────────────
        # Buyers reject LOWS → long lower wick = bullish rejection
        # Sellers reject HIGHS → long upper wick = bearish rejection
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l

        if lower_wick > upper_wick * 1.5 and lower_wick > rng * 0.25:
            wick_rejection = "BULL"     # buyers defended low strongly
        elif upper_wick > lower_wick * 1.5 and upper_wick > rng * 0.25:
            wick_rejection = "BEAR"     # sellers defended high strongly
        else:
            wick_rejection = "NONE"

        # ── Signal 3: Volume Surge ────────────────────────────────────────
        # Is there institutional money fueling this candle?
        vol_col  = "volume" if "volume" in df.columns else "tick_volume"
        avg_vol  = float(df[vol_col].tail(20).mean())
        curr_vol = float(last.get(vol_col, 0))
        vol_data_valid = avg_vol > 0 and curr_vol > 0
        vol_surge = vol_data_valid and (curr_vol >= avg_vol * self.vol_surge_mult)

        # ── Signal 4: EMA9 Slope (Momentum Direction) ────────────────────
        ema9 = df["close"].ewm(span=self.ema_period, adjust=False).mean()
        e_now  = float(ema9.iloc[-1])
        e_prev = float(ema9.iloc[-2])
        e_diff = e_now - e_prev

        # Normalise slope relative to price to avoid large-number bias
        slope_pct = abs(e_diff) / (float(last["close"]) + 1e-9)

        if e_diff > 0 and slope_pct > 0.00005:    # > 0.005% meaningful slope
            ema_slope = "UP"
        elif e_diff < 0 and slope_pct > 0.00005:
            ema_slope = "DOWN"
        else:
            ema_slope = "FLAT"

        # ── Signal 5: Delta Volume Sequence ──────────────────────────────
        # Count bull vs bear candles in recent lookback window.
        bull_candles = int((recent["close"] > recent["open"]).sum())
        bear_candles = int((recent["close"] < recent["open"]).sum())

        if bull_candles > bear_candles + 1:
            delta_seq = "BULL"
        elif bear_candles > bull_candles + 1:
            delta_seq = "BEAR"
        else:
            delta_seq = "NONE"

        # ── Scoring ──────────────────────────────────────────────────────
        bull_score = 0
        bear_score = 0

        # Body Pressure
        if body_pressure == "BULL":  bull_score += 1
        elif body_pressure == "BEAR": bear_score += 1

        # Wick Rejection
        if wick_rejection == "BULL":  bull_score += 1
        elif wick_rejection == "BEAR": bear_score += 1

        # Volume Surge (counts toward whichever side closed)
        if vol_surge:
            if c >= o:  bull_score += 1   # surge on a bull candle → buyers funded
            else:       bear_score += 1

        # EMA Slope
        if ema_slope == "UP":   bull_score += 1
        elif ema_slope == "DOWN": bear_score += 1

        # Delta Sequence
        if delta_seq == "BULL":  bull_score += 1
        elif delta_seq == "BEAR": bear_score += 1

        total_score = bull_score + bear_score

        if bull_score > bear_score and bull_score >= self.decisive_score:
            winner = "BUYERS"
            score  = bull_score
        elif bear_score > bull_score and bear_score >= self.decisive_score:
            winner = "SELLERS"
            score  = bear_score
        else:
            winner = "NEUTRAL"
            score  = max(bull_score, bear_score)

        is_decisive = winner != "NEUTRAL"

        detail = (
            f"Winner={winner} Score={bull_score}B/{bear_score}S | "
            f"Body={body_pressure} Wick={wick_rejection} "
            f"Vol={'SURGE' if vol_surge else 'normal'} "
            f"EMA={ema_slope} Seq={delta_seq}"
        )

        if is_decisive:
            logger.info(f"[BvS] ⚔️  FIGHT DECIDED: {detail}")
        else:
            logger.debug(f"[BvS] Fight ongoing: {detail}")

        return {
            "winner":         winner,
            "score":          score,
            "bull_score":     bull_score,
            "bear_score":     bear_score,
            "body_pressure":  body_pressure,
            "wick_rejection": wick_rejection,
            "vol_surge":      vol_surge,
            "ema_slope":      ema_slope,
            "delta_seq":      delta_seq,
            "is_decisive":    is_decisive,
            "detail":         detail,
        }


# ── Module-level singleton (for easy import) ─────────────────────────────────
bvs_engine = BvSFightEngine()
