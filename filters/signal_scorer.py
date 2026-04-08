"""
signal_scorer.py — 100 Point Master Signal Scoring Engine
Generates a weighted score out of 100 for every incoming trade signal.

v2.0 — Added Range Filter, RQK, WAE, and Supertrend scoring slots.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger("agniv.filters.scorer")


class SignalScorer:
    """
    Score breakdown (Total: 100):

    Core Technicals (55 pts):
      RSI in ideal zone (40–60):        +10 points
      EMA alignment correct:            +10 points
      MACD confirms direction:          +10 points
      Volume spike present:             +10 points
      Order Block or FVG present:       +15 points

    Context Filters (30 pts):
      Kill Zone active:                 +10 points
      News sentiment agrees:            +5  points
      Multi timeframe confluence:       +5  points
      Supertrend agrees:                +5  points
      RQK trend agrees:                 +5  points

    Momentum Gates (15 pts):
      Range Filter not CHOPPY:          +8  points
      WAE confirms explosion:           +7  points
    """

    def __init__(self):
        self.min_trade_threshold = 68   # ≥68/100 required to trade
        self.min_log_threshold   = 55   # ≥55/100 logged as near-miss

    def evaluate_signal(self, metrics: Dict[str, Any], direction: str) -> Dict[str, Any]:
        """
        Receives raw indicators and flags, calculates final score out of 100.
        Returns a dict containing the score and decision.

        metrics keys:
          rsi                : float  — current RSI value
          ema_aligned        : bool   — 9>21>50 for BUY, reversed for SELL
          macd_confirms      : bool   — MACD hist > 0 for BUY, < 0 for SELL
          volume_spike       : bool   — RVOL > 1.3
          structure_present  : bool   — OB or FVG present
          kill_zone_active   : bool   — London/NY session active
          news_agrees        : bool   — FinBERT sentiment matches direction
          mtf_confluence     : int    — number of confirming timeframes (0-5)
          supertrend_agrees  : bool   — Supertrend trend matches direction
          rqk_agrees         : bool   — RQK kernel trend matches direction
          range_filter_ok    : bool   — Range Filter is not CHOPPY
          wae_confirms       : bool   — WAE explosion confirmed in direction
        """
        score     = 0
        breakdown = {}
        is_buy    = direction.upper() == "BUY"

        # ── CORE TECHNICALS (55 pts) ──────────────────────────

        # 1. RSI Zone (+10) — ideal pullback zone 35-65
        rsi = metrics.get("rsi", 50)
        rsi_pts = 0
        if is_buy and 35 <= rsi <= 65:
            rsi_pts = 10
        elif not is_buy and 35 <= rsi <= 65:
            rsi_pts = 10
        score += rsi_pts
        breakdown["rsi"] = rsi_pts

        # 2. EMA Alignment (+10)
        ema_aligned = metrics.get("ema_aligned", False)
        ema_pts = 10 if ema_aligned else 0
        score += ema_pts
        breakdown["ema_aligned"] = ema_pts

        # 3. MACD Confirmation (+10)
        macd_confirms = metrics.get("macd_confirms", False)
        macd_pts = 10 if macd_confirms else 0
        score += macd_pts
        breakdown["macd_confirms"] = macd_pts

        # 4. Volume Spike (+10)
        volume_spike = metrics.get("volume_spike", False)
        vol_pts = 10 if volume_spike else 0
        score += vol_pts
        breakdown["volume_spike"] = vol_pts

        # 5. Market Structure — OB / FVG (+15) — highest weight: structure is key
        structure_present = metrics.get("structure_present", False)
        struct_pts = 15 if structure_present else 0
        score += struct_pts
        breakdown["structure_present"] = struct_pts

        # ── CONTEXT FILTERS (30 pts) ──────────────────────────

        # 6. Kill Zone (+10)
        kz_active = metrics.get("kill_zone_active", False)
        kz_pts = 10 if kz_active else 0
        score += kz_pts
        breakdown["kill_zone_active"] = kz_pts

        # 7. News Sentiment (+5)
        news_agrees = metrics.get("news_agrees", False)
        news_pts = 5 if news_agrees else 0
        score += news_pts
        breakdown["news_agrees"] = news_pts

        # 8. MTF Confluence (+5)
        mtf_score = metrics.get("mtf_confluence", 0)
        mtf_pts = 5 if mtf_score >= 3 else 0
        score += mtf_pts
        breakdown["mtf_confluence"] = mtf_pts

        # 9. Supertrend agrees (+5) — NEW from ZPayab Pine Script port
        supertrend_agrees = metrics.get("supertrend_agrees", False)
        st_pts = 5 if supertrend_agrees else 0
        score += st_pts
        breakdown["supertrend_agrees"] = st_pts

        # 10. RQK Kernel agrees (+5) — NEW from ZPayab Pine Script port
        rqk_agrees = metrics.get("rqk_agrees", False)
        rqk_pts = 5 if rqk_agrees else 0
        score += rqk_pts
        breakdown["rqk_agrees"] = rqk_pts

        # ── MOMENTUM GATES (15 pts) ───────────────────────────

        # 11. Range Filter not CHOPPY (+8) — NEW from ZPayab Pine Script port
        range_filter_ok = metrics.get("range_filter_ok", False)
        rf_pts = 8 if range_filter_ok else 0
        score += rf_pts
        breakdown["range_filter_ok"] = rf_pts

        # 12. WAE explosion confirmed (+7) — NEW from ZPayab Pine Script port
        wae_confirms = metrics.get("wae_confirms", False)
        wae_pts = 7 if wae_confirms else 0
        score += wae_pts
        breakdown["wae_confirms"] = wae_pts

        # ── DECISION ─────────────────────────────────────────

        can_trade = score >= self.min_trade_threshold

        status = "IGNORE"
        if can_trade:
            status = "EXECUTE"
        elif score >= self.min_log_threshold:
            status = "LOG_ONLY"

        if status == "EXECUTE":
            logger.info(
                f"✅ [SCORER] {direction} PASS — {score}/100 "
                f"| RF={breakdown['range_filter_ok']} "
                f"| RQK={breakdown['rqk_agrees']} "
                f"| WAE={breakdown['wae_confirms']} "
                f"| ST={breakdown['supertrend_agrees']}"
            )
        elif status == "LOG_ONLY":
            logger.warning(
                f"⚠️ [SCORER] {direction} NEAR MISS — {score}/100 (need {self.min_trade_threshold})"
            )
        else:
            logger.debug(f"❌ [SCORER] {direction} IGNORED — {score}/100")

        return {
            "score"     : score,
            "breakdown" : breakdown,
            "status"    : status,
            "direction" : direction,
        }


global_signal_scorer = SignalScorer()
