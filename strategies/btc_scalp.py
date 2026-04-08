"""
btc_scalp.py — BTC Scalping Strategy (1m / 5m)
=============================================
High-frequency triggers based on:
1. EMA 9/21 cross
2. RSI 40-60 level
3. Volume spikes
4. SMC (FVG/OB) alignment
5. [NEW] Range Filter — choppy market gate
6. [NEW] RQK Kernel   — trend bias confirmation
7. [NEW] WAE          — momentum explosion gate
8. [NEW] Supertrend   — HTF trend direction gate
"""

import logging
from datetime import datetime
import pandas as pd # type: ignore
from analysis.btc_indicators import BTCIndicators # type: ignore
from analysis.btc_market_structure import BTCMarketStructure # type: ignore

try:
    from rl.ppo_agent import PPOAgent  # type: ignore
    _PPO_AVAILABLE = True
except ImportError:
    _PPO_AVAILABLE = False

# ── New filters ported from ZPayab Pine Script ───────────────────
try:
    from filters.range_filter   import btc_range_filter   # type: ignore
    from filters.rqk_filter     import btc_rqk_filter     # type: ignore
    from filters.wae_filter     import btc_wae_filter     # type: ignore
    from filters.supertrend_filter import btc_supertrend  # type: ignore
    from filters.bullbyte_engine   import BullByteEngine  # type: ignore
    _NEW_FILTERS_AVAILABLE = True
except ImportError:
    _NEW_FILTERS_AVAILABLE = False

logger = logging.getLogger("agniv.btc_scalp")

class BTCScalpStrategy:
    """Fast scalping for BTC."""

    def __init__(self, risk_reward: float = 2.0):
        self.risk_reward = risk_reward
        # Minimum signal confidence required to open a trade
        self.min_strength = 0.82
        # PPO Reinforcement Learning agent (lazy-loaded)
        self._ppo: "PPOAgent | None" = PPOAgent("BTCUSD") if _PPO_AVAILABLE else None  # type: ignore
        # New filter gate (requires ≥2 of 4 external filters to agree before signal passes)
        self.filter_gate_min = 2

    def generate_signal(self, df: pd.DataFrame, df_h1: pd.DataFrame = None, 
                        is_nano: bool = False, ignore_sessions: bool = False, is_sniper: bool = False) -> dict:
        """
        BTC Scalping with Sniper precision.
        """
        empty = {"signal": "HOLD", "strength": 0.0, "reason": "No setup", "atr": 0.0, "sl": 0.0, "tp": 0.0}

        if len(df) < 50:
            return {**empty, "reason": "Insufficient data"}

        row = df.iloc[-1]
        prev_row = df.iloc[-2]
        live_close_val = float(row["close"])

        # 1. HTF Trend Alignment
        h1_bullish, h1_bearish = True, True
        if df_h1 is not None and not df_h1.empty:
            df_h1 = BTCIndicators.add_all_indicators(df_h1)
            h1_last = df_h1.iloc[-1]
            ema100_h1 = h1_last.get("ema_100", h1_last.get("ema_200", 0))
            h1_bullish = float(h1_last["close"]) > ema100_h1
            h1_bearish = float(h1_last["close"]) < ema100_h1

        # 2. Indicators & SMC (ICT Sniper V2.0)
        df = BTCIndicators.add_all_indicators(df)
        
        # Refresh snapshots after indicators are added
        row = df.iloc[-1]
        prev_row = df.iloc[-2]

        rsi   = float(row.get("rsi", 50))
        atr   = float(row.get("atr", 0))

        from strategies.smc import SMCEngine # type: ignore
        smc_v2 = SMCEngine.get_smc_context(df, live_close_val)
        
        # Ensure required columns are present (prevents KeyError if TA fails)
        required = ["ema_9", "ema_21", "ema_50", "rsi", "atr"]
        if not (all(col in row for col in required) and all(col in prev_row for col in required)):
             return {**empty, "reason": "BTCScalp: Waiting for indicators to stabilise"}

        # BTC trades 24/7 — no session restrictions apply

        # 4. Momentum: 50 EMA + RSI (M1/M5)
        ema50 = float(row.get("ema_50", 0))

        # Instantaneous crossover
        rsi_bullish = rsi > 50 and prev_row.get("rsi", 50) <= 50
        rsi_bearish = rsi < 50 and prev_row.get("rsi", 50) >= 50
        momentum_scalp_up   = live_close_val > ema50 and rsi_bullish
        momentum_scalp_down = live_close_val < ema50 and rsi_bearish

        # Sustained trend — mid-session entry when bot starts after the cross
        momentum_sustained_up   = rsi > 55 and live_close_val > ema50
        momentum_sustained_down = rsi < 45 and live_close_val < ema50

        # Use only CLOSED candles (all except the live partial bar) for the average.
        # df.iloc[-1] is the current in-progress candle; its volume is always tiny,
        # which caused RVOL to read ~0.1 and permanently block every signal.
        closed_vols = df["volume"].iloc[-21:-1]   # last 20 fully-closed bars
        vol_avg     = closed_vols.mean() if len(closed_vols) > 0 else 0.0
        raw_vol     = float(row.get("volume", 0))
        # If vol data is missing/zero from the feed, treat as neutral (don't block)
        vol_data_valid = vol_avg > 0 and raw_vol > 0
        rvol = (raw_vol / vol_avg) if vol_data_valid else 1.0
        
        # ── ICT/SMC Triggers ──────────────────
        from analysis.btc_market_structure import BTCMarketStructure # type: ignore
        smc_legacy = BTCMarketStructure.detect_structure(df)
        
        def near_fvg_btc(price, fvgs):
            for f in fvgs:
                if f["bottom"] <= price <= f["top"]: return True
            return False

        ict_buy_setup = smc_v2["bullish_sweep"] and smc_v2["displacement"] and near_fvg_btc(live_close_val, smc_legacy.get("fvgs", []))
        ict_sell_setup = smc_v2["bearish_sweep"] and smc_v2["displacement"] and near_fvg_btc(live_close_val, smc_legacy.get("fvgs", []))

        # ── Channel & Heiken Ashi Breakout ─────
        ema_55_high = float(row.get("ema_55_high", 0))
        ema_55_low  = float(row.get("ema_55_low", 0))
        ha_bull     = bool(row.get("ha_bull", False))
        
        channel_break_up = live_close_val > ema_55_high and ha_bull
        channel_break_dn = live_close_val < ema_55_low and not ha_bull
        in_channel = ema_55_low <= live_close_val <= ema_55_high

        if is_sniper and in_channel:
             return {**empty, "reason": "Sniper BTC: Price inside 55-MA Channel (Choppy)"}

        # ── Trigger Selection ─────────────────
        
        signal   = "HOLD"
        strength = 0.0
        reasons  = []
        hold_reasons = []

        # BUY Logic
        ema_cross_up = row["ema_9"] > row["ema_21"] and prev_row["ema_9"] <= prev_row["ema_21"]
        # Sustained: EMA9 above EMA21 for ≥3 consecutive closed candles
        last3 = df.iloc[-4:-1]
        ema_sustained_up = all(last3.iloc[i]["ema_9"] > last3.iloc[i]["ema_21"] for i in range(len(last3))) if len(last3) >= 3 else False
        ema_sustained_dn = all(last3.iloc[i]["ema_9"] < last3.iloc[i]["ema_21"] for i in range(len(last3))) if len(last3) >= 3 else False
        # BULLBYTE ULTIMATE SCALPER
        try:
            bb_eval = BullByteEngine.evaluate(df)
            bb_buy = bb_eval.get("buy", False)
            bb_sell = bb_eval.get("sell", False)
        except Exception:
            bb_buy, bb_sell = False, False

        is_buy_trigger = ict_buy_setup or momentum_scalp_up or ema_cross_up or channel_break_up or \
                         (ema_sustained_up and momentum_sustained_up) or bb_buy
        
        if is_buy_trigger:
            if not h1_bullish: hold_reasons.append("H1 Trend Bearish")
            if vol_data_valid and rvol < 1.1: hold_reasons.append(f"Low RVOL ({rvol:.1f})")
            if is_sniper and not ha_bull: hold_reasons.append("HA Candle Red")
            
            if not hold_reasons:
                if is_sniper or is_nano:
                    in_bull_ob = any(ob["type"] == "BULL_OB" and ob["low"] <= live_close_val <= ob["high"] for ob in smc_legacy["obs"])
                    if not (ict_buy_setup or in_bull_ob or channel_break_up):
                         return {**empty, "reason": "Sniper BTC: Waiting for ICT Sweep, OB or HA Breakout"}

                signal   = "BUY"
                strength = 0.75
                if bb_buy:                strength += 0.25; reasons.append("BullByte Ultimate (Kinetic+Memory)")
                elif ict_buy_setup:       strength += 0.15; reasons.append("ICT Sweep+FVG")
                elif channel_break_up:    strength += 0.10; reasons.append("HA Breakout")
                elif momentum_scalp_up:   strength += 0.07; reasons.append("EMA+RSI Cross")
                elif ema_sustained_up:    strength += 0.03; reasons.append("EMA+RSI Sustained")
                
                if h1_bullish: strength += 0.10; reasons.append("H1 Trend+")
                if rvol > 1.5: strength += 0.05; reasons.append("Vol Spike")

        # SELL Logic
        if signal == "HOLD":
            ema_cross_down = row["ema_9"] < row["ema_21"] and prev_row["ema_9"] >= prev_row["ema_21"]
            is_sell_trigger = ict_sell_setup or momentum_scalp_down or ema_cross_down or channel_break_dn or \
                              (ema_sustained_dn and momentum_sustained_down) or bb_sell
            
            if is_sell_trigger:
                if not h1_bearish: hold_reasons.append("H1 Trend Bullish")
                if vol_data_valid and rvol < 1.1: hold_reasons.append(f"Low RVOL ({rvol:.1f})")
                if is_sniper and ha_bull: hold_reasons.append("HA Candle Green")

                if not hold_reasons:
                    if is_sniper or is_nano:
                        in_bear_ob = any(ob["type"] == "BEAR_OB" and ob["low"] <= live_close_val <= ob["high"] for ob in smc_legacy["obs"])
                        if not (ict_sell_setup or in_bear_ob or channel_break_dn):
                             return {**empty, "reason": "Sniper BTC: Waiting for ICT Sweep, OB or HA Breakout"}

                    signal   = "SELL"
                    strength = 0.75
                    if bb_sell:               strength += 0.25; reasons.append("BullByte Ultimate (Kinetic+Memory)")
                    elif ict_sell_setup:      strength += 0.15; reasons.append("ICT Sweep+FVG")
                    elif channel_break_dn:    strength += 0.10; reasons.append("HA Breakout")
                    elif momentum_scalp_down: strength += 0.07; reasons.append("EMA+RSI Cross")
                    elif ema_sustained_dn:    strength += 0.03; reasons.append("EMA+RSI Sustained")
                    
                    if h1_bearish: strength += 0.10; reasons.append("H1 Trend-")
                    if rvol > 1.5: strength += 0.05; reasons.append("Vol Spike")

        # Final Calculation
        if signal != "HOLD":
            sl = live_close_val - (1.6 * atr) if signal == "BUY" else live_close_val + (1.6 * atr)
            tp = live_close_val + (self.risk_reward * 1.6 * atr) if signal == "BUY" else live_close_val - (self.risk_reward * 1.6 * atr)

            # ── New Filter Gate (Range Filter + RQK + WAE + Supertrend) ──
            if _NEW_FILTERS_AVAILABLE:
                is_buy = signal == "BUY"
                filter_votes = 0
                filter_log   = []

                # 1. Range Filter — block if market is choppy
                rf = btc_range_filter.evaluate(df)
                rf_ok = (rf["upward"] if is_buy else rf["downward"])
                if rf_ok:
                    filter_votes += 1
                    filter_log.append("RF✅")
                else:
                    filter_log.append(f"RF❌({rf['trend']})")

                # 2. RQK Kernel — trend bias must agree
                rqk = btc_rqk_filter.evaluate(df)
                rqk_ok = (rqk["bullish"] if is_buy else rqk["bearish"])
                if rqk_ok:
                    filter_votes += 1
                    filter_log.append("RQK✅")
                else:
                    filter_log.append(f"RQK❌({rqk['trend']})")

                # 3. WAE — momentum explosion must be confirmed
                wae = btc_wae_filter.evaluate(df)
                wae_ok = (wae["safe_buy"] if is_buy else wae["safe_sell"])
                if wae_ok:
                    filter_votes += 1
                    filter_log.append("WAE✅")
                else:
                    filter_log.append("WAE❌(no explosion)")

                # 4. Supertrend — HTF trend must agree
                st = btc_supertrend.evaluate(df)
                st_ok = (st["safe_buy"] if is_buy else st["safe_sell"])
                if st_ok:
                    filter_votes += 1
                    filter_log.append("ST✅")
                else:
                    filter_log.append(f"ST❌({st['trend']})")

                filter_summary = " | ".join(filter_log)
                logger.info(
                    f"[BTCScalp] Filter Gate: {filter_votes}/4 votes | "
                    f"{filter_summary} | Signal={signal}"
                )

                if filter_votes < self.filter_gate_min:
                    return {
                        **empty,
                        "reason": f"Filter Gate Failed ({filter_votes}/4): {filter_summary}"
                    }

                # Boost strength for each confirming filter
                strength = min(strength + (filter_votes * 0.03), 1.0)

            # ── PPO Confirmation Filter ───────────────────────────
            if self._ppo is not None and self._ppo.is_available():
                ema9_val  = float(row.get("ema_9",  0))
                ema21_val = float(row.get("ema_21", 0))
                atr_raw   = float(row.get("atr", 1))
                obs_dict  = {
                    "rsi":            rsi / 100.0,
                    "ema_diff_pct":   (ema9_val - ema21_val) / (live_close_val + 1e-9),
                    "atr_norm":       atr_raw / (live_close_val + 1e-9),
                    "rvol":           min(rvol, 5.0),
                    "macd_hist_norm": 0.0,
                    "bb_pct":         0.5,
                    "ha_bull":        1.0 if ha_bull else 0.0,
                    "h1_trend":       1.0 if h1_bullish else 0.0,
                    "session_id":     0.5,
                    "close_norm":     0.0,
                }
                ppo_result = self._ppo.predict(obs_dict)
                ppo_action = ppo_result.get("action", "HOLD")
                ppo_conf   = ppo_result.get("confidence", 1.0)

                if ppo_action == signal:
                    strength = min(strength + 0.10, 1.0)
                    reasons.append(f"PPO:Confirm({ppo_conf:.0%})")
                elif ppo_action == "HOLD":
                    strength = max(strength - 0.10, 0.0)
                    reasons.append("PPO:Uncertain")
                else:
                    strength = max(strength - 0.30, 0.0)
                    reasons.append(f"PPO:Disagree({ppo_action})")
                    if strength < 0.60:
                        return {**empty, "reason": "PPO Override: " + ", ".join(reasons)}

            # ── Minimum Strength Gate ─────────────────────────────────────
            strength = float(round(float(min(strength, 1.0)), 3))
            if strength < self.min_strength:
                logger.info(
                    f"[BTCScalp] Low confidence {signal} rejected: "
                    f"{strength:.0%} < {self.min_strength:.0%} minimum. Reasons: {', '.join(reasons)}"
                )
                return {**empty, "reason": f"Low confidence ({strength:.0%}) — needs {self.min_strength:.0%}"}

            return {
                "signal":   signal,
                "strength": strength,
                "reason":   ", ".join(reasons),
                "atr":      atr,
                "sl":       float(f"{sl:.2f}"),
                "tp":       float(f"{tp:.2f}"),
                "rsi":      rsi,
                "rvol":     rvol
            }

        return {**empty, "reason": "BTC Searching: " + (", ".join(hold_reasons) if hold_reasons else "No Setup")}

