"""
btc_scalp.py — BTC Scalping Strategy (1m / 5m)
=============================================
High-frequency triggers based on:
1. EMA 9/21 cross
2. RSI 40-60 level
3. Volume spikes
4. SMC (FVG/OB) alignment
"""

import logging
from datetime import datetime, timezone
import pandas as pd # type: ignore
from analysis.btc_indicators import BTCIndicators # type: ignore
from analysis.btc_market_structure import BTCMarketStructure # type: ignore

logger = logging.getLogger("apexalgo.btc_scalp")

class BTCScalpStrategy:
    """Fast scalping for BTC."""

    def __init__(self, risk_reward: float = 2.0):
        self.risk_reward = risk_reward

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

        # 3. Filter: 13:00 - 17:00 GMT (NY Session) for Sniper precision
        is_ny_kill_zone = 13 <= datetime.now(timezone.utc).hour <= 17
        if is_sniper and not is_ny_kill_zone:
             return {**empty, "reason": "Sniper BTC: Outside High-Liquidity Window (13-17 GMT)"}

        # 4. Momentum: 50 EMA + RSI (M1/M5)
        ema50 = float(row.get("ema_50", 0))
        rsi_bullish = rsi > 50 and prev_row.get("rsi", 50) <= 50
        rsi_bearish = rsi < 50 and prev_row.get("rsi", 50) >= 50
        momentum_scalp_up = close > ema50 and rsi_bullish
        momentum_scalp_down = close < ema50 and rsi_bearish

        vol_avg = df["volume"].tail(20).mean()
        rvol = row["volume"] / vol_avg if vol_avg > 0 else 1.0
        
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
        is_buy_trigger = ict_buy_setup or momentum_scalp_up or ema_cross_up or channel_break_up
        
        if is_buy_trigger:
            if not h1_bullish: hold_reasons.append("H1 Trend Bearish")
            if rvol < 1.1:           hold_reasons.append(f"Low RVOL ({rvol:.1f})")
            if is_sniper and not ha_bull: hold_reasons.append("HA Candle Red")
            
            if not hold_reasons:
                if is_sniper or is_nano:
                    in_bull_ob = any(ob["type"] == "BULL_OB" and ob["low"] <= live_close_val <= ob["high"] for ob in smc_legacy["obs"])
                    if not (ict_buy_setup or in_bull_ob or channel_break_up):
                         return {**empty, "reason": "Sniper BTC: Waiting for ICT Sweep, OB or HA Breakout"}

                signal   = "BUY"
                strength = 0.75
                if ict_buy_setup: strength += 0.15; reasons.append("ICT Sweep+FVG")
                elif channel_break_up: strength += 0.10; reasons.append("HA Breakout")
                elif momentum_scalp_up: strength += 0.05; reasons.append("EMA+RSI Scalp")
                
                if h1_bullish: strength += 0.10; reasons.append("H1 Trend+")
                if rvol > 1.5: strength += 0.05; reasons.append("Vol Spike")

        # SELL Logic
        if signal == "HOLD":
            ema_cross_down = row["ema_9"] < row["ema_21"] and prev_row["ema_9"] >= prev_row["ema_21"]
            is_sell_trigger = ict_sell_setup or momentum_scalp_down or ema_cross_down or channel_break_dn
            
            if is_sell_trigger:
                if not h1_bearish: hold_reasons.append("H1 Trend Bullish")
                if rvol < 1.1:           hold_reasons.append(f"Low RVOL ({rvol:.1f})")
                if is_sniper and ha_bull: hold_reasons.append("HA Candle Green")

                if not hold_reasons:
                    if is_sniper or is_nano:
                        in_bear_ob = any(ob["type"] == "BEAR_OB" and ob["low"] <= live_close_val <= ob["high"] for ob in smc_legacy["obs"])
                        if not (ict_sell_setup or in_bear_ob or channel_break_dn):
                             return {**empty, "reason": "Sniper BTC: Waiting for ICT Sweep, OB or HA Breakout"}

                    signal   = "SELL"
                    strength = 0.75
                    if ict_sell_setup: strength += 0.15; reasons.append("ICT Sweep+FVG")
                    elif channel_break_dn: strength += 0.10; reasons.append("HA Breakout")
                    elif momentum_scalp_down: strength += 0.05; reasons.append("EMA+RSI Scalp")
                    
                    if h1_bearish: strength += 0.10; reasons.append("H1 Trend-")
                    if rvol > 1.5: strength += 0.05; reasons.append("Vol Spike")

        # Final Calculation
        if signal != "HOLD":
            sl = live_close_val - (1.6 * atr) if signal == "BUY" else live_close_val + (1.6 * atr)
            tp = live_close_val + (self.risk_reward * 1.6 * atr) if signal == "BUY" else live_close_val - (self.risk_reward * 1.6 * atr)

            return {
                "signal":   signal,
                "strength": float(round(float(min(strength, 1.0)), 3)),
                "reason":   ", ".join(reasons),
                "atr":      atr,
                "sl":       float(f"{sl:.2f}"),
                "tp":       float(f"{tp:.2f}"),
                "rsi":      rsi,
                "rvol":     rvol
            }

        return {**empty, "reason": "BTC Searching: " + (", ".join(hold_reasons) if hold_reasons else "No Setup")}
