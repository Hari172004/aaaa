"""
btc_swing.py — BTC Swing Strategy (1H / 4H / Daily)
==================================================
Longer-term triggers based on:
1. Trend (EMA 50/200)
2. S/R and SMC (BOS, CHOCH)
3. Ichimoku Cloud agreement
4. On-chain health score integration
"""

import logging
import pandas as pd # type: ignore
from analysis.btc_indicators import BTCIndicators # type: ignore
from analysis.btc_market_structure import BTCMarketStructure # type: ignore
from analysis.btc_onchain import BTCOnChain # type: ignore

logger = logging.getLogger("apexalgo.btc_swing")

class BTCSwingStrategy:
    """Swing trading for BTC."""

    def __init__(self, risk_reward: float = 3.0):
        self.risk_reward = risk_reward
        self.onchain = BTCOnChain()

    def generate_signal(self, df: pd.DataFrame, is_nano: bool = False, ignore_sessions: bool = False, is_sniper: bool = False) -> dict:
        """
        Analyzes 1H/4H/Daily data with Sniper consistency.
        """
        empty = {"signal": "HOLD", "strength": 0.0, "reason": "No Setup", "atr": 0.0, "sl": 0.0, "tp": 0.0}
        
        if len(df) < 200:
            return {**empty, "reason": "Insufficient history (200 candles required)"}

        df = BTCIndicators.add_all_indicators(df)
        smc = BTCMarketStructure.detect_structure(df)
        health_score = self.onchain.get_health_score()
        
        row = df.iloc[-1]
        close = float(row["close"])
        ema50 = float(row["ema_50"])
        ema200 = float(row["ema_200"])
        atr = float(row["atr"])

        # Sniper Bias Check (Trend Direction)
        trend_up   = ema50 > ema200 and close > ema200
        trend_down = ema50 < ema200 and close < ema200

        buy_score = 0
        sell_score = 0
        reasons = []

        # ── 1. Trend Direction ────────────────────────────────
        if trend_up:
            buy_score += 2
            reasons.append("Bullish Trend Alignment")
        elif trend_down:
            sell_score += 2
            reasons.append("Bearish Trend Alignment")

        # ── 2. Market Structure ──────────────────────────────
        if smc["bos"] == "BULLISH":
            buy_score += 2
            reasons.append("Bullish BOS Detected")
        elif smc["bos"] == "BEARISH":
            sell_score += 2
            reasons.append("Bearish BOS Detected")

        # ── 3. On-Chain Health ───────────────────────────────
        if health_score > 0.7:
            buy_score += 1
            reasons.append("Positive On-Chain Health")
        elif health_score < 0.3:
            sell_score += 1
            reasons.append("Weak On-Chain Health")

        # ── Decision ──────────────────────────────────────────
        signal = "HOLD"
        strength = 0.0
        
        if buy_score >= 4 and buy_score > sell_score:
            if is_sniper and not trend_up: return {**empty, "reason": "Sniper BTC Swing: Counter-trend inhibited"}
            signal = "BUY"
            strength = buy_score / 6.0
        elif sell_score >= 4 and sell_score > buy_score:
            if is_sniper and not trend_down: return {**empty, "reason": "Sniper BTC Swing: Counter-trend inhibited"}
            signal = "SELL"
            strength = sell_score / 6.0

        # SL and TP
        if signal != "HOLD":
            sl = close - (1.5 * atr) if signal == "BUY" else close + (1.5 * atr)
            tp = close + (self.risk_reward * atr) if signal == "BUY" else close - (self.risk_reward * atr)
            
            return {
                "signal": signal,
                "strength": float(int(min(strength, 1.0) * 100) / 100.0),
                "reason": ", ".join(reasons),
                "atr": atr,
                "sl": float(f"{sl:.2f}"),
                "tp": float(f"{tp:.2f}")
            }

        return {**empty, "reason": "Waiting for Trend & Structure alignment"}
