"""
supertrend_filter.py — Supertrend Direction Filter
===================================================
Proper Supertrend implementation (ATR-based dynamic S/R).
Fires BUY when trend flips bullish, SELL when bearish.
Used as a HIGH-TIMEFRAME trend-direction gate.

Original logic from ZPayab Pine Script — translated to Python/Pandas.
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger("agniv.filters.supertrend")


class SupertrendFilter:
    """
    Supertrend — Dynamic Support/Resistance Trend Filter.

    How it works:
      Upper Band = HL/2 + multiplier * ATR  → acts as resistance in uptrend
      Lower Band = HL/2 - multiplier * ATR  → acts as support  in uptrend

    When close crosses above upper band → BULL trend starts.
    When close crosses below lower band → BEAR trend starts.

    Parameters
    ----------
    period     : ATR period (default 10)
    multiplier : ATR multiplier for band width (default 3.0)
    """

    def __init__(self, period: int = 10, multiplier: float = 3.0):
        self.period     = period
        self.multiplier = multiplier

    def _atr(self, df: pd.DataFrame) -> pd.Series:
        """Wilder's ATR."""
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - df["close"].shift(1)).abs()
        tr3 = (df["low"]  - df["close"].shift(1)).abs()
        tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(alpha=1.0 / self.period, adjust=False).mean()

    def evaluate(self, df: pd.DataFrame) -> dict:
        """
        Computes Supertrend on the full DataFrame.

        Returns:
          {
            "trend"       : "BULL" | "BEAR",
            "safe_buy"    : bool,
            "safe_sell"   : bool,
            "buy_signal"  : bool — trend just flipped bullish (entry signal),
            "sell_signal" : bool — trend just flipped bearish (entry signal),
            "support"     : float — current Supertrend support/resistance level,
          }
        """
        min_bars = self.period * 3
        if len(df) < min_bars:
            return {
                "trend": "BULL", "safe_buy": True, "safe_sell": False,
                "buy_signal": False, "sell_signal": False, "support": 0.0
            }

        close = df["close"].values
        high  = df["high"].values
        low   = df["low"].values

        atr_series = self._atr(df).values
        src = (high + low) / 2.0

        upper_band = src + self.multiplier * atr_series
        lower_band = src - self.multiplier * atr_series

        # Initialise arrays
        st_upper = upper_band.copy()
        st_lower = lower_band.copy()
        trend    = np.ones(len(close), dtype=int)   # 1 = BULL, -1 = BEAR
        supertrend_line = np.zeros(len(close))

        for i in range(1, len(close)):
            # Lower band: only move up (tighten from below)
            if lower_band[i] > st_lower[i - 1] or close[i - 1] < st_lower[i - 1]:
                st_lower[i] = lower_band[i]
            else:
                st_lower[i] = st_lower[i - 1]

            # Upper band: only move down (tighten from above)
            if upper_band[i] < st_upper[i - 1] or close[i - 1] > st_upper[i - 1]:
                st_upper[i] = upper_band[i]
            else:
                st_upper[i] = st_upper[i - 1]

            # Trend direction
            prev_trend = trend[i - 1]
            if prev_trend == -1 and close[i] > st_upper[i - 1]:
                trend[i] = 1   # flipped BULL
            elif prev_trend == 1 and close[i] < st_lower[i - 1]:
                trend[i] = -1  # flipped BEAR
            else:
                trend[i] = prev_trend

            supertrend_line[i] = st_lower[i] if trend[i] == 1 else st_upper[i]

        current_trend = trend[-1]
        prev_trend    = trend[-2]

        is_bull       = current_trend == 1
        is_bear       = current_trend == -1
        buy_signal    = (current_trend == 1  and prev_trend == -1)   # Just flipped bull
        sell_signal   = (current_trend == -1 and prev_trend == 1)    # Just flipped bear

        logger.debug(
            f"[Supertrend] trend={'BULL' if is_bull else 'BEAR'} | "
            f"support={supertrend_line[-1]:.4f} | "
            f"buy_signal={buy_signal} | sell_signal={sell_signal}"
        )

        return {
            "trend"       : "BULL" if is_bull else "BEAR",
            "safe_buy"    : is_bull,
            "safe_sell"   : is_bear,
            "buy_signal"  : buy_signal,
            "sell_signal" : sell_signal,
            "support"     : float(supertrend_line[-1]),
        }


# Gold Singleton
gold_supertrend = SupertrendFilter(period=10, multiplier=3.0)
