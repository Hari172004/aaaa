"""
range_filter.py — Range Filter (Ported from ZPayab Pine Script)
================================================================
Eliminates choppy/sideways market noise by requiring price to
move a meaningful distance before a new signal is registered.

Original Pine Script logic by ZPayab — translated to Python/Pandas.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("agniv.filters.range_filter")


class RangeFilter:
    """
    Smoothed Range Filter — filters out low-momentum signals.

    How it works:
      1. Calculate a smoothed average range (EMA of ATR).
      2. Build a 'filter' line that only moves when price
         breaks out by that range distance.
      3. If price stays within the range band → CHOPPY → block trade.
      4. If price breaks out and range filter is rising → UPWARD trend.
      5. If price breaks out and range filter is falling → DOWNWARD trend.
    """

    def __init__(self, period: int = 100, multiplier: float = 3.0):
        """
        period     : Lookback window for smoothing (default 100 — higher = smoother)
        multiplier : Range size scaling factor (default 3.0)
        """
        self.period = period
        self.multiplier = multiplier

    def _smooth_range(self, close: pd.Series) -> pd.Series:
        """EMA of |price change|, smoothed again — gives the adaptive range size."""
        period = self.period
        mult = self.multiplier

        # Short EMA of absolute change
        avg_rng = close.diff().abs().ewm(span=period, adjust=False).mean()
        # Double-smooth: EMA of the EMA (wper = 2*period - 1)
        smrng = avg_rng.ewm(span=(period * 2 - 1), adjust=False).mean() * mult
        return smrng

    def _build_filter_line(self, close: pd.Series, smrng: pd.Series) -> pd.Series:
        """
        Builds the range filter line. It only moves when close breaks
        beyond the current filter by more than smrng.
        """
        filt = np.zeros(len(close))
        filt[0] = close.iloc[0]

        for i in range(1, len(close)):
            prev = filt[i - 1]
            c = close.iloc[i]
            r = smrng.iloc[i]

            if c - r > prev:
                filt[i] = c - r
            elif c + r < prev:
                filt[i] = c + r
            else:
                filt[i] = prev

        return pd.Series(filt, index=close.index)

    def evaluate(self, df: pd.DataFrame) -> dict:
        """
        Main entry point. Analyses the dataframe and returns:
          {
            "trend"    : "UP" | "DOWN" | "CHOPPY",
            "safe"     : bool,
            "filt"     : float  — current filter line value,
            "upward"   : bool,
            "downward" : bool,
          }
        """
        min_bars = self.period * 2 + 5
        if len(df) < min_bars:
            return {"trend": "CHOPPY", "safe": False, "filt": 0.0,
                    "upward": False, "downward": False}

        close = df["close"]
        smrng = self._smooth_range(close)
        filt  = self._build_filter_line(close, smrng)

        # Direction counters (same as Pine Script)
        upward   = 0.0
        downward = 0.0

        for i in range(1, len(filt)):
            if filt.iloc[i] > filt.iloc[i - 1]:
                upward   = upward + 1
                downward = 0.0
            elif filt.iloc[i] < filt.iloc[i - 1]:
                downward = downward + 1
                upward   = 0.0

        last_close = close.iloc[-1]
        last_filt  = filt.iloc[-1]
        prev_close = close.iloc[-2]

        rf_upward   = (last_close > last_filt and last_close > prev_close and upward > 0) or \
                      (last_close > last_filt and last_close < prev_close and upward > 0)
        rf_downward = (last_close < last_filt and last_close < prev_close and downward > 0) or \
                      (last_close < last_filt and last_close > prev_close and downward > 0)

        if rf_upward:
            trend = "UP"
        elif rf_downward:
            trend = "DOWN"
        else:
            trend = "CHOPPY"

        safe = trend != "CHOPPY"

        logger.debug(
            f"[RangeFilter] trend={trend} | filt={last_filt:.2f} | "
            f"close={last_close:.2f} | upward={upward} | downward={downward}"
        )

        return {
            "trend"    : trend,
            "safe"     : safe,
            "filt"     : float(last_filt),
            "upward"   : rf_upward,
            "downward" : rf_downward,
        }


# Singleton instance for Gold
gold_range_filter = RangeFilter(period=100, multiplier=3.0)
