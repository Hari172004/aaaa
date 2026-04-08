"""
macd_filter.py — MACD Filter (Ported from ZPayab Pine Script)
=============================================================
Supports two modes:
  - "crossover"    : MACD line crossover/crossunder of the Signal line
  - "zero_line"    : MACD line crossover/crossunder of the zero line

Original Pine Script logic by ZPayab — translated to Python/Pandas.
"""

import logging
import pandas as pd

logger = logging.getLogger("agniv.filters.macd")


class MACDFilter:
    """
    MACD signal filter.

    Modes:
      crossover  → long if MACD > signal, short if MACD < signal
      zero_line  → long if MACD > signal AND MACD > 0, short otherwise
    """

    def __init__(
        self,
        fast_length:   int = 12,
        slow_length:   int = 26,
        signal_length: int = 9,
        mode: str = "crossover",   # "crossover" | "zero_line"
    ):
        self.fast_length   = fast_length
        self.slow_length   = slow_length
        self.signal_length = signal_length
        self.mode          = mode

    def evaluate(self, df: pd.DataFrame) -> dict:
        """
        Returns:
          { "long": bool, "short": bool, "macd": float, "signal": float, "hist": float }
        """
        min_bars = self.slow_length + self.signal_length + 5
        empty = {"long": False, "short": False, "macd": 0.0, "signal": 0.0, "hist": 0.0}
        if len(df) < min_bars:
            return empty

        close  = df["close"]
        fast   = close.ewm(span=self.fast_length,   adjust=False).mean()
        slow   = close.ewm(span=self.slow_length,   adjust=False).mean()
        macd   = fast - slow
        signal = macd.ewm(span=self.signal_length,  adjust=False).mean()
        hist   = macd - signal

        m = float(macd.iloc[-1])
        s = float(signal.iloc[-1])
        h = float(hist.iloc[-1])

        if self.mode == "zero_line":
            is_long  = m > s and m > 0.0
            is_short = m < s and m < 0.0
        else:  # crossover
            is_long  = m > s
            is_short = m < s

        logger.debug(f"[MACD] macd={m:.5f} signal={s:.5f} long={is_long} short={is_short}")
        return {"long": is_long, "short": is_short, "macd": m, "signal": s, "hist": h}


# Singleton instances
macd_filter_scalp = MACDFilter(fast_length=12, slow_length=26, signal_length=9, mode="crossover")
macd_filter_swing = MACDFilter(fast_length=12, slow_length=26, signal_length=9, mode="zero_line")
