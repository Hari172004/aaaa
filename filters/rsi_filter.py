"""
rsi_filter.py — RSI Filter (Ported from ZPayab Pine Script)
============================================================
Supports three modes:
  - "ma_cross"    : RSI crossover/crossunder of its MA
  - "ob_os_exit"  : RSI exits overbought/oversold zones
  - "level"       : RSI vs a fixed midline level

Original Pine Script logic by ZPayab — translated to Python/Pandas.
"""

import logging
import pandas as pd

logger = logging.getLogger("agniv.filters.rsi")


class RSIFilter:
    """
    RSI-based signal confirmation/filter.

    Modes:
      ma_cross   → long when RSI crosses above its MA, short when crosses below
      ob_os_exit → long when RSI crosses UP from oversold; short when crosses DOWN from overbought
      level      → long when RSI > midline; short when RSI < midline
    """

    def __init__(
        self,
        rsi_length:   int   = 14,
        ma_length:    int   = 14,
        overbought:   int   = 80,
        oversold:     int   = 20,
        midline:      int   = 50,
        long_limit:   int   = 40,     # RSI must be >= this to allow long
        short_limit:  int   = 60,     # RSI must be <= this to allow short
        use_limits:   bool  = False,
        mode: str = "ma_cross",       # "ma_cross" | "ob_os_exit" | "level"
    ):
        self.rsi_length  = rsi_length
        self.ma_length   = ma_length
        self.overbought  = overbought
        self.oversold    = oversold
        self.midline     = midline
        self.long_limit  = long_limit
        self.short_limit = short_limit
        self.use_limits  = use_limits
        self.mode        = mode

    def _rsi(self, close: pd.Series) -> pd.Series:
        delta  = close.diff()
        gain   = delta.clip(lower=0).ewm(alpha=1/self.rsi_length, adjust=False).mean()
        loss   = (-delta.clip(upper=0)).ewm(alpha=1/self.rsi_length, adjust=False).mean()
        rs     = gain / loss.replace(0, float("nan"))
        return 100 - (100 / (1 + rs))

    def evaluate(self, df: pd.DataFrame) -> dict:
        """
        Returns:
          { "long": bool, "short": bool, "rsi": float, "rsi_ma": float }
        """
        min_bars = self.rsi_length + self.ma_length + 5
        empty = {"long": False, "short": False, "rsi": 50.0, "rsi_ma": 50.0}
        if len(df) < min_bars:
            return empty

        close   = df["close"]
        rsi     = self._rsi(close)
        rsi_ma  = rsi.rolling(self.ma_length).mean()

        cur_rsi  = float(rsi.iloc[-1])
        prev_rsi = float(rsi.iloc[-2])
        cur_ma   = float(rsi_ma.iloc[-1])
        prev_ma  = float(rsi_ma.iloc[-2])

        if self.mode == "ob_os_exit":
            is_long  = cur_rsi > self.oversold  and prev_rsi <= self.oversold
            is_short = cur_rsi < self.overbought and prev_rsi >= self.overbought
        elif self.mode == "level":
            is_long  = cur_rsi > self.midline
            is_short = cur_rsi < self.midline
        else:  # ma_cross
            is_long  = cur_rsi > cur_ma
            is_short = cur_rsi < cur_ma

        # Apply optional hard limits
        if self.use_limits:
            if is_long  and cur_rsi < self.long_limit:
                is_long = False
            if is_short and cur_rsi > self.short_limit:
                is_short = False

        logger.debug(f"[RSI] rsi={cur_rsi:.1f} ma={cur_ma:.1f} long={is_long} short={is_short}")
        return {"long": is_long, "short": is_short, "rsi": cur_rsi, "rsi_ma": cur_ma}


# Singleton instances
rsi_filter_scalp = RSIFilter(rsi_length=14, mode="ma_cross")
rsi_filter_swing = RSIFilter(rsi_length=14, mode="level", midline=50)
