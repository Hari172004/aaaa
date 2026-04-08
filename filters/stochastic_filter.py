"""
stochastic_filter.py — Stochastic Oscillator Filter (Ported from ZPayab Pine Script)
=====================================================================================
Supports three modes:
  - "crossover"       : K crosses D at any level
  - "ob_os_crossover" : K crosses D only within overbought/oversold zones
  - "k_vs_d"          : K is simply above or below D

Original Pine Script logic by ZPayab — translated to Python/Pandas.
"""

import logging
import pandas as pd

logger = logging.getLogger("agniv.filters.stochastic")


class StochasticFilter:
    """
    Stochastic Oscillator confirmation filter.
    """

    def __init__(
        self,
        length:      int = 14,
        smooth_k:    int = 3,
        smooth_d:    int = 3,
        overbought:  int = 80,
        oversold:    int = 20,
        mode: str = "crossover",  # "crossover" | "ob_os_crossover" | "k_vs_d"
    ):
        self.length     = length
        self.smooth_k   = smooth_k
        self.smooth_d   = smooth_d
        self.overbought = overbought
        self.oversold   = oversold
        self.mode       = mode

    def evaluate(self, df: pd.DataFrame) -> dict:
        """
        Returns:
          { "long": bool, "short": bool, "k": float, "d": float }
        """
        min_bars = self.length + self.smooth_k + self.smooth_d + 5
        empty = {"long": False, "short": False, "k": 50.0, "d": 50.0}
        if len(df) < min_bars:
            return empty

        low_min  = df["low"].rolling(self.length).min()
        high_max = df["high"].rolling(self.length).max()
        stoch    = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, float("nan"))
        k        = stoch.rolling(self.smooth_k).mean()
        d        = k.rolling(self.smooth_d).mean()

        k_cur  = float(k.iloc[-1])
        k_prev = float(k.iloc[-2])
        d_cur  = float(d.iloc[-1])
        d_prev = float(d.iloc[-2])

        if self.mode == "ob_os_crossover":
            cross_up   = k_prev < d_prev and k_cur > d_cur
            cross_down = k_prev > d_prev and k_cur < d_cur
            is_long    = cross_up   and k_prev < self.oversold  and k_cur > self.oversold
            is_short   = cross_down and k_prev > self.overbought and k_cur < self.overbought
        elif self.mode == "k_vs_d":
            is_long  = k_cur > d_cur
            is_short = k_cur < d_cur
        else:  # crossover
            is_long  = k_prev < d_prev and k_cur > d_cur
            is_short = k_prev > d_prev and k_cur < d_cur

        logger.debug(f"[Stoch] K={k_cur:.1f} D={d_cur:.1f} long={is_long} short={is_short}")
        return {"long": is_long, "short": is_short, "k": k_cur, "d": d_cur}


# Singleton instances
stoch_filter_scalp = StochasticFilter(length=14, mode="ob_os_crossover")
stoch_filter_swing = StochasticFilter(length=14, mode="crossover")
