"""
adx_filter.py — ADX / DMI Filter (Ported from ZPayab Pine Script)
==================================================================
Directional Movement Index. Confirms that price is trending strongly
enough to trade, and identifies the direction.

Modes:
  "adx_only"     : ADX value must be above threshold
  "adx_di"       : ADX above threshold AND +DI vs -DI crossover (default)
  "advance"      : ADX cycle-aware — uses crossover cycles for more precise entries

Original Pine Script logic by ZPayab — translated to Python/Pandas.
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger("agniv.filters.adx")


class ADXFilter:
    """ADX / DMI trend strength and direction filter."""

    def __init__(
        self,
        di_length:  int   = 10,
        adx_length: int   = 5,
        key_level:  int   = 20,
        mode: str = "adx_di",  # "adx_only" | "adx_di" | "advance"
    ):
        self.di_length  = di_length
        self.adx_length = adx_length
        self.key_level  = key_level
        self.mode       = mode

    def _rma(self, series: pd.Series, period: int) -> pd.Series:
        return series.ewm(alpha=1.0 / period, adjust=False).mean()

    def _dirmov(self, df: pd.DataFrame) -> tuple:
        """Returns (+DI, -DI, TR_RMA)."""
        high  = df["high"]
        low   = df["low"]
        close = df["close"]

        up   = high.diff().clip(lower=0)
        down = (-low.diff()).clip(lower=0)

        tr1  = high - low
        tr2  = (high - close.shift(1)).abs()
        tr3  = (low  - close.shift(1)).abs()
        tr   = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr  = self._rma(tr, self.di_length)

        plus_dm  = up.where((up > down) & (up > 0), 0.0)
        minus_dm = down.where((down > up) & (down > 0), 0.0)

        plus_di  = 100 * self._rma(plus_dm,  self.di_length) / atr.replace(0, np.nan)
        minus_di = 100 * self._rma(minus_dm, self.di_length) / atr.replace(0, np.nan)
        return plus_di.fillna(0), minus_di.fillna(0)

    def _adx(self, plus_di: pd.Series, minus_di: pd.Series) -> pd.Series:
        total = (plus_di + minus_di).replace(0, 1)
        adx   = 100 * self._rma((plus_di - minus_di).abs() / total, self.adx_length)
        return adx

    def evaluate(self, df: pd.DataFrame) -> dict:
        """
        Returns:
          { "long": bool, "short": bool, "adx": float, "plus_di": float, "minus_di": float }
        """
        min_bars = self.di_length + self.adx_length + 10
        empty = {"long": False, "short": False, "adx": 0.0, "plus_di": 0.0, "minus_di": 0.0}
        if len(df) < min_bars:
            return empty

        plus_di, minus_di = self._dirmov(df)
        adx_series        = self._adx(plus_di, minus_di)

        adx_val   = float(adx_series.iloc[-1])
        plus_val  = float(plus_di.iloc[-1])
        minus_val = float(minus_di.iloc[-1])
        adx_prev  = float(adx_series.iloc[-2])

        above_threshold = adx_val >= self.key_level

        if self.mode == "adx_only":
            is_long  = above_threshold
            is_short = above_threshold
        elif self.mode == "adx_di":
            is_long  = above_threshold and plus_val > minus_val
            is_short = above_threshold and minus_val > plus_val
        else:  # advance
            trending = adx_val > adx_prev and above_threshold and adx_val < 55
            is_long  = trending and plus_val > minus_val  and (plus_val - minus_val) > 1
            is_short = trending and minus_val > plus_val  and (minus_val - plus_val) > 1

        logger.debug(
            f"[ADX] adx={adx_val:.1f} +DI={plus_val:.1f} -DI={minus_val:.1f} "
            f"long={is_long} short={is_short}"
        )
        return {"long": is_long, "short": is_short, "adx": adx_val,
                "plus_di": plus_val, "minus_di": minus_val}


# Singleton instances
adx_filter_scalp = ADXFilter(di_length=10, adx_length=5,  key_level=20, mode="adx_di")
adx_filter_swing = ADXFilter(di_length=14, adx_length=14, key_level=25, mode="advance")
