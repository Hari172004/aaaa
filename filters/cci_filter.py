"""
cci_filter.py — CCI Filter (Ported from ZPayab Pine Script)
============================================================
Commodity Channel Index — generates long/short when CCI
breaks above the upper band or below the lower band.

Original Pine Script logic by ZPayab — translated to Python/Pandas.
"""

import logging
import pandas as pd

logger = logging.getLogger("agniv.filters.cci")


class CCIFilter:
    """
    CCI — Commodity Channel Index signal filter.

    Long  signal when CCI > upper_band (bullish momentum)
    Short signal when CCI < lower_band (bearish momentum)
    """

    def __init__(
        self,
        length:      int = 20,
        upper_band:  int = 100,
        lower_band:  int = -100,
    ):
        self.length     = length
        self.upper_band = upper_band
        self.lower_band = lower_band

    def evaluate(self, df: pd.DataFrame) -> dict:
        """
        Returns:
          { "long": bool, "short": bool, "cci": float }
        """
        empty = {"long": False, "short": False, "cci": 0.0}
        if len(df) < self.length + 5:
            return empty

        src = (df["high"] + df["low"] + df["close"]) / 3
        ma  = src.rolling(self.length).mean()
        mad = src.rolling(self.length).apply(lambda x: (x - x.mean()).abs().mean(), raw=True)
        cci = (src - ma) / (0.015 * mad.replace(0, float("nan")))

        cci_val  = float(cci.iloc[-1])
        is_long  = cci_val > self.upper_band
        is_short = cci_val < self.lower_band

        logger.debug(f"[CCI] cci={cci_val:.1f} long={is_long} short={is_short}")
        return {"long": is_long, "short": is_short, "cci": cci_val}


# Singleton instances
cci_filter_scalp = CCIFilter(length=20, upper_band=100, lower_band=-100)
cci_filter_swing = CCIFilter(length=20, upper_band=100, lower_band=-100)
