"""
ssl_channel_filter.py — SSL Channel Filter (Ported from ZPayab Pine Script)
===========================================================================
SSL Channel identifies trend direction by comparing SMA of Highs vs SMA of Lows
and whether price is above or below those bands.

Original Pine Script logic by ZPayab — translated to Python/Pandas.
"""

import logging
import pandas as pd

logger = logging.getLogger("agniv.filters.ssl_channel")


class SSLChannelFilter:
    """
    SSL Channel — Trend Direction Filter.

    Logic:
      - SMA(High, period) = upper band
      - SMA(Low,  period) = lower band
      - If close > upper band → Hlv = +1 (bullish)
      - If close < lower band → Hlv = -1 (bearish)
      - SSL Up   = Hlv == +1 ? lower : upper
      - SSL Down = Hlv == +1 ? upper : lower
      - Long  when SSL_Up > SSL_Down
      - Short when SSL_Up < SSL_Down
    """

    def __init__(self, period: int = 10):
        self.period = period

    def evaluate(self, df: pd.DataFrame) -> dict:
        """
        Returns:
          { "long": bool, "short": bool, "ssl_up": float, "ssl_down": float }
        """
        empty = {"long": False, "short": False, "ssl_up": 0.0, "ssl_down": 0.0}
        if len(df) < self.period + 5:
            return empty

        high  = df["high"]
        low   = df["low"]
        close = df["close"]

        sma_high = high.rolling(self.period).mean()
        sma_low  = low.rolling(self.period).mean()

        # Build Hlv series iteratively (stateful)
        hlv = pd.Series(0, index=close.index)
        for i in range(1, len(close)):
            if close.iloc[i] > sma_high.iloc[i]:
                hlv.iloc[i] = 1
            elif close.iloc[i] < sma_low.iloc[i]:
                hlv.iloc[i] = -1
            else:
                hlv.iloc[i] = hlv.iloc[i - 1]

        ssl_down = hlv.apply(lambda h: sma_high.iloc[hlv.index.get_loc(h)] if False else 0)  # placeholder
        # Vectorised approach
        ssl_up_series   = sma_low.where(hlv < 0, sma_high)
        ssl_down_series = sma_high.where(hlv < 0, sma_low)

        ssl_up_val   = float(ssl_up_series.iloc[-1])
        ssl_down_val = float(ssl_down_series.iloc[-1])

        is_long  = ssl_up_val > ssl_down_val
        is_short = ssl_up_val < ssl_down_val

        logger.debug(f"[SSL] ssl_up={ssl_up_val:.2f} ssl_down={ssl_down_val:.2f} long={is_long}")
        return {"long": is_long, "short": is_short, "ssl_up": ssl_up_val, "ssl_down": ssl_down_val}


# Singleton instances
ssl_filter_scalp = SSLChannelFilter(period=10)
ssl_filter_swing = SSLChannelFilter(period=20)
