"""
diy_custom_builder.py — DIY Custom Strategy Builder [ZP] - Python Port
=======================================================================
Ported from the TradingView Pine Script "DIY Custom Strategy Builder [ZP] - v1"
by ZPayab.

Design:
  1. A single "Leading Indicator" fires a PENDING signal (BUY or SELL).
  2. The bot then waits up to `signal_expiry` candles for ALL enabled
     "Confirmation Filters" to agree with the pending direction.
  3. If confirmation is achieved within the window → FINAL signal is emitted.
  4. If the window expires without confirmation → signal is cancelled.

This supports both SCALP and HOLD modes via swappable JSON config files:
  - diy_scalp_config.json   → tight expiry, fast indicators
  - diy_swing_config.json   → wider expiry, slower indicators

Usage:
    from strategies.diy_custom_builder import DIYCustomStrategy
    strategy = DIYCustomStrategy(config_path="diy_scalp_config.json")
    signal = strategy.generate_signal(df)   # returns "BUY", "SELL", or "HOLD"
"""

import json
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("agniv.strategies.diy_custom_builder")

# ---------------------------------------------------------------------------
# Helper: MA types (mirrors Pine Script ma() function)
# ---------------------------------------------------------------------------

def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def _rma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False).mean()

def _wma(s: pd.Series, n: int) -> pd.Series:
    weights = np.arange(1, n + 1, dtype=float)
    return s.rolling(n).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def _vwma(close: pd.Series, volume: pd.Series, n: int) -> pd.Series:
    return (close * volume).rolling(n).sum() / volume.rolling(n).sum()

def _ma(source: pd.Series, length: int, ma_type: str,
        volume: Optional[pd.Series] = None) -> pd.Series:
    t = ma_type.upper()
    if t == "SMA":  return _sma(source, length)
    if t == "EMA":  return _ema(source, length)
    if t == "RMA":  return _rma(source, length)
    if t == "WMA":  return _wma(source, length)
    if t == "VWMA" and volume is not None:
        return _vwma(source, volume, length)
    return _ema(source, length)   # default fallback


# ===========================================================================
#  LEADING INDICATOR ENGINES
# ===========================================================================

class _RangeFilterLeading:
    """Smooth range filter (Pine Script default version)."""

    def __init__(self, period: int = 100, multiplier: float = 3.0):
        self.period = period
        self.mult   = multiplier

    def _smooth_range(self, close: pd.Series) -> pd.Series:
        wper   = self.period * 2 - 1
        avrng  = _ema(close.diff().abs(), self.period)
        return _ema(avrng, wper) * self.mult

    def evaluate(self, df: pd.DataFrame) -> dict:
        min_bars = self.period * 3
        empty = {"long": False, "short": False}
        if len(df) < min_bars:
            return empty

        close  = df["close"]
        smrng  = self._smooth_range(close)

        # Iterative range filter (stateful)
        filt = close.copy()
        for i in range(1, len(close)):
            prev = filt.iloc[i - 1]
            c    = close.iloc[i]
            r    = smrng.iloc[i]
            if c > prev:
                filt.iloc[i] = max(c - r, prev)
            else:
                filt.iloc[i] = min(c + r, prev)

        upward   = (filt > filt.shift(1)).astype(int).cumsum()
        downward = (filt < filt.shift(1)).astype(int).cumsum()

        c  = close.iloc[-1]
        c1 = close.iloc[-2]
        f  = filt.iloc[-1]
        up = float(upward.iloc[-1]) > 0
        dn = float(downward.iloc[-1]) > 0

        rf_long  = (c > f and c > c1 and up) or (c > f and c < c1 and up)
        rf_short = (c < f and c < c1 and dn) or (c < f and c > c1 and dn)

        logger.debug(f"[RangeFilter] long={rf_long} short={rf_short}")
        return {"long": rf_long, "short": rf_short}


class _RQKLeading:
    """Rational Quadratic Kernel (Nadaraya-Watson estimator)."""

    def __init__(self, lookback: float = 8.0, relative_weight: float = 8.0,
                 start_bar: int = 25, lag: int = 2):
        self.h   = lookback
        self.r   = relative_weight
        self.x0  = start_bar
        self.lag = lag

    def _kernel(self, src: pd.Series, h: float) -> float:
        n      = len(src)
        limit  = min(n - 1, n + self.x0)
        cw, tw = 0.0, 0.0
        for i in range(limit):
            w = (1 + (i * i) / (2 * self.r * h * h)) ** (-self.r)
            cw += src.iloc[-(i + 1)] * w
            tw += w
        return cw / tw if tw else float("nan")

    def evaluate(self, df: pd.DataFrame) -> dict:
        empty = {"long": False, "short": False}
        min_bars = max(50, self.x0 + 10)
        if len(df) < min_bars:
            return empty

        src = df["close"]
        y1_0 = self._kernel(src, self.h)
        y1_1 = self._kernel(src.iloc[:-1], self.h)
        y1_2 = self._kernel(src.iloc[:-2], self.h)
        y2_0 = self._kernel(src, self.h - self.lag)

        rqk_long  = y1_1 < y1_0   # currently going up
        rqk_short = y1_1 > y1_0   # currently going down

        logger.debug(f"[RQK] y1_0={y1_0:.2f} y1_1={y1_1:.2f} long={rqk_long} short={rqk_short}")
        return {"long": rqk_long, "short": rqk_short}


class _SupertrendLeading:
    """Supertrend leading indicator."""

    def __init__(self, period: int = 10, multiplier: float = 3.0):
        self.period = period
        self.mult   = multiplier

    def evaluate(self, df: pd.DataFrame) -> dict:
        empty = {"long": False, "short": False}
        if len(df) < self.period + 5:
            return empty

        hl2   = (df["high"] + df["low"]) / 2
        atr   = _rma(pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"]  - df["close"].shift(1)).abs()
        ], axis=1).max(axis=1), self.period)

        upper = hl2 + self.mult * atr
        lower = hl2 - self.mult * atr

        # Track trend state
        trend = pd.Series(1, index=df.index)
        up    = lower.copy()
        dn    = upper.copy()

        for i in range(1, len(df)):
            prev_close = df["close"].iloc[i - 1]
            up.iloc[i]    = max(lower.iloc[i], up.iloc[i-1])   if prev_close > up.iloc[i-1]   else lower.iloc[i]
            dn.iloc[i]    = min(upper.iloc[i], dn.iloc[i-1])   if prev_close < dn.iloc[i-1]   else upper.iloc[i]
            if trend.iloc[i-1] == -1 and df["close"].iloc[i] > dn.iloc[i-1]:
                trend.iloc[i] = 1
            elif trend.iloc[i-1] == 1 and df["close"].iloc[i] < up.iloc[i-1]:
                trend.iloc[i] = -1
            else:
                trend.iloc[i] = trend.iloc[i-1]

        t = int(trend.iloc[-1])
        return {"long": t == 1, "short": t == -1}


class _EMALeading:
    """2-EMA cross leading indicator."""

    def __init__(self, fast: int = 50, slow: int = 200):
        self.fast = fast
        self.slow = slow

    def evaluate(self, df: pd.DataFrame) -> dict:
        empty = {"long": False, "short": False}
        if len(df) < self.slow + 5:
            return empty
        fast_ema = _ema(df["close"], self.fast)
        slow_ema = _ema(df["close"], self.slow)
        return {
            "long":  float(fast_ema.iloc[-1]) > float(slow_ema.iloc[-1]),
            "short": float(fast_ema.iloc[-1]) < float(slow_ema.iloc[-1]),
        }


class _MACDLeading:
    """MACD leading indicator (zero line crossover mode)."""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast   = fast
        self.slow   = slow
        self.signal = signal

    def evaluate(self, df: pd.DataFrame) -> dict:
        empty = {"long": False, "short": False}
        if len(df) < self.slow + self.signal + 5:
            return empty
        macd   = _ema(df["close"], self.fast) - _ema(df["close"], self.slow)
        sig    = _ema(macd, self.signal)
        m, s   = float(macd.iloc[-1]), float(sig.iloc[-1])
        return {
            "long":  m > s and m > 0,
            "short": m < s and m < 0,
        }


class _RSILeading:
    """RSI vs midline leading indicator."""

    def __init__(self, length: int = 14, midline: int = 50):
        self.length  = length
        self.midline = midline

    def evaluate(self, df: pd.DataFrame) -> dict:
        empty = {"long": False, "short": False}
        if len(df) < self.length + 5:
            return empty
        delta = df["close"].diff()
        gain  = delta.clip(lower=0).ewm(alpha=1/self.length, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(alpha=1/self.length, adjust=False).mean()
        rsi   = 100 - (100 / (1 + gain / loss.replace(0, float("nan"))))
        val   = float(rsi.iloc[-1])
        return {"long": val > self.midline, "short": val < self.midline}


# ===========================================================================
#  CONFIRMATION FILTER ENGINES  (light wrappers around our filter modules)
# ===========================================================================

def _load_filter(name: str, params: dict):
    """Dynamically load and return a filter instance by name."""
    name = name.lower()

    if name == "range_filter":
        from filters.range_filter import RangeFilter
        return RangeFilter(**{k: v for k, v in params.items()
                              if k in ("period", "multiplier", "filter_type", "use_ema_average")})

    if name == "rqk":
        from filters.rqk_filter import RQKFilter
        # Map config key 'relative_weight' to RQKFilter's 'rel_weight'
        mapped = {}
        for k, v in params.items():
            if k == "relative_weight":
                mapped["rel_weight"] = v
            elif k in ("lookback", "rel_weight", "start_bar", "lag"):
                mapped[k] = v
        return RQKFilter(**mapped)

    if name == "wae":
        from filters.wae_filter import WAEFilter
        return WAEFilter(**{k: v for k, v in params.items()
                            if k in ("sensitivity", "fast_ema", "slow_ema", "bb_length", "bb_mult",
                                     "deadzone_rma", "deadzone_mult")})

    if name == "supertrend":
        from filters.supertrend_filter import SupertrendFilter
        return SupertrendFilter(**{k: v for k, v in params.items()
                                   if k in ("period", "multiplier")})

    if name == "rsi":
        from filters.rsi_filter import RSIFilter
        return RSIFilter(**{k: v for k, v in params.items()
                            if k in ("rsi_length", "ma_length", "overbought", "oversold",
                                     "midline", "long_limit", "short_limit", "use_limits", "mode")})

    if name == "macd":
        from filters.macd_filter import MACDFilter
        return MACDFilter(**{k: v for k, v in params.items()
                             if k in ("fast_length", "slow_length", "signal_length", "mode")})

    if name == "stochastic":
        from filters.stochastic_filter import StochasticFilter
        return StochasticFilter(**{k: v for k, v in params.items()
                                   if k in ("length", "smooth_k", "smooth_d", "overbought", "oversold", "mode")})

    if name == "adx":
        from filters.adx_filter import ADXFilter
        return ADXFilter(**{k: v for k, v in params.items()
                            if k in ("di_length", "adx_length", "key_level", "mode")})

    if name == "cci":
        from filters.cci_filter import CCIFilter
        return CCIFilter(**{k: v for k, v in params.items()
                            if k in ("length", "upper_band", "lower_band")})

    if name == "ssl_channel":
        from filters.ssl_channel_filter import SSLChannelFilter
        return SSLChannelFilter(**{k: v for k, v in params.items()
                                   if k in ("period",)})

    if name == "ema_filter":
        # Simple price vs EMA filter
        class _EMAFilter:
            def __init__(self, period=200):
                self.period = period
            def evaluate(self, df):
                ema = _ema(df["close"], self.period)
                c   = float(df["close"].iloc[-1])
                e   = float(ema.iloc[-1])
                return {"long": c > e, "short": c < e}
        return _EMAFilter(**{k: v for k, v in params.items() if k == "period"})

    logger.warning(f"[DIY] Unknown filter '{name}' — skipping")
    return None


def _evaluate_filter(f, df: pd.DataFrame, direction: str) -> bool:
    """Run a confirmation filter and return True if it agrees with direction."""
    try:
        result = f.evaluate(df)
        if direction == "BUY":
            return (
                result.get("long",     False) or
                result.get("safe_buy", False) or
                result.get("up",       False) or
                result.get("bullish",  False) or
                result.get("trend", "") == "UP"
            )
        else:
            return (
                result.get("short",     False) or
                result.get("safe_sell", False) or
                result.get("down",      False) or
                result.get("bearish",   False) or
                result.get("trend", "") == "DOWN"
            )
    except Exception as e:
        logger.error(f"[DIY] Filter evaluate error: {e}")
        return False


# ===========================================================================
#  MAIN STRATEGY CLASS
# ===========================================================================

class DIYCustomStrategy:
    """
    DIY Custom Strategy Builder — Python port of ZPayab Pine Script v1.

    Supports two modes via config file:
      - Scalp profile  (diy_scalp_config.json)  → 1m/5m/15m timeframes
      - Swing/Hold profile (diy_swing_config.json) → 1h/4h/1d timeframes

    Signal flow:
      bar N:   Leading Indicator fires → pending_signal = "BUY" | "SELL"
      bar N+k: All confirmation filters agree → emit FINAL signal
      bar N+expiry: Expiry reached without confirmation → reset to "HOLD"

    alternate_signal: If True, a BUY can flip to SELL immediately (Pine default)
    """

    # Map of supported leading indicator names to their engine classes
    LEADING_INDICATORS = {
        "Range Filter":       _RangeFilterLeading,
        "RQK":                _RQKLeading,
        "Supertrend":         _SupertrendLeading,
        "2 EMA Cross":        _EMALeading,
        "MACD":               _MACDLeading,
        "RSI":                _RSILeading,
    }

    def __init__(self, config_path: str = "diy_scalp_config.json"):
        self.config       = self._load_config(config_path)
        self.symbol       = self.config.get("symbol", "XAUUSD")
        self.timeframe    = self.config.get("timeframe", "M5")
        self.signal_expiry    = self.config.get("signal_expiry", 3)
        self.alternate_signal = self.config.get("alternate_signal", True)

        # State for the expiry mechanism
        self._pending_direction: Optional[str] = None   # "BUY" | "SELL"
        self._pending_bars: int = 0                      # how many bars elapsed

        # Last calculated indicator values for the dashboard
        self._last_metrics = {
            "trend": "Sideways",
            "momentum": "Neutral",
            "volume": "Neutral",
            "rsi": 50.0,
            "adx": 20.0,
            "vwap": 0.0,
            "regime": "Low Volatility",
        }

        # Build leading indicator
        li_cfg  = self.config.get("leading_indicator", {})
        li_name = li_cfg.get("name", "Range Filter")
        li_params = li_cfg.get("params", {})
        cls = self.LEADING_INDICATORS.get(li_name)
        if cls is None:
            logger.warning(f"[DIY] Unknown leading indicator '{li_name}', defaulting to Range Filter")
            cls = _RangeFilterLeading
        self._leading = cls(**li_params) if li_params else cls()

        # Build confirmation filters
        self._filters = []
        for f_cfg in self.config.get("confirmation_filters", []):
            if not f_cfg.get("enabled", True):
                continue
            f_obj = _load_filter(f_cfg["name"], f_cfg.get("params", {}))
            if f_obj is not None:
                self._filters.append((f_cfg["name"], f_obj))

        logger.info(
            f"[DIY] Loaded: leading='{li_name}' | "
            f"filters={[n for n, _ in self._filters]} | "
            f"expiry={self.signal_expiry} | mode={'SCALP' if 'scalp' in config_path else 'SWING'}"
        )

    # ------------------------------------------------------------------

    def _load_config(self, config_path: str) -> dict:
        """Load JSON config from file or absolute path."""
        # Try relative to bot root first
        bot_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates = [
            config_path,
            os.path.join(bot_root, config_path),
            os.path.join(bot_root, "configs", config_path),
        ]
        for path in candidates:
            if os.path.exists(path):
                with open(path, "r") as f:
                    logger.info(f"[DIY] Config loaded from: {path}")
                    return json.load(f)
        logger.warning(f"[DIY] Config '{config_path}' not found — using defaults")
        return self._default_scalp_config()

    def _default_scalp_config(self) -> dict:
        return {
            "symbol":           "XAUUSD",
            "timeframe":        "M5",
            "signal_expiry":    3,
            "alternate_signal": True,
            "leading_indicator": {
                "name":   "Range Filter",
                "params": {"period": 100, "multiplier": 3.0}
            },
            "confirmation_filters": [
                {"name": "rqk",  "enabled": True, "params": {}},
                {"name": "wae",  "enabled": True, "params": {}},
            ]
        }

    # ------------------------------------------------------------------

    def generate_signal(self, df: pd.DataFrame) -> str:
        """
        Evaluate the leading indicator and all confirmation filters.

        Args:
            df: OHLCV DataFrame with columns [open, high, low, close, volume].
                Must have at least 200 rows for reliable signals.

        Returns:
            "BUY" | "SELL" | "HOLD"
        """
        if df is None or len(df) < 50:
            return "HOLD"

        try:
            return self._evaluate(df)
        except Exception as e:
            logger.exception(f"[DIY] generate_signal error: {e}")
            return "HOLD"

    def _evaluate(self, df: pd.DataFrame) -> str:
        # --- Step 1: Run the leading indicator ---
        li_result = self._leading.evaluate(df)
        li_long   = li_result.get("long", False)
        li_short  = li_result.get("short", False)

        new_direction: Optional[str] = None
        if li_long:
            new_direction = "BUY"
        elif li_short:
            new_direction = "SELL"

        # --- Step 2: Manage pending state ---
        if new_direction is not None:
            if self._pending_direction is None:
                # Fresh signal from leading indicator
                self._pending_direction = new_direction
                self._pending_bars      = 0
                logger.debug(f"[DIY] Pending signal set: {new_direction}")
            elif self.alternate_signal and new_direction != self._pending_direction:
                # Flip to opposite direction immediately
                self._pending_direction = new_direction
                self._pending_bars      = 0
                logger.debug(f"[DIY] Signal flipped to: {new_direction}")

        # Increment bar counter if we have a pending signal
        if self._pending_direction is not None:
            self._pending_bars += 1

        # Check expiry
        if self._pending_bars > self.signal_expiry:
            logger.debug(
                f"[DIY] Signal expired after {self._pending_bars} bars — resetting"
            )
            self._pending_direction = None
            self._pending_bars      = 0
            return "HOLD"

        # --- Step 3: Compute Dashboard metrics ---
        try:
            self._update_dashboard_metrics(df)
        except Exception as e:
            logger.debug(f"[DIY] update metrics error: {e}")

        if self._pending_direction is None:
            return "HOLD"

        # --- Step 4: Check all confirmation filters ---
        direction = self._pending_direction
        confirmations = []
        for fname, fobj in self._filters:
            ok = _evaluate_filter(fobj, df, direction)
            confirmations.append((fname, ok))
            logger.debug(f"[DIY] Filter '{fname}' → {ok} for {direction}")

        all_confirmed = all(ok for _, ok in confirmations)

        if all_confirmed:
            logger.info(
                f"[DIY] ✅ {direction} confirmed in {self._pending_bars} bars "
                f"| Filters: {confirmations}"
            )
            # Reset state after confirmed signal
            self._pending_direction = None
            self._pending_bars      = 0
            return direction

        # Still waiting for confirmation
        logger.debug(
            f"[DIY] Waiting: bar {self._pending_bars}/{self.signal_expiry} "
            f"| {direction} confirmations: {confirmations}"
        )
        return "HOLD"

    # ------------------------------------------------------------------

    def reset(self):
        """Reset pending signal state (call when switching symbols or timeframes)."""
        self._pending_direction = None
        self._pending_bars      = 0

    def _update_dashboard_metrics(self, df: pd.DataFrame):
        """Internal helper to calculate common metrics for the Rich dashboard."""
        if len(df) < 20: return

        close = df["close"]
        high  = df["high"]
        low   = df["low"]
        vol   = df["volume"]

        # 1. RSI (Standard 14)
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi   = 100 - (100 / (1 + rs))
        self._last_metrics["rsi"] = round(float(rsi.iloc[-1]), 2) if not np.isnan(rsi.iloc[-1]) else 50.0

        # 2. VWAP (Simple session-based)
        vwap = (close * vol).cumsum() / vol.cumsum()
        self._last_metrics["vwap"] = round(float(vwap.iloc[-1]), 2)

        # 3. ADX (Standard 14)
        up   = high.diff()
        down = -low.diff()
        tr   = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
        tr14 = tr.rolling(14).sum()
        plus_di  = 100 * (up.where((up > down) & (up > 0), 0)).rolling(14).sum() / tr14
        minus_di = 100 * (down.where((down > up) & (down > 0), 0)).rolling(14).sum() / tr14
        dx  = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(14).mean()
        self._last_metrics["adx"] = round(float(adx.iloc[-1]), 2) if not np.isnan(adx.iloc[-1]) else 20.0

        # 4. Trend / Momentum / Volume / Regime logic
        ema20 = close.ewm(span=20).mean()
        ema50 = close.ewm(span=50).mean()
        
        if ema20.iloc[-1] > ema50.iloc[-1] + 1.0: self._last_metrics["trend"] = "UP"
        elif ema20.iloc[-1] < ema50.iloc[-1] - 1.0: self._last_metrics["trend"] = "DOWN"
        else: self._last_metrics["trend"] = "Sideways"

        self._last_metrics["momentum"] = "Bullish" if close.iloc[-1] > ema20.iloc[-1] else "Bearish"
        
        vol_avg = vol.rolling(20).mean().iloc[-1]
        self._last_metrics["volume"] = "High" if vol.iloc[-1] > vol_avg * 1.5 else "Neutral"
        
        atr = tr.rolling(14).mean().iloc[-1]
        self._last_metrics["regime"] = "High Volatility" if atr > tr.rolling(100).mean().iloc[-1] * 1.2 else "Low Volatility"

    @property
    def name(self) -> str:
        return "DIY Custom Strategy Builder"

    def get_status(self) -> dict:
        """Return current pending state for dashboard/logging."""
        return {
            "pending_direction": self._pending_direction,
            "pending_bars":      self._pending_bars,
            "signal_expiry":     self.signal_expiry,
            "leading_indicator": type(self._leading).__name__,
            "active_filters":   [n for n, _ in self._filters],
            "metrics":           self._last_metrics,
        }
