"""
gold_market_structure.py -- Full institutional SMC analysis for Gold (XAUUSD)
Detects: BOS, CHOCH, Order Blocks, FVGs, Liquidity Sweeps, Equal H/L,
         Asian Range, London Open breakout, Kill Zone timing.

v2.0 — Added:
  [NEW] detect_supply_demand_zones()  — Institutional S&D zones for 15m context
  [NEW] detect_key_levels()           — Recent swing high/low key price levels
  [NEW] detect_15m_bias()             — Overall 15m directional bias
"""

import pandas as pd # type: ignore
import numpy as np  # type: ignore
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("agniv.gold_smc")


# ── Swing Point Detection ─────────────────────────────────────────────────

def _find_swings(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """Basic N-bar fractal swing detection."""
    df["swing_high"] = False
    df["swing_low"]  = False

    for i in range(lookback, len(df) - lookback):
        window_h = df["high"].iloc[i - lookback: i + lookback + 1]
        window_l = df["low"].iloc[i - lookback: i + lookback + 1]
        if df["high"].iloc[i] == window_h.max():
            df.at[df.index[i], "swing_high"] = True
        if df["low"].iloc[i] == window_l.min():
            df.at[df.index[i], "swing_low"] = True
    return df


# ── Main Analysis ─────────────────────────────────────────────────────────

def detect_gold_smc(df: pd.DataFrame) -> dict:
    """
    Full SMC analysis on XAUUSD candle data.
    Returns structured context dict consumed by strategies.
    """
    if df.empty or len(df) < 60:
        return _empty_context()

    df = _find_swings(df, lookback=5)

    sh = df[df["swing_high"]]
    sl = df[df["swing_low"]]

    last_sh = float(sh["high"].iloc[-1])  if not sh.empty else 0.0
    last_sl = float(sl["low"].iloc[-1])   if not sl.empty else 0.0

    current_close = float(df["close"].iloc[-1])

    # 1. BOS / CHOCH
    bos, choch = _detect_bos_choch(df, last_sh, last_sl, current_close)

    # 2. Trend via HH/HL and LH/LL
    trend = _detect_trend(sh, sl)

    # 3. Order Blocks
    bull_obs = _find_order_blocks(df, direction="bull")
    bear_obs  = _find_order_blocks(df, direction="bear")

    # 4. Fair Value Gaps
    fvgs = _find_fvgs(df)

    # 5. Liquidity Sweeps
    sweeps = _find_liquidity_sweeps(df, last_sh, last_sl)

    # 6. Equal Highs / Equal Lows
    equal_highs = _find_equal_levels(sh["high"].values, tolerance=0.05)  # type: ignore
    equal_lows  = _find_equal_levels(sl["low"].values,  tolerance=0.05)  # type: ignore

    # 7. Asian Range
    asian_range = _get_asian_range(df)

    return {
        "trend":        trend,
        "bos":          bos,
        "choch":        choch,
        "bull_obs":     list(bull_obs)[:3],  # type: ignore
        "bear_obs":     list(bear_obs)[:3],  # type: ignore
        "fvgs":         list(fvgs)[:5],      # type: ignore
        "sweeps":       sweeps,
        "equal_highs":  equal_highs,
        "equal_lows":   equal_lows,
        "asian_range":  asian_range,
        "last_sh":      last_sh,
        "last_sl":      last_sl,
    }


def _empty_context() -> dict:
    return {
        "trend": "UNKNOWN", "bos": "NONE", "choch": "NONE",
        "bull_obs": [], "bear_obs": [], "fvgs": [],
        "sweeps": [], "equal_highs": [], "equal_lows": [],
        "asian_range": {"high": 0.0, "low": 0.0},
        "last_sh": 0.0, "last_sl": 0.0,
    }


# ── BOS and CHOCH ─────────────────────────────────────────────────────────

def _detect_bos_choch(df: pd.DataFrame, last_sh: float, last_sl: float, current_close: float):
    bos   = "NONE"
    choch = "NONE"

    if current_close > last_sh and last_sh > 0:
        bos = "BULLISH"
    elif current_close < last_sl and last_sl > 0:
        bos = "BEARISH"

    # CHOCH: price breaks opposite structure level suggesting reversal
    if bos == "BULLISH" and current_close < last_sl and last_sl > 0:
        choch = "BEARISH_REVERSAL"
    elif bos == "BEARISH" and current_close > last_sh and last_sh > 0:
        choch = "BULLISH_REVERSAL"

    return bos, choch


# ── Trend: HH/HL vs LH/LL ────────────────────────────────────────────────

def _detect_trend(sh: pd.DataFrame, sl: pd.DataFrame) -> str:
    if len(sh) < 2 or len(sl) < 2:
        return "UNKNOWN"

    sh_vals = np.array(sh["high"].values, dtype=np.float64)
    sl_vals = np.array(sl["low"].values,  dtype=np.float64)

    hh = sh_vals[-1] > sh_vals[-2]
    hl = sl_vals[-1] > sl_vals[-2]
    lh = sh_vals[-1] < sh_vals[-2]
    ll = sl_vals[-1] < sl_vals[-2]

    if hh and hl:
        return "UPTREND"
    elif lh and ll:
        return "DOWNTREND"
    else:
        return "RANGING"


# ── Order Blocks ──────────────────────────────────────────────────────────

def _find_order_blocks(df: pd.DataFrame, direction: str = "bull") -> list:
    obs = []
    for i in range(len(df) - 2, 5, -1):
        candle = df.iloc[i]
        is_bear_candle = candle["close"] < candle["open"]
        is_bull_candle = candle["close"] > candle["open"]

        if direction == "bull" and is_bear_candle:
            # OB valid if followed by significant upward move
            future_high = df["high"].iloc[i + 1:i + 10].max() if i + 10 <= len(df) else df["high"].iloc[i + 1:].max()
            if future_high > candle["high"] * 1.002:
                obs.append({
                    "type": "BULLISH_OB",
                    "top":    float(candle["high"]),
                    "bottom": float(candle["low"]),
                    "mid":    float((candle["high"] + candle["low"]) / 2),
                })
        elif direction == "bear" and is_bull_candle:
            future_low = df["low"].iloc[i + 1:i + 10].min() if i + 10 <= len(df) else df["low"].iloc[i + 1:].min()
            if future_low < candle["low"] * 0.998:
                obs.append({
                    "type": "BEARISH_OB",
                    "top":    float(candle["high"]),
                    "bottom": float(candle["low"]),
                    "mid":    float((candle["high"] + candle["low"]) / 2),
                })
    return obs


# ── Fair Value Gaps ───────────────────────────────────────────────────────

def _find_fvgs(df: pd.DataFrame) -> list:
    fvgs = []
    for i in range(2, len(df)):
        low_2  = float(df["low"].iloc[i - 2])
        high_2 = float(df["high"].iloc[i - 2])
        low_0  = float(df["low"].iloc[i])
        high_0 = float(df["high"].iloc[i])

        if low_2 > high_0:  # Bullish FVG
            fvgs.append({"type": "BULL_FVG", "top": low_2, "bottom": high_0})
        elif high_2 < low_0:  # Bearish FVG
            fvgs.append({"type": "BEAR_FVG", "top": low_0, "bottom": high_2})

    return list(fvgs)[-8:] if len(fvgs) > 8 else fvgs  # type: ignore


# ── Liquidity Sweeps ──────────────────────────────────────────────────────

def _find_liquidity_sweeps(df: pd.DataFrame, last_sh: float, last_sl: float) -> list:
    sweeps = []
    recent = df.tail(10)

    for _, row in recent.iterrows():
        if last_sh > 0 and float(row["high"]) > last_sh and float(row["close"]) < last_sh:
            sweeps.append({"type": "SELL_SIDE_SWEEP", "level": last_sh})
        if last_sl > 0 and float(row["low"]) < last_sl and float(row["close"]) > last_sl:
            sweeps.append({"type": "BUY_SIDE_SWEEP", "level": last_sl})

    return sweeps


# ── Equal Highs / Equal Lows ──────────────────────────────────────────────

def _find_equal_levels(values: np.ndarray, tolerance: float = 0.05) -> list:
    """Find price levels that repeat within tolerance (liquidity pools)."""
    clusters = []
    if len(values) < 2:
        return clusters
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            if abs(float(values[i]) - float(values[j])) <= tolerance: # type: ignore
                clusters.append(float(round(float((values[i] + values[j]) / 2), 3))) # type: ignore
    return list(set(clusters))[:5]  # type: ignore


# ── Asian Range ───────────────────────────────────────────────────────────

def _get_asian_range(df: pd.DataFrame) -> dict:
    """
    Approximate Asian session range from last 24 candles on M5,
    or last 8 candles on H1 (00:00-08:00 GMT).
    """
    tail = df.tail(24)
    return {
        "high": float(tail["high"].max()),
        "low":  float(tail["low"].min()),
        "mid":  float((tail["high"].max() + tail["low"].min()) / 2),
    }


# ── Supply & Demand Zones (for 15m Context Layer) ────────────────────────

def detect_supply_demand_zones(df: pd.DataFrame, lookback: int = 50,
                                impulse_mult: float = 1.6) -> dict:
    """
    Detect institutional Supply and Demand zones on the 15m chart.

    A Demand zone is the BASE before a strong bullish impulse move.
    A Supply zone is the BASE before a strong bearish impulse move.

    An "impulse" candle is one whose body is >= impulse_mult × avg body.

    Returns:
        {
            "demand": [{"top": float, "bottom": float, "strength": str}, ...],
            "supply": [{"top": float, "bottom": float, "strength": str}, ...],
        }
    """
    if df is None or len(df) < lookback + 5:
        return {"demand": [], "supply": []}

    body      = (df["close"] - df["open"]).abs()
    avg_body  = body.rolling(20).mean()
    direction = (df["close"] > df["open"]).astype(int) * 2 - 1  # +1 bull, -1 bear

    demand_zones: list = []
    supply_zones: list = []

    # Scan recent history (newest first)
    for i in range(len(df) - 1, lookback, -1):
        avg = float(avg_body.iloc[i]) if not pd.isna(avg_body.iloc[i]) else 1e-6
        if avg <= 0:
            continue

        impulse_ratio = float(body.iloc[i]) / avg

        # ── Demand Zone: strong bullish impulse ──
        if int(direction.iloc[i]) == 1 and impulse_ratio >= impulse_mult:
            # The base is the candle BEFORE the impulse (the ranging / down candle)
            base_idx = i - 1
            if base_idx >= 0:
                base = df.iloc[base_idx]
                zone_top    = float(max(base["open"], base["close"]))
                zone_bottom = float(base["low"])
                strength = "Strong" if impulse_ratio >= impulse_mult * 1.3 else "Normal"
                demand_zones.append({
                    "top":      zone_top,
                    "bottom":   zone_bottom,
                    "mid":      (zone_top + zone_bottom) / 2,
                    "strength": strength,
                    "index":    df.index[base_idx],
                })

        # ── Supply Zone: strong bearish impulse ──
        elif int(direction.iloc[i]) == -1 and impulse_ratio >= impulse_mult:
            base_idx = i - 1
            if base_idx >= 0:
                base = df.iloc[base_idx]
                zone_top    = float(base["high"])
                zone_bottom = float(min(base["open"], base["close"]))
                strength = "Strong" if impulse_ratio >= impulse_mult * 1.3 else "Normal"
                supply_zones.append({
                    "top":      zone_top,
                    "bottom":   zone_bottom,
                    "mid":      (zone_top + zone_bottom) / 2,
                    "strength": strength,
                    "index":    df.index[base_idx],
                })

        if len(demand_zones) + len(supply_zones) >= 10:
            break

    logger.debug(
        f"[S&D] Demand zones={len(demand_zones)} | Supply zones={len(supply_zones)}"
    )
    return {
        "demand": demand_zones[:3],
        "supply": supply_zones[:3],
    }


def detect_key_levels(df: pd.DataFrame, n_levels: int = 3,
                       swing_lookback: int = 5) -> dict:
    """
    Return the most recent N swing highs and N swing lows as key price levels.
    Used by the 15m tier to define where the bot should be watching for setups.

    Returns:
        {
            "highs": [float, ...],   # sorted descending (nearest first)
            "lows":  [float, ...],   # sorted ascending  (nearest first)
        }
    """
    if df is None or len(df) < swing_lookback * 3:
        return {"highs": [], "lows": []}

    df2 = _find_swings(df.copy(), lookback=swing_lookback)
    sh  = df2[df2["swing_high"]]["high"].values
    sl  = df2[df2["swing_low"]]["low"].values

    # Most recent N levels, nearest to current price first
    highs = sorted(sh[-n_levels:].tolist(), reverse=True) if len(sh) >= n_levels else sorted(sh.tolist(), reverse=True)
    lows  = sorted(sl[-n_levels:].tolist())               if len(sl) >= n_levels else sorted(sl.tolist())

    return {"highs": highs, "lows": lows}


def detect_15m_bias(df_15m: pd.DataFrame) -> dict:
    """
    Determine the overall directional bias from the 15-minute chart.

    Uses:
      1. HH/HL (Higher Highs / Higher Lows) vs LH/LL (Lower Highs / Lower Lows)
      2. Price vs EMA 50 and EMA 21
      3. Supply / Demand zone awareness

    Returns:
        {
            "bias":      "BULLISH" | "BEARISH" | "NEUTRAL",
            "trend":     "UPTREND" | "DOWNTREND" | "RANGING",
            "sd_zones":  dict,     # raw supply/demand zones
            "key_levels":dict,     # raw key levels
            "ema21":     float,
            "ema50":     float,
        }
    """
    empty = {
        "bias": "NEUTRAL", "trend": "RANGING",
        "sd_zones": {"demand": [], "supply": []},
        "key_levels": {"highs": [], "lows": []},
        "ema21": 0.0, "ema50": 0.0,
    }

    if df_15m is None or len(df_15m) < 60:
        return empty

    # EMA 21, EMA 50 and EMA 200 (macro trend baseline)
    ema21  = float(df_15m["close"].ewm(span=21,  adjust=False).mean().iloc[-1])
    ema50  = float(df_15m["close"].ewm(span=50,  adjust=False).mean().iloc[-1])
    ema200 = float(df_15m["close"].ewm(span=200, adjust=False).mean().iloc[-1])
    current_price = float(df_15m["close"].iloc[-1])

    # Trend via swing analysis
    df2 = _find_swings(df_15m.copy(), lookback=5)
    sh  = df2[df2["swing_high"]]
    sl  = df2[df2["swing_low"]]
    trend = _detect_trend(sh, sl)

    # Bias scoring
    bull_pts = 0
    bear_pts = 0

    if trend == "UPTREND":   bull_pts += 2
    elif trend == "DOWNTREND": bear_pts += 2

    if current_price > ema21: bull_pts += 1
    else:                     bear_pts += 1

    if current_price > ema50: bull_pts += 1
    else:                     bear_pts += 1

    if ema21 > ema50:  bull_pts += 1
    else:              bear_pts += 1

    # EMA200: macro trend direction — most weight
    if current_price > ema200: bull_pts += 2
    else:                      bear_pts += 2

    if ema50 > ema200:  bull_pts += 1
    else:               bear_pts += 1

    if bull_pts >= 4:
        bias = "BULLISH"
    elif bear_pts >= 4:
        bias = "BEARISH"
    else:
        bias = "NEUTRAL"

    sd  = detect_supply_demand_zones(df_15m)
    kl  = detect_key_levels(df_15m)

    logger.info(
        f"[15m Bias] Trend={trend} Bias={bias} "
        f"EMA21={ema21:.2f} EMA50={ema50:.2f} EMA200={ema200:.2f} Price={current_price:.2f} "
        f"(Bull={bull_pts} Bear={bear_pts})"
    )

    return {
        "bias":       bias,
        "trend":      trend,
        "sd_zones":   sd,
        "key_levels": kl,
        "ema21":      ema21,
        "ema50":      ema50,
        "ema200":     ema200,
        "bull_pts":   bull_pts,
        "bear_pts":   bear_pts,
    }


def price_near_zone(price: float, zones: list, buffer_pct: float = 0.002) -> bool:
    """
    Check if current price is inside or near (within buffer_pct %) any zone.
    Used to check if price has returned to a 15m S&D zone for a setup.
    """
    buffer = price * buffer_pct
    for zone in zones:
        if (zone["bottom"] - buffer) <= price <= (zone["top"] + buffer):
            return True
    return False


# ── Price nearness helper ─────────────────────────────────────────────────

def near_ob(price: float, obs: list, threshold_pips: float = 50.0) -> bool:
    """Check if current price is inside or near an Order Block."""
    for ob in obs:
        if ob["bottom"] - threshold_pips * 0.01 <= price <= ob["top"] + threshold_pips * 0.01:
            return True
    return False


def near_fvg(price: float, fvgs: list, threshold_pips: float = 30.0) -> bool:
    """Check if current price is inside or very near a Fair Value Gap."""
    for fvg in fvgs:
        bottom = fvg.get("bottom", 0) - threshold_pips * 0.01
        top    = fvg.get("top", 0) + threshold_pips * 0.01
        if bottom <= price <= top:
            return True
    return False
