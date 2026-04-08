"""
gold_risk_manager.py — Professional Gold Risk Manager for XAUUSD
================================================================
Built for client delivery: supports $10 micro-accounts with broker leverage,
anti-martingale compounding, and pyramid order sizing (up to 5 orders on strong signals).

Lot size formula for standard accounts with leverage:
    Margin needed = (lot_size * 100oz * price) / leverage
    Risk per pip  = lot * 10  (for XAUUSD, 1 pip = $1/0.01 lot)

Signal strength pyramid:
    ≥ 0.95 → up to 5 orders
    ≥ 0.85 → up to 3 orders
    ≥ 0.75 → up to 2 orders
    ≥ 0.50 → 1 order
"""

import json
import logging
import pathlib
from datetime import datetime, date, timezone
from typing import List

from analysis.gold_sessions import is_lbma_fix_time  # type: ignore

logger = logging.getLogger("agniv.gold_risk")

# ── Persistent daily stats file ────────────────────────────────────────────
_DAILY_FILE = pathlib.Path("data/daily_gold_stats.json")

# ── Risk constants ─────────────────────────────────────────────────────────
MIN_LOT               = 0.01    # absolute minimum lot
MAX_LOT               = 10.0    # absolute maximum lot per single order
MAX_OPEN_GOLD_TRADES  = 10      # up to 5-order pyramids so allow 10 max
DAILY_LOSS_LIMIT_PCT  = 5.0     # stop trading after 5% daily drawdown
MAX_SPREAD_POINTS     = 3.0     # 30 pips = 3.0 XAU points — pause if wider
ATR_SPIKE_MULTIPLIER  = 3.5     # reduce size 70% if ATR > 3.5x rolling avg
MIN_SL_POINTS         = 1.0     # minimum SL distance (10 pips)

# News blackout window around high-impact events
NEWS_BLACKOUT_MINS = 30

# Pyramid order lot weighting (% of base_lot per order number 1–5)
PYRAMID_WEIGHTS = [1.0, 0.75, 0.50, 0.35, 0.25]


# ── Persistent helpers ────────────────────────────────────────────────────

def _today() -> str:
    return date.today().isoformat()


def _load_daily() -> dict:
    try:
        _DAILY_FILE.parent.mkdir(parents=True, exist_ok=True)
        if _DAILY_FILE.exists():
            return json.loads(_DAILY_FILE.read_text())
    except Exception as e:
        logger.warning(f"[GoldRisk] Could not load daily stats: {e}")
    return {}


def _save_daily(data: dict):
    try:
        _DAILY_FILE.parent.mkdir(parents=True, exist_ok=True)
        _DAILY_FILE.write_text(json.dumps(data, indent=2))
    except Exception as e:
        logger.warning(f"[GoldRisk] Could not save daily stats: {e}")


def record_daily_loss(loss_usd: float):
    key   = _today()
    daily = _load_daily()
    daily.setdefault(key, {"loss": 0.0, "profit": 0.0, "trades": 0})
    daily[key]["loss"]   += abs(loss_usd)
    daily[key]["trades"] += 1
    _save_daily(daily)
    logger.info(f"[GoldRisk] Daily loss → ${daily[key]['loss']:.2f}")


def record_daily_profit(profit_usd: float):
    key   = _today()
    daily = _load_daily()
    daily.setdefault(key, {"loss": 0.0, "profit": 0.0, "trades": 0})
    daily[key]["profit"] += abs(profit_usd)
    daily[key]["trades"] += 1
    _save_daily(daily)
    logger.info(f"[GoldRisk] Daily profit → ${daily[key]['profit']:.2f}")


def _daily_loss_exceeded(balance: float) -> bool:
    key   = _today()
    daily = _load_daily()
    loss  = daily.get(key, {}).get("loss", 0.0)
    limit = balance * (DAILY_LOSS_LIMIT_PCT / 100)
    exceeded = loss >= limit
    if exceeded:
        logger.warning(f"[GoldRisk] Daily loss limit hit: ${loss:.2f} >= ${limit:.2f}")
    return exceeded


# ── Main Risk Manager ─────────────────────────────────────────────────────

class GoldRiskManager:
    """
    Gold-only professional risk manager.

    Supports:
      - $10 micro accounts via leverage
      - Anti-martingale compounding (lot grows with balance)
      - Pyramid orders (1–5 orders on strong signals)
      - Session-aware risk (London/NY boost, Asian reduction)
      - ATR spike detection and size reduction
    """

    def __init__(self, config):
        self.config   = config
        self.funded   = getattr(config, "mode", "DEMO") == "FUNDED"
        self.leverage = getattr(config, "leverage", 500)
        self.low_balance_mode = False
        self.current_balance  = 0.0

    def set_dynamic_safety(self, balance: float):
        """Adjust risk parameters based on current account size."""
        self.low_balance_mode = (balance < 200)
        self.current_balance  = balance
        tier = self._get_tier(balance)
        logger.info(f"[GoldRisk] Balance=${balance:.2f} | Tier={tier['name']} | BaseRisk={tier['risk_pct']:.1f}%")

    # ── Tier System ────────────────────────────────────────────────────

    def _get_tier(self, balance: float) -> dict:
        """
        Anti-martingale compounding tiers.
        As the account grows, the risk % and lot size grow proportionally.
        """
        if self.funded:
            return {"name": "FUNDED", "risk_pct": 1.0, "max_orders": 2}

        if balance < 20:
            return {"name": "NANO",    "risk_pct": 3.0, "max_orders": 5}
        elif balance < 50:
            return {"name": "MICRO",   "risk_pct": 4.0, "max_orders": 5}
        elif balance < 200:
            return {"name": "STARTER", "risk_pct": 3.5, "max_orders": 5}
        elif balance < 500:
            return {"name": "GROWTH",  "risk_pct": 2.5, "max_orders": 5}
        elif balance < 2000:
            return {"name": "PRO",     "risk_pct": 2.0, "max_orders": 5}
        else:
            return {"name": "ELITE",   "risk_pct": getattr(self.config, "risk_pct", 1.5), "max_orders": 5}

    # ── Lot Size Calculator ────────────────────────────────────────────

    def calculate_base_lot(
        self,
        balance:      float,
        sl_points:    float,
        atr_spike:    bool = False,
        session_mult: float = 1.0,
    ) -> float:
        """
        Calculate the base anchor lot size for one trade.

        XAUUSD standard account (with leverage):
          - 1 lot = 100 oz of gold
          - 1 point move = $100 per lot
          - Risk per trade = balance * risk_pct / 100
          - Volume = risk_usd / (sl_points * 100)

        For micro accounts with $10:
          - 2% risk = $0.20 → 0.01 lot (minimum)
          - With 500:1 leverage, margin needed = 0.01 * 100 * 2000 / 500 = $4 ✓
        """
        tier      = self._get_tier(balance)
        risk_pct  = tier["risk_pct"] * session_mult
        risk_usd  = balance * (risk_pct / 100)

        # ATR spike guard — reduce size 70%
        if atr_spike:
            risk_usd *= 0.30
            logger.warning(f"[GoldRisk] ATR spike active — lot reduced 70%")

        sl_pts  = max(sl_points, MIN_SL_POINTS)
        raw_lot = risk_usd / (sl_pts * 100)

        # Funded account cap
        if self.funded:
            max_funded = balance * 0.005 / (sl_pts * 100)
            raw_lot = min(raw_lot, max_funded)

        lot = round(max(MIN_LOT, min(raw_lot, MAX_LOT)), 2)
        logger.debug(f"[GoldRisk] base_lot={lot} | risk_usd=${risk_usd:.2f} | sl={sl_pts:.2f}pts")
        return lot

    # ── Pyramid Order Calculator ───────────────────────────────────────

    def calculate_pyramid_lots(
        self,
        balance:          float,
        signal_strength:  float,
        sl_points:        float,
        atr_spike:        bool  = False,
        session_mult:     float = 1.0,
        open_gold_trades: int   = 0,
    ) -> List[float]:
        """
        Return a list of lot sizes for pyramid orders.

        signal_strength controls how many orders to place:
          ≥ 0.95 → 5 orders (perfect — all filters confirmed)
          ≥ 0.85 → 3 orders
          ≥ 0.75 → 2 orders
          ≥ 0.50 → 1 order

        Returns [] if risk checks fail.
        """
        tier       = self._get_tier(balance)
        max_orders = tier["max_orders"]

        # Determine number of orders from signal strength
        if signal_strength >= 0.95:
            n_orders = min(5, max_orders)
        elif signal_strength >= 0.85:
            n_orders = min(3, max_orders)
        elif signal_strength >= 0.75:
            n_orders = min(2, max_orders)
        else:
            n_orders = 1

        # Clamp to remaining slot capacity
        remaining_slots = MAX_OPEN_GOLD_TRADES - open_gold_trades
        n_orders = max(0, min(n_orders, remaining_slots))

        if n_orders == 0:
            return []

        base_lot = self.calculate_base_lot(balance, sl_points, atr_spike, session_mult)
        lots     = []
        for i in range(n_orders):
            w   = PYRAMID_WEIGHTS[i]
            lot = round(max(MIN_LOT, base_lot * w), 2)
            lots.append(lot)

        logger.info(
            f"[GoldRisk] Pyramid plan: {n_orders} orders | strength={signal_strength:.0%} "
            f"| lots={lots} | tier={tier['name']}"
        )
        return lots

    # ── Full Rule Check ────────────────────────────────────────────────

    def check_all_rules(
        self,
        balance:         float,
        signal:          str,
        atr:             float,
        open_gold_pos:   int,
        spread_points:   float = 0.0,
        news_pause:      bool  = False,
        avg_atr:         float = 0.0,
        signal_strength: float = 0.7,
        strategy:        str   = "SCALP",
    ) -> dict:
        """
        Run all gold risk checks and return pyramid lot plan.

        Returns:
          {
            can_trade:        bool,
            reason:           str,
            lots:             List[float],   ← pyramid lot sizes
            volume:           float,          ← first/anchor lot
            sl_value:         float,
            risk_usd:         float,
            n_orders:         int,
          }
        """
        if signal == "HOLD":
            return {"can_trade": False, "reason": "No signal", "lots": [], "volume": 0.0}

        # 1. Daily loss limit
        if _daily_loss_exceeded(balance):
            return {"can_trade": False, "reason": f"Daily {DAILY_LOSS_LIMIT_PCT}% loss limit reached",
                    "lots": [], "volume": 0.0}

        # 2. Max concurrent trades
        if open_gold_pos >= MAX_OPEN_GOLD_TRADES:
            return {"can_trade": False, "reason": f"Max {MAX_OPEN_GOLD_TRADES} gold trades open",
                    "lots": [], "volume": 0.0}

        # 3. Spread check
        if spread_points > MAX_SPREAD_POINTS:
            return {"can_trade": False, "reason": f"Spread {spread_points:.2f}pts too wide",
                    "lots": [], "volume": 0.0}

        # 4. LBMA fix window
        if strategy != "SCALP" and is_lbma_fix_time():
            return {"can_trade": False, "reason": "LBMA gold fix window — paused",
                    "lots": [], "volume": 0.0}

        # 5. News pause
        if news_pause:
            return {"can_trade": False, "reason": "High-impact news pause active",
                    "lots": [], "volume": 0.0}

        # 6. ATR spike detection
        atr_spike = (atr > 0 and avg_atr > 0 and atr > avg_atr * ATR_SPIKE_MULTIPLIER)

        # 7. Session multiplier
        now_hour    = datetime.now(timezone.utc).hour
        is_vol_zone = 12 <= now_hour <= 16   # London/NY overlap
        is_asian    = now_hour >= 22 or now_hour < 7
        session_mult = 1.25 if is_vol_zone else (0.5 if is_asian else 1.0)

        if is_asian:
            logger.info(f"[GoldRisk] Asian Session: risk halved")
        elif is_vol_zone:
            logger.info(f"[GoldRisk] London/NY Overlap: risk +25%")

        # 8. Calculate SL
        sl_val = max(float(atr) * 1.2, MIN_SL_POINTS)

        # 9. Calculate pyramid lot plan
        lots = self.calculate_pyramid_lots(
            balance          = balance,
            signal_strength  = signal_strength,
            sl_points        = sl_val,
            atr_spike        = atr_spike,
            session_mult     = session_mult,
            open_gold_trades = open_gold_pos,
        )

        if not lots:
            return {"can_trade": False, "reason": "No lots calculated — capacity full",
                    "lots": [], "volume": 0.0}

        tier = self._get_tier(balance)
        risk_usd = balance * (tier["risk_pct"] / 100) * session_mult

        logger.info(
            f"[GoldRisk] ✅ TRADE APPROVED | balance=${balance:.2f} "
            f"| lots={lots} | sl={sl_val:.2f}pts | risk=${risk_usd:.2f}"
        )

        return {
            "can_trade":   True,
            "lots":        lots,
            "volume":      lots[0],          # anchor lot for backward compat
            "sl_value":    sl_val,
            "risk_usd":    risk_usd,
            "risk_pct":    tier["risk_pct"],
            "n_orders":    len(lots),
            "is_vol_zone": is_vol_zone,
            "tier":        tier["name"],
        }

    # ── Utility ───────────────────────────────────────────────────────

    def is_weekend(self) -> bool:
        return datetime.now(timezone.utc).weekday() >= 5

    def check_funded_consistency(self, this_trade_profit: float, total_day_profit: float) -> bool:
        if total_day_profit <= 0:
            return True
        return this_trade_profit / total_day_profit <= 0.30

    def stats(self) -> dict:
        key   = _today()
        daily = _load_daily()
        day_stats = daily.get(key, {"loss": 0.0, "profit": 0.0, "trades": 0})
        return {
            "daily_loss": day_stats.get("loss", 0.0),
            "daily_profit": day_stats.get("profit", 0.0),
            "trades": day_stats.get("trades", 0)
        }
