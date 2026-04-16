"""
gold_sessions.py -- Gold session detector with Kill Zone timing and LBMA avoidance
"""

from datetime import datetime, timezone
import logging

logger = logging.getLogger("agniv.gold_sessions")

# All times in UTC (equivalent to GMT)
SESSIONS = {
    "ASIAN":     (0,  7),
    "LONDON":    (7, 12),
    "NEW_YORK": (12, 20),
    "LATENIGHT":(20, 24),
}

# Kill Zones — highest probability entry windows for Gold
LONDON_KZ_START = 7
LONDON_KZ_END   = 10
NY_KZ_START     = 12
NY_KZ_END       = 15

# LBMA official gold fix times (London time = UTC+0 in winter, UTC+1 in summer)
# 10:30 AM London = 10:30 UTC (winter) / 09:30 UTC (summer)
# 3:00  PM London = 15:00 UTC (winter) / 14:00 UTC (summer)
LBMA_FIX_WINDOWS = [
    (10, 11),   # AM fix window (avoid 10:00-11:00 UTC)
    (14, 16),   # PM fix window (avoid 14:00-16:00 UTC)
]

# Pre-session sniper window (minutes before open)
PRE_SESSION_BUFFER = 15
# Session-open "Washout" guard (minutes after open to block)
WASHOUT_PERIOD     = 5


def _utc_hour() -> int:
    return datetime.now(timezone.utc).hour


def _utc_minute() -> int:
    return datetime.now(timezone.utc).minute


def get_current_gold_session() -> dict:
    """
    Returns full session context for the current UTC time.
    """
    hour   = _utc_hour()
    minute = _utc_minute()

    # Identify session
    session = "LATENIGHT"
    for name, (start, end) in SESSIONS.items():
        if start <= hour < end:
            session = name
            break

    is_london_kz = LONDON_KZ_START <= hour < LONDON_KZ_END
    is_ny_kz     = NY_KZ_START     <= hour < NY_KZ_END
    is_killzone  = is_london_kz or is_ny_kz

    active_kz = "NONE"
    if is_london_kz:
        active_kz = "LONDON"
    elif is_ny_kz:
        active_kz = "NY"

    # LBMA fix avoidance
    is_lbma_fix = any(start <= hour < end for start, end in LBMA_FIX_WINDOWS)

    # Minutes until next kill zone
    if hour < LONDON_KZ_START:
        mins_to_kz = (LONDON_KZ_START - hour) * 60 - minute
    elif hour < NY_KZ_START and not is_ny_kz:
        mins_to_kz = (NY_KZ_START - hour) * 60 - minute
    else:
        mins_to_kz = 0 if is_killzone else 999

    # Mapping for ML features
    session_map = {"ASIAN": 1, "LONDON": 2, "NEW_YORK": 3, "LATENIGHT": 4}

    # ── Pre-Session Sniper Logic ──
    # Starts 15 mins before London (6:45) and NY (11:45)
    is_pre_london = (hour == 6 and minute >= (60 - PRE_SESSION_BUFFER))
    is_pre_ny     = (hour == 11 and minute >= (60 - PRE_SESSION_BUFFER))
    is_pre_session = is_pre_london or is_pre_ny

    # ── Session-Open Washout Logic ──
    # Blocks first 5 mins of London (7:00-7:05) and NY (12:00-12:05)
    is_washout_london = (hour == 7 and minute < WASHOUT_PERIOD)
    is_washout_ny     = (hour == 12 and minute < WASHOUT_PERIOD)
    is_washout = is_washout_london or is_washout_ny

    return {
        "session":       session,
        "session_id":    session_map.get(session, 1),
        "utc_hour":      hour,
        "utc_minute":    minute,
        "is_killzone":   is_killzone,
        "is_pre_session": is_pre_session,
        "is_washout":    is_washout,
        "active_kz":     active_kz,
        "is_lbma_fix":   is_lbma_fix,
        "is_asian":      session == "ASIAN",
        "mins_to_kz":    mins_to_kz,
    }


def get_current_gold_session_simple() -> dict:
    """Alias for backward compatibility."""
    return get_current_gold_session()


def is_gold_scalp_time(ignore_lbma: bool = False, ignore_asian: bool = False) -> bool:
    """
    True during London/NY Kill Zone OR the Pre-Session window.
    """
    info = get_current_gold_session()
    if info["is_lbma_fix"] and not ignore_lbma:
        logger.info("[Sessions] LBMA fix window active — scalping paused")
        return False
    
    if info["is_asian"] and not info["is_pre_session"] and not ignore_asian:
        return False
        
    return info["is_killzone"] or info["is_pre_session"] or ignore_lbma or ignore_asian


def is_washout_period() -> bool:
    """Returns True during the first 5 minutes of a major session open."""
    return get_current_gold_session()["is_washout"]


def is_lbma_fix_time() -> bool:
    """Returns True during LBMA gold fix windows."""
    return get_current_gold_session()["is_lbma_fix"]


def mins_until_london_open() -> int:
    """Minutes until London session opens (7:00 UTC)."""
    hour   = _utc_hour()
    minute = _utc_minute()
    if hour >= 7:
        return 0
    return (7 - hour) * 60 - minute
