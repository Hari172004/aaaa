"""
macro_monitor.py — Real-time tracking of Interest Rates (^TNX) and USD Index (DXY)
=============================================================================
Provides a directional bias for Gold (XAUUSD) scalping based on macro correlations.
"""

import logging
import threading
import time
from datetime import datetime, timezone

# Lazy import inside the class to prevent startup hang
yf = None

logger = logging.getLogger("agniv.macro_monitor")

class MacroMonitor:
    def __init__(self, interval_seconds: int = 60):
        self.interval = interval_seconds
        self.last_tnx = 0.0
        self.last_dxy = 0.0
        self.tnx_momentum = "Neutral"  # "Rising", "Falling", "Neutral"
        self.dxy_momentum = "Neutral"
        self.macro_bias = "Neutral"     # "Bullish" (Gold Up), "Bearish" (Gold Down)
        
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        
    def start(self):
        """Start the background monitor."""
        t = threading.Thread(target=self._monitor_loop, daemon=True, name="macro-monitor")
        t.start()
        logger.info("[Macro] Macro-Monitor service started (polled every 60s)")

    def stop(self):
        self._stop_event.set()

    def _monitor_loop(self):
        while not self._stop_event.is_set():
            try:
                self._update_data()
            except Exception as e:
                logger.error(f"[Macro] Error fetching yields: {e}")
            self._stop_event.wait(self.interval)

    def _update_data(self):
        global yf
        if yf is None:
            try:
                import yfinance as _yf
                yf = _yf
                logger.info("[Macro] yfinance library loaded in background.")
            except Exception as e:
                logger.warning(f"[Macro] Could not load yfinance: {e}")
                return

        # 1. Fetch TNX (10Y Yield)
        tnx_ticker = yf.Ticker("^TNX")
        tnx_hist = tnx_ticker.history(period="1d", interval="1m")
        
        # 2. Fetch DXY (USD Index - using UUP as proxy if ^DXY is slow)
        dxy_ticker = yf.Ticker("UUP") # UUP is the Dollar Index ETF
        dxy_hist = dxy_ticker.history(period="1d", interval="1m")
        
        with self._lock:
            if not tnx_hist.empty:
                cur_tnx = float(tnx_hist['Close'].iloc[-1])
                prev_tnx = float(tnx_hist['Close'].iloc[-2]) if len(tnx_hist) > 1 else cur_tnx
                diff = cur_tnx - prev_tnx
                
                if diff > 0.001: self.tnx_momentum = "Rising"
                elif diff < -0.001: self.tnx_momentum = "Falling"
                else: self.tnx_momentum = "Neutral"
                
                self.last_tnx = cur_tnx
                
            if not dxy_hist.empty:
                cur_dxy = float(dxy_hist['Close'].iloc[-1])
                prev_dxy = float(dxy_hist['Close'].iloc[-2]) if len(dxy_hist) > 1 else cur_dxy
                diff_d = cur_dxy - prev_dxy
                
                if diff_d > 0.001: self.dxy_momentum = "Rising"
                elif diff_d < -0.001: self.dxy_momentum = "Falling"
                else: self.dxy_momentum = "Neutral"
                
                self.last_dxy = cur_dxy

            # Determine Gold Bias (Gold is Inverse to Yields and Dollar)
            # Bullish Gold = Falling Yields AND Falling Dollar
            if self.tnx_momentum == "Falling" or self.dxy_momentum == "Falling":
                self.macro_bias = "Bullish"
            elif self.tnx_momentum == "Rising" or self.dxy_momentum == "Rising":
                self.macro_bias = "Bearish"
            else:
                self.macro_bias = "Neutral"

    def get_status(self) -> dict:
        with self._lock:
            return {
                "tnx": self.last_tnx,
                "dxy": self.last_dxy,
                "tnx_momentum": self.tnx_momentum,
                "bias": self.macro_bias
            }
