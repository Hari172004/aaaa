import numpy as np
import pandas as pd

class BullByteEngine:
    """
    Ultimate Scalping Engine inspired by BullByte tools.
    Combines:
    1. Kinetic VWMA Ribbon (EMA vs VWMA)
    2. Price Memory Heatmaps (High Volume Nodes)
    3. Volume Flow Pressure (Oscillator)
    """

    @staticmethod
    def evaluate(df: pd.DataFrame, lookback: int = 150) -> dict:
        """
        Evaluates the current candle for a BullByte Ultimate Snipe.
        Returns: {"buy": bool, "sell": bool, "memory_levels": list}
        """
        if len(df) < lookback + 5:
            return {"buy": False, "sell": False, "memory_levels": []}

        try:
            # 1. Kinetic Momentum (VWMA vs EMA)
            # ---------------------------------
            ema_9 = df['close'].ewm(span=9, adjust=False).mean()
            
            # VWMA 20: sum(Price * Volume) / sum(Volume)
            pv = df['close'] * df['volume']
            vwma_20 = pv.rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
            
            # 2. Volume Flow Pressure
            # -----------------------
            vol_ema_fast = df['volume'].ewm(span=5, adjust=False).mean()
            vol_ema_slow = df['volume'].ewm(span=20, adjust=False).mean()
            volume_push = vol_ema_fast > vol_ema_slow

            # 3. Price Memory Heatmap (HVN - High Volume Nodes)
            # -------------------------------------------------
            # Exclude exactly current live candle for memory build
            window = df.iloc[-lookback-1:-1] 
            
            min_p = window['low'].min()
            max_p = window['high'].max()
            
            # Create 10 dynamic price bins
            bins = np.linspace(min_p, max_p, 11) 
            bin_volumes = np.zeros(10)
            
            # Sum volume per bin (using fast Numpy arrays)
            typ_prices = (window['high'].values + window['low'].values + window['close'].values) / 3.0
            volumes = window['volume'].values
            
            # digitize returns 1 to num_bins if inside range
            bin_indices = np.digitize(typ_prices, bins) - 1
            bin_indices = np.clip(bin_indices, 0, 9)
            
            # Add volumes to corresponding bins
            for i in range(len(bin_indices)):
                bin_volumes[bin_indices[i]] += volumes[i]
                
            # Get indices of top 3 high volume nodes
            top_3_idx = np.argsort(bin_volumes)[-3:]
            
            # Convert bin indices back to center price levels
            memory_levels = []
            for b in top_3_idx:
                center_price = (bins[b] + bins[b+1]) / 2.0
                memory_levels.append(float(center_price))

            # 4. Synthesizer (The Trigger)
            # ----------------------------
            last = df.iloc[-1]
            prev = df.iloc[-2]
            price = float(last['close'])
            
            curr_ema9    = float(ema_9.iloc[-1])
            curr_vwma20  = float(vwma_20.iloc[-1])
            curr_vol_push= bool(volume_push.iloc[-1])
            
            prev_ema9    = float(ema_9.iloc[-2])
            prev_vwma20  = float(vwma_20.iloc[-2])

            # Kinetic Cross
            kinetic_buy  = (curr_ema9 > curr_vwma20) and (prev_ema9 <= prev_vwma20)
            kinetic_sell = (curr_ema9 < curr_vwma20) and (prev_ema9 >= prev_vwma20)
            
            # Sustained Kinetic Trend
            kinetic_bull_trend = curr_ema9 > curr_vwma20
            kinetic_bear_trend = curr_ema9 < curr_vwma20

            # Memory Retest Check
            # Check if price is extremely close to a high-volume memory node
            # Gold usually sweeps these by ~0.10% distance. Crypto by maybe 0.15%.
            dev = price * 0.0012 
            near_memory = any(abs(price - level) <= dev for level in memory_levels)

            # Heiken-Ashi style confirmation (just basic directional flow for safety)
            bull_flow = last['close'] > last['open'] and last['close'] > prev['close']
            bear_flow = last['close'] < last['open'] and last['close'] < prev['close']

            # Ultimate Trigger
            is_buy  = (kinetic_buy and curr_vol_push) or (near_memory and curr_vol_push and kinetic_bull_trend and bull_flow)
            is_sell = (kinetic_sell and curr_vol_push) or (near_memory and curr_vol_push and kinetic_bear_trend and bear_flow)

            return {
                "buy": bool(is_buy),
                "sell": bool(is_sell),
                "memory_levels": memory_levels
            }

        except Exception as e:
            # Safe failover
            return {"buy": False, "sell": False, "memory_levels": []}
