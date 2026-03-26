import sys
import os
import logging

# Add the project root to sys.path
sys.path.append(os.getcwd())

from core import AgniVBot, BotConfig, ASSETS_XAUUSD

# Suppress noisy logs
logging.getLogger("agniv").setLevel(logging.WARNING)

def check_current_status():
    print("Checking current Agni-V status...")
    
    # Minimal config to initialize components
    config = BotConfig(
        mt5_account  = int(os.getenv("MT5_ACCOUNT", "0")),
        mt5_password = os.getenv("MT5_PASSWORD", ""),
        mt5_server   = os.getenv("MT5_SERVER", ""),
        sniper_mode  = os.getenv("SNIPER_MODE", "true").lower() == "true"
    )
    
    bot = AgniVBot(config)
    
    try:
        # Connect to MT5
        if not bot.mt5.connect(config.mt5_account, config.mt5_password, config.mt5_server):
            print("Failed to connect to MT5.")
            return

        # Get data and generate signal logic (simulated)
        symbol = config.assets if config.assets != "BOTH" else ASSETS_XAUUSD
        df = bot.mt5.get_ohlcv(symbol, "M5", 300)
        
        if df.empty:
            print("No data received from MT5.")
            return

        from analysis.gold_indicators import calculate_gold_indicators
        df = calculate_gold_indicators(df)
        print(f"\n--- Data Snapshot (Tail 5) ---")
        print(df[['open', 'high', 'low', 'close']].tail(5))
        last = df.iloc[-1]
        
        width = last['bb_width']
        avg_width = df['bb_width'].rolling(50).mean().iloc[-1]
        squeeze_ratio = width / avg_width if avg_width > 0 else 0
        
        print(f"\n--- Agni-V Diagnostic Report (XAUUSD M5) ---")
        print(f"Current Price:  {last['close']:.2f}")
        print(f"BB Width:       {width:.6f}")
        print(f"BB Width Avg:   {avg_width:.6f}")
        print(f"Squeeze Ratio:  {squeeze_ratio:.2%}")
        print(f"Squeeze Active: {'YES 🔴 (Waiting for breakout)' if last['bb_squeeze'] else 'NO 🟢'}")
        print(f"ATR (14):       {last['atr']:.4f}")
        print(f"RSI (14):       {last['rsi']:.1f}")
        print(f"----------------------------------------------")
        
        live_close_val = float(last['close'])
        
        # H1 Trend
        df_h1 = bot.mt5.get_ohlcv(symbol, "H1", 200)
        if not df_h1.empty:
            from analysis.gold_indicators import calculate_gold_indicators
            df_h1 = calculate_gold_indicators(df_h1)
            h1_last = df_h1.iloc[-1]
            ema100_h1 = h1_last.get("ema_100", h1_last.get("ema_200", 0))
            h1_trend = "BULLISH" if h1_last["close"] > ema100_h1 else "BEARISH"
            print(f"H1 Trend:      {h1_trend} (Price: {h1_last['close']:.2f}, EMA100: {ema100_h1:.2f})")

        # Check Strategy Reasons
        res = bot.gold_scalp.generate_signal(df, df_h1=df_h1, is_sniper=config.sniper_mode)
        print(f"Strategy Signal: {res['signal']}")
        print(f"Strategy Reason: {res['reason']}")

        # SMC Details
        from analysis.gold_market_structure import detect_gold_smc
        smc = detect_gold_smc(df)
        print(f"\n--- SMC Zones Found ---")
        print(f"Bullish OBs: {len(smc['bull_obs'])}")
        for ob in smc['bull_obs']:
            dist = live_close_val - ob['top']
            print(f"  - Bull OB at {ob['bottom']:.2f}-{ob['top']:.2f} (Dist: {dist:.2f})")
        
        print(f"Bearish OBs: {len(smc['bear_obs'])}")
        for ob in smc['bear_obs']:
            dist = ob['bottom'] - live_close_val
            print(f"  - Bear OB at {ob['bottom']:.2f}-{ob['top']:.2f} (Dist: {dist:.2f})")
        
        from analysis.gold_sessions import get_current_gold_session
        sess = get_current_gold_session()
        
        from datetime import datetime, timezone
        now_hour = datetime.now(timezone.utc).hour
        is_london = 7 <= now_hour <= 10
        is_ny     = 13 <= now_hour <= 17
        
        print(f"\nSession: {sess['active_kz']} (London: {is_london}, NY: {is_ny})")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        bot.mt5.disconnect()

if __name__ == "__main__":
    check_current_status()
