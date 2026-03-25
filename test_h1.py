import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import os

def test_h1_history():
    if not mt5.initialize():
        print("MT5 initialize() failed")
        return

    symbol = "XAUUSD"
    for s in ["XAUUSD", "GOLD", "XAUUSD.a", "XAUUSDm"]:
        if mt5.symbol_select(s, True):
            symbol = s
            break
    
    print(f"Testing H1 history for {symbol} from 2001...")
    utc_from = datetime(2001, 1, 1)
    utc_to = datetime.now()
    
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, utc_from, utc_to)
    
    if rates is None or len(rates) == 0:
        print("Failed to get H1 rates for that range.")
    else:
        print(f"Successfully retrieved {len(rates)} H1 bars.")
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/gold_h1_history_temp.csv", index=False)

    mt5.shutdown()

if __name__ == "__main__":
    test_h1_history()
