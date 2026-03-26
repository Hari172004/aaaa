import MetaTrader5 as mt5
import pandas as pd

def check_symbols():
    if not mt5.initialize():
        print("Failed to initialize MT5")
        return

    print("--- Symbol Check ---")
    symbols = ["XAUUSD", "GOLD", "XAUUSD.m", "XAUUSD.i", "XAUUSD.pro"]
    for sym in symbols:
        info = mt5.symbol_info(sym)
        if info:
            print(f"{sym}: Bid={info.bid} | Description={info.description} | Digits={info.digits}")
        else:
            print(f"{sym}: Not Found")
            
    print("\n--- All XAU/GOLD Symbols ---")
    all_symbols = mt5.symbols_get()
    for s in all_symbols:
        name = s.name.upper()
        if "XAU" in name or "GOLD" in name:
            print(f"{s.name}: Bid={s.bid} | Description={s.description}")

    mt5.shutdown()

if __name__ == "__main__":
    check_symbols()
