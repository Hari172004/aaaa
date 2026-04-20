import MetaTrader5 as mt5
import logging

def test():
    print("Testing bare init...")
    res = mt5.initialize()
    print(f"Bare Init Result: {res}")
    if res:
        print(f"Account: {mt5.account_info().login}")
    else:
        print(f"Error: {mt5.last_error()}")
    mt5.shutdown()

    print("\nTesting init with path...")
    res2 = mt5.initialize(path=r"C:\Program Files\MetaTrader 5 EXNESS\terminal64.exe")
    print(f"Path Init Result: {res2}")
    if res2:
        print(f"Account: {mt5.account_info().login}")
    else:
        print(f"Error: {mt5.last_error()}")
    mt5.shutdown()

test()
