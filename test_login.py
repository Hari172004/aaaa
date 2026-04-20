import MetaTrader5 as mt5
import os

def test():
    print("Testing connection with NO plus sign...")
    # Initialize with the explicit path to Exness, but without the + in server name
    path = r"C:\Program Files\MetaTrader 5 EXNESS\terminal64.exe"
    
    # 1. Start terminal headless first to see if it links up
    init1 = mt5.initialize(path=path)
    print(f"Base initialize: {init1}")
    
    if init1:
        # 2. Login directly
        login = mt5.login(413670633, password="Forex123$", server="Exness-MT5Trial6")
        print(f"Login result: {login}")
        if login:
            print(f"Successfully connected! Balance: {mt5.account_info().balance}")
        else:
            print(f"Login failed: {mt5.last_error()}")
            
    mt5.shutdown()

test()
