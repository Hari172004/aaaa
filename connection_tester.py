import MetaTrader5 as mt5
import os
from dotenv import load_dotenv

def test_connection():
    load_dotenv(override=True)
    account = int(os.getenv("MT5_ACCOUNT", "0"))
    password = os.getenv("MT5_PASSWORD", "")
    server = os.getenv("MT5_SERVER", "")
    path = os.getenv("MT5_PATH", "")
    
    print(f"--- MetaTrader 5 Connection Diagnostics ---")
    print(f"Account:  {account}")
    print(f"Server:   {server}")
    print(f"Path:     {path}")
    print(f"Password: {'*' * len(password)}")
    print(f"-------------------------------------------")
    
    servers = [
        server,
        "XMGlobal-MT5 9",
        "XMGlobal-MT5 5",
        "XMGlobal-MT5",
        "XMGlobal-MT9",
        "XMGlobal-Real 9",
        "XMGlobal-Real 36"
    ]
    
    print(f"Testing connection for Account: {account}")
    
    for server in servers:
        print(f"--- Trying Server: '{server}' ---")
        # Try initialize with params
        if mt5.initialize(login=account, password=password, server=server):
            print(f"SUCCESS with initialize(params) on {server}")
            info = mt5.account_info()
            if info:
                print(f"Account Info: {info.login}, {info.server}, Balance: {info.balance}")
            mt5.shutdown()
            return
        else:
            err = mt5.last_error()
            print(f"FAILED initialize(params) on {server}: {err}")
            mt5.shutdown()
            
        # Try generic initialize + login
        if mt5.initialize():
            if mt5.login(account, password=password, server=server):
                print(f"SUCCESS with login() on {server}")
                mt5.shutdown()
                return
            else:
                print(f"FAILED login() on {server}: {mt5.last_error()}")
            mt5.shutdown()
    
    print("\nFinal Result: Could not connect with any known server combinations.")

if __name__ == "__main__":
    test_connection()
