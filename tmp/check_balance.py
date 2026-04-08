
import MetaTrader5 as mt5
import sys

def check():
    if not mt5.initialize():
        print(f"initialize() failed, error code = {mt5.last_error()}")
        return

    account = 336225828
    password = "Hari@123Arun"
    server = "XMGlobal-MT5 9"

    authorized = mt5.login(account, password=password, server=server)
    if authorized:
        info = mt5.account_info()
        if info:
            print(f"BALANCE: {info.balance}")
            print(f"EQUITY: {info.equity}")
            print(f"PROFIT: {info.profit}")
        else:
            print("Failed to get account info")
    else:
        print(f"Failed to connect to account #{account}, error code = {mt5.last_error()}")

    mt5.shutdown()

if __name__ == "__main__":
    check()
