import yfinance as yf
import time

def test_yield():
    print("Fetching US 10Y Yield (^TNX)...")
    try:
        ticker = yf.Ticker("^TNX")
        data = ticker.history(period="1d", interval="1m")
        if not data.empty:
            cur_yield = data['Close'].iloc[-1]
            prev_yield = data['Close'].iloc[-2]
            change = cur_yield - prev_yield
            print(f"Current Yield: {cur_yield:.4f}% | Change: {change:+.4f}%")
        else:
            print("No data found.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_yield()
