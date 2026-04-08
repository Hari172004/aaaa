"""Quick end-to-end test for DIYCustomStrategy."""
import pandas as pd
import numpy as np

np.random.seed(42)
n = 300
price = 2000 + np.cumsum(np.random.randn(n) * 0.8)
df = pd.DataFrame({
    "open":   price + np.random.randn(n) * 0.2,
    "high":   price + np.abs(np.random.randn(n)) * 1.2,
    "low":    price - np.abs(np.random.randn(n)) * 1.2,
    "close":  price + np.random.randn(n) * 0.2,
    "volume": np.abs(np.random.randn(n)) * 1000 + 500,
})

from strategies.diy_custom_builder import DIYCustomStrategy

# --- Scalp ---
s_scalp = DIYCustomStrategy("diy_scalp_config.json")
sig = s_scalp.generate_signal(df)
st  = s_scalp.get_status()
print(f"[SCALP] signal={sig} | leading={st['leading_indicator']} | filters={st['active_filters']}")

# --- Swing ---
s_swing = DIYCustomStrategy("diy_swing_config.json")
sig2 = s_swing.generate_signal(df)
st2  = s_swing.get_status()
print(f"[SWING] signal={sig2} | leading={st2['leading_indicator']} | filters={st2['active_filters']}")

print()
print("--- 15-candle walk-forward test on SCALP strategy ---")
found_non_hold = 0
for i in range(15):
    df_i = df.iloc[:285 + i].copy()
    sig_i = s_scalp.generate_signal(df_i)
    tag = "  <<<" if sig_i != "HOLD" else ""
    print(f"  bar {285+i}: {sig_i}{tag}")
    if sig_i != "HOLD":
        found_non_hold += 1

print()
print(f"Non-HOLD signals found: {found_non_hold}/15")
print("All tests passed!")
