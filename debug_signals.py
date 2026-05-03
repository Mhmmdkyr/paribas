import os
import pandas as pd
from binance.client import Client
from dotenv import load_dotenv

load_dotenv()

api_key    = os.getenv("BINANCE_API_KEY", "")
api_secret = os.getenv("BINANCE_API_SECRET", "")
client = Client(api_key, api_secret)

symbol = "SOLUSDT"
start  = "2025-01-01"
end    = "2025-04-27"

raw = client.futures_historical_klines(symbol=symbol, interval="5m", start_str=start, end_str=end)
df = pd.DataFrame(raw, columns=[
    "open_time","open","high","low","close","volume",
    "close_time","quote_asset_volume","number_of_trades",
    "taker_buy_base_asset_volume","taker_buy_quote_asset_volume","ignore",
])
for col in ["open","high","low","close"]:
    df[col] = pd.to_numeric(df[col])
df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

df["ema8"]   = df["close"].ewm(span=8,   adjust=False).mean()
df["ema13"]  = df["close"].ewm(span=13,  adjust=False).mean()
df["ema21"]  = df["close"].ewm(span=21,  adjust=False).mean()
df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

# EMA200 from M15 and H1
raw_m15 = client.futures_historical_klines(symbol=symbol, interval="15m", start_str=start, end_str=end)
df_m15 = pd.DataFrame(raw_m15, columns=[
    "open_time","open","high","low","close","volume",
    "close_time","quote_asset_volume","number_of_trades",
    "taker_buy_base_asset_volume","taker_buy_quote_asset_volume","ignore",
])
df_m15["close"] = pd.to_numeric(df_m15["close"])
df_m15["close_time"] = pd.to_datetime(df_m15["close_time"], unit="ms", utc=True)
df_m15["ema200_m15"] = df_m15["close"].ewm(span=200, adjust=False).mean()
df_m15["ema13_m15"]  = df_m15["close"].ewm(span=13,  adjust=False).mean()
df_m15["ema21_m15"]  = df_m15["close"].ewm(span=21,  adjust=False).mean()
df_m15 = df_m15.set_index("close_time").sort_index()

raw_h1 = client.futures_historical_klines(symbol=symbol, interval="1h", start_str=start, end_str=end)
df_h1 = pd.DataFrame(raw_h1, columns=[
    "open_time","open","high","low","close","volume",
    "close_time","quote_asset_volume","number_of_trades",
    "taker_buy_base_asset_volume","taker_buy_quote_asset_volume","ignore",
])
df_h1["close"] = pd.to_numeric(df_h1["close"])
df_h1["close_time"] = pd.to_datetime(df_h1["close_time"], unit="ms", utc=True)
df_h1["ema200_h1"] = df_h1["close"].ewm(span=200, adjust=False).mean()
df_h1 = df_h1.set_index("close_time").sort_index()

df = df.set_index("close_time").sort_index()
df["ema200_m15"] = df_m15["ema200_m15"].reindex(df.index, method="ffill")
df["ema200_h1"]  = df_h1["ema200_h1"].reindex(df.index, method="ffill")

df = df.dropna(subset=["ema200_m15","ema200_h1"]).reset_index()

print(f"Toplam mum: {len(df)}")

# Count signals step by step
goldens = 0
deaths  = 0
long_with_trend = 0
long_with_slope = 0
long_final = 0
long_confirm = 0

SLOPE_BARS = 3

for i in range(SLOPE_BARS + 2, len(df)):
    prev = df.iloc[i - 1]
    curr = df.iloc[i]
    price = float(curr["close"])

    trend_up   = price > float(curr["ema200_m15"]) and price > float(curr["ema200_h1"])
    slope_ref  = float(df.iloc[i - SLOPE_BARS]["ema200"])
    slope_up   = float(curr["ema200"]) > slope_ref

    # fast EMA = True => ema8/ema13
    fast_prev_a, fast_prev_b = float(prev["ema8"]),  float(prev["ema13"])
    fast_curr_a, fast_curr_b = float(curr["ema8"]),  float(curr["ema13"])

    golden = fast_prev_a <= fast_prev_b and fast_curr_a > fast_curr_b
    if golden:
        goldens += 1

    long_signal = trend_up and slope_up and golden and price > float(curr["ema200"])

    if trend_up and slope_up and golden:
        long_with_slope += 1

    if golden and trend_up:
        long_with_trend += 1

    if long_signal:
        long_final += 1

    # entry_confirm_bar
    if long_signal and i >= 2:
        pprev = df.iloc[i - 2]
        pg_a, pg_b = float(pprev["ema8"]), float(pprev["ema13"])
        cg_a, cg_b = float(prev["ema8"]),  float(prev["ema13"])
        curr_a, curr_b = float(curr["ema8"]), float(curr["ema13"])
        prev_golden  = pg_a <= pg_b and cg_a > cg_b
        still_golden = cg_a > cg_b and curr_a > curr_b
        if prev_golden and still_golden:
            long_confirm += 1

print(f"EMA8 > EMA13 crossovers (golden): {goldens}")
print(f"Golden + trend_up:                {long_with_trend}")
print(f"Golden + trend_up + slope_up:     {long_with_slope}")
print(f"Final long signals (+ ema200):    {long_final}")
print(f"After entry_confirm_bar:          {long_confirm}")
