import pandas as pd

df = pd.read_csv("backtest_trades.csv")
df["entry_time"] = pd.to_datetime(df["entry_time"])
df["exit_time"]  = pd.to_datetime(df["exit_time"])
df = df.sort_values("entry_time").reset_index(drop=True)

initial = 1000.0
target_pct = 0.23
target = initial * (1 + target_pct)

df["cumulative_balance"] = initial + df["net_pnl"].cumsum()

hit = df[df["cumulative_balance"] >= target]
if len(hit) > 0:
    first_hit = hit.iloc[0]
    start     = df["entry_time"].iloc[0]
    end       = first_hit["exit_time"]
    days      = (end - start).total_seconds() / 86400
    trade_no  = hit.index[0] + 1
    print(f"Hedef       : %{target_pct*100:.0f} kar -> {target:.2f} USDT")
    print(f"Ulaşılan    : {first_hit['cumulative_balance']:.2f} USDT")
    print(f"Başlangıç   : {start.date()}")
    print(f"Hedefe tarih: {end.date()}")
    print(f"Süre        : {days:.0f} gün  ({days/30:.1f} ay  |  {days/7:.1f} hafta)")
    print(f"Kaçıncı işlem: {trade_no}/{len(df)}")
else:
    last_bal = initial + df["net_pnl"].sum()
    print(f"Hedef ({target:.2f}) hiç ulaşılmadı. Son bakiye: {last_bal:.2f}")

print()
print("=== Aylık Bakiye İlerlemesi ===")
df["month"] = df["exit_time"].dt.to_period("M")
monthly = df.groupby("month").agg(
    trades=("net_pnl", "count"),
    pnl=("net_pnl", "sum"),
    wins=("net_pnl", lambda x: (x > 0).sum()),
).reset_index()

balance = initial
rows = []
for _, row in monthly.iterrows():
    balance += row["pnl"]
    wr = row["wins"] / row["trades"] * 100 if row["trades"] > 0 else 0
    rows.append({
        "Ay": str(row["month"]),
        "Bakiye": f"{balance:.2f}",
        "PnL": f"{row['pnl']:+.2f}",
        "İşlem": int(row["trades"]),
        "WR%": f"{wr:.0f}%",
    })

result = pd.DataFrame(rows)
print(result.to_string(index=False))
