import pandas as pd

df = pd.read_csv("backtest_trades.csv")
df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
df["month"] = df["exit_time"].dt.to_period("M")

initial_balance = 1000.0

monthly = df.groupby("month").agg(
    net_pnl=("net_pnl", "sum"),
    trades=("net_pnl", "count"),
    wins=("net_pnl", lambda x: (x > 0).sum()),
).reset_index()

monthly["pct_return"] = (monthly["net_pnl"] / initial_balance) * 100
monthly["win_rate"]   = (monthly["wins"] / monthly["trades"] * 100).round(1)

print("=== AYLIK ANALIZ (sabit 1000 USD baslangic) ===")
print(f"{'Ay':<10}  {'Net PnL':>10}  {'Return':>8}  {'Islem':>6}  {'WR':>6}")
print("-" * 50)
for _, r in monthly.iterrows():
    sign = "+" if r["pct_return"] >= 0 else ""
    print(
        f"{str(r['month']):<10}  {r['net_pnl']:>+10.2f}  "
        f"{sign}{r['pct_return']:>7.2f}%  {int(r['trades']):>6}  {r['win_rate']:>5.1f}%"
    )

avg_monthly       = monthly["pct_return"].mean()
median_monthly    = monthly["pct_return"].median()
profitable_months = int((monthly["pct_return"] > 0).sum())
total_months      = len(monthly)
best  = monthly.loc[monthly["pct_return"].idxmax()]
worst = monthly.loc[monthly["pct_return"].idxmin()]

print("-" * 50)
print(f"Ortalama aylik kar   : {avg_monthly:+.2f}%  (~{avg_monthly/100*initial_balance:+.2f} USD/ay)")
print(f"Medyan aylik kar     : {median_monthly:+.2f}%  (~{median_monthly/100*initial_balance:+.2f} USD/ay)")
print(f"Karli ay sayisi      : {profitable_months}/{total_months}")
print(f"En iyi ay            : {best['month']} -> {best['pct_return']:+.2f}%")
print(f"En kotu ay           : {worst['month']} -> {worst['pct_return']:+.2f}%")

# --- Bileşik büyüme simülasyonu (gerçek aylık return'ler ile) ---
print("\n=== BİLEŞİK BÜYÜME SİMÜLASYONU (100 USD başlangıç, para çekilmiyor) ===")
print(f"{'Ay':<10}  {'Return':>8}  {'Bakiye':>12}  {'Kazanç':>12}")
print("-" * 50)
balance_c = 100.0
for _, r in monthly.iterrows():
    ret = r["pct_return"] / 100.0
    gain = balance_c * ret
    balance_c += gain
    sign = "+" if gain >= 0 else ""
    print(f"{str(r['month']):<10}  {r['pct_return']:>+7.2f}%  {balance_c:>12.2f} USD  {sign}{gain:>10.2f} USD")

print("-" * 50)
total_gain = balance_c - 100.0
total_ret  = (balance_c / 100.0 - 1) * 100
avg_monthly_compound = ((balance_c / 100.0) ** (1 / len(monthly)) - 1) * 100
print(f"Başlangıç            : 100.00 USD")
print(f"Bitiş                : {balance_c:.2f} USD")
print(f"Toplam kazanç        : +{total_gain:.2f} USD  (+{total_ret:.1f}%)")
print(f"Geometrik aylık ort  : +{avg_monthly_compound:.2f}%")
print(f"Süre                 : {len(monthly)} ay")

# Projeksiyon: sabit geometrik ortalama ile gelecek 12 ay
print(f"\n=== PROJEKSIYON ({avg_monthly_compound:.2f}% aylık bileşik, 100 USD başlangıç) ===")
print(f"{'Ay':>4}  {'Bakiye':>12}  {'Toplam Kazanç':>14}")
b = 100.0
for m in range(1, 13):
    b *= (1 + avg_monthly_compound / 100)
    print(f"{m:>4}  {b:>12.2f} USD  +{b - 100:.2f} USD")
