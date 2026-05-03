# Binance EMA Crossover + Trend Bot

Bu bot, `python-binance`, `pandas`, `pandas-ta` kullanarak aşağıdaki stratejiyi uygular:

- Ana zaman dilimi: `M5`
- Trend filtre: `M15` ve `H1` üzerinde `EMA200`
- Kesişim: `EMA13` / `EMA21`
- Trend eğimi: M5 `EMA200(current) > EMA200(5 mum önce)` (LONG için)
- SHORT tarafı LONG koşullarının tersidir.
- SL: Son 3 mum seviyesi ve EMA21 %0.5 kuralına göre
- TP: Minimum RR `1:1.5` + ters kesişimde çıkış

## Kurulum

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ortam Değişkenleri

Proje kökünde `.env` dosyası kullanılır. Örnek:

```bash
BINANCE_API_KEY=YOUR_KEY
BINANCE_API_SECRET=YOUR_SECRET
SYMBOL=BTCUSDT
MARKET_TYPE=futures
BALANCE_USAGE_PCT=0.75
LEVERAGE=max
REST_LIMIT=2000
DRY_RUN=true
TESTNET=false

# Backtest (opsiyonel)
BT_START=2024-01-01
BT_END=2024-12-31
BT_INITIAL_BALANCE=1000
BT_FEE_RATE=0.0004
```

`LEVERAGE=max` verildiğinde bot, sembolün Binance tarafından izin verilen en yüksek kaldıraç değerini otomatik kullanır.

## Çalıştırma

```bash
python ema_crossover_bot.py
```

## Backtest

Backtest scripti: `backtest_ema_crossover.py`

```bash
python backtest_ema_crossover.py
```

Parametreli örnek:

```bash
python backtest_ema_crossover.py \
  --symbol BTCUSDT \
  --market-type futures \
  --start 2024-01-01 \
  --end 2025-01-01 \
  --initial-balance 1000 \
  --fee-rate 0.0004 \
  --balance-usage 0.75 \
  --leverage max
```

Backtest sonunda maksimum kar/zarar metrikleri konsola yazdırılır.

- İşlem listesi: `backtest_trades.csv`
- Bakiye tükenme olayları: `backtest_depletion_events.csv`

## Notlar

- Bot başlangıçta her zaman dilimi için en az `2000` mum REST ile çeker.
- Canlı güncellemeler için `User Data Stream` + `Kline Stream` kullanır.
- WebSocket hata/kopma durumunda otomatik reconnect yapar.
- Spot modunda SHORT işlemler atlanır.
- Gerçek para ile kullanmadan önce testnet ve `DRY_RUN=true` ile doğrulama yapın.
