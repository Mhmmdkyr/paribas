import itertools
import logging
import multiprocessing
import os
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from binance.client import Client
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)s | %(message)s")


@dataclass
class OptParams:
    rr_ratio: float
    slope_bars: int
    rr_exit_ratio: float
    reverse_confirm_bars: int
    sl_ema_pct: float


def fetch_klines_df(client: Client, symbol: str, interval: str, start_time: str, end_time: str, market_type: str) -> pd.DataFrame:
    if market_type == "futures":
        raw = client.futures_historical_klines(symbol=symbol, interval=interval, start_str=start_time, end_str=end_time)
    else:
        raw = client.get_historical_klines(symbol=symbol, interval=interval, start_str=start_time, end_str=end_time)

    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore",
    ])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    return df[["open_time", "high", "low", "close", "close_time"]]


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema8"]   = out["close"].ewm(span=8,   adjust=False).mean()
    out["ema13"]  = out["close"].ewm(span=13,  adjust=False).mean()
    out["ema21"]  = out["close"].ewm(span=21,  adjust=False).mean()
    out["ema200"] = out["close"].ewm(span=200, adjust=False).mean()
    delta = out["close"].diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    out["rsi14"] = 100 - 100 / (1 + rs)
    high_low   = out["high"] - out["low"]
    high_close = (out["high"] - out["close"].shift()).abs()
    low_close  = (out["low"]  - out["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    out["atr14"] = tr.ewm(com=13, adjust=False).mean()
    up_move   = out["high"].diff()
    down_move = (-out["low"].diff())
    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr_s    = pd.Series(tr.values, index=out.index)
    plus_di  = 100 * pd.Series(plus_dm,  index=out.index).ewm(com=13, adjust=False).mean() / atr_s.ewm(com=13, adjust=False).mean()
    minus_di = 100 * pd.Series(minus_dm, index=out.index).ewm(com=13, adjust=False).mean() / atr_s.ewm(com=13, adjust=False).mean()
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float("nan")))
    out["adx14"] = dx.ewm(com=13, adjust=False).mean()
    return out


def fetch_merged_frame(client: Client, symbol: str, start_time: str, end_time: str, market_type: str) -> pd.DataFrame:
    print("Veri çekiliyor...")
    m5  = fetch_klines_df(client, symbol, Client.KLINE_INTERVAL_5MINUTE, start_time, end_time, market_type)
    m15 = fetch_klines_df(client, symbol, Client.KLINE_INTERVAL_15MINUTE, start_time, end_time, market_type)
    h1  = fetch_klines_df(client, symbol, Client.KLINE_INTERVAL_1HOUR, start_time, end_time, market_type)

    m5  = add_indicators(m5).dropna().reset_index(drop=True)
    m15 = add_indicators(m15).dropna().reset_index(drop=True)
    h1  = add_indicators(h1).dropna().reset_index(drop=True)

    m15_view = m15[["close_time", "ema200", "ema13", "ema21"]].rename(
        columns={"ema200": "ema200_m15", "ema13": "ema13_m15", "ema21": "ema21_m15"}
    )
    h1_view  = h1[["close_time", "ema200", "rsi14"]].rename(columns={"ema200": "ema200_h1", "rsi14": "rsi14_h1"})

    merged = pd.merge_asof(m5.sort_values("close_time"), m15_view.sort_values("close_time"), on="close_time", direction="backward")
    merged = pd.merge_asof(merged.sort_values("close_time"), h1_view.sort_values("close_time"), on="close_time", direction="backward")
    print(f"Veri hazır: {len(merged)} mum")
    return merged.dropna().reset_index(drop=True)


def run_single(
    arr_close: "np.ndarray",
    arr_high: "np.ndarray",
    arr_low: "np.ndarray",
    arr_ema8: "np.ndarray",
    arr_ema13: "np.ndarray",
    arr_ema21: "np.ndarray",
    arr_ema200: "np.ndarray",
    arr_ema200_m15: "np.ndarray",
    arr_ema13_m15: "np.ndarray",
    arr_ema21_m15: "np.ndarray",
    arr_ema200_h1: "np.ndarray",
    arr_rsi14_h1: "np.ndarray",
    arr_atr14: "np.ndarray",
    arr_adx14: "np.ndarray",
    arr_dates: list,
    leverage: int,
    risk_per_trade_pct: float,
    rr_ratio: float,
    slope_bars: int,
    reverse_confirm_bars: int,
    rr_exit_ratio: float,
    sl_ema_pct: float,
    use_fast_ema: bool = False,
    rsi_filter: bool = False,
    atr_sl_mult: float = 0.0,
    m15_trend_filter: bool = False,
    adx_min: float = 0.0,
    entry_confirm_bar: bool = False,
    initial_balance: float = 1000.0,
    fee_rate: float = 0.0004,
    max_daily_loss_pct: float = 0.10,
    min_balance_pct: float = 0.20,
    max_trade_bars: int = 288,
    lock_loss_pct: float = 0.50,
    depletion_parts: int = 4,
    depletion_interval_bars: int = 12,
) -> Dict:
    balance = initial_balance
    equity_peak = balance
    max_dd = 0.0
    min_balance_seen = balance
    daily_loss: float = 0.0
    daily_loss_date: str = ""
    trading_halted: bool = False

    pos_side: str = ""
    pos_entry: float = 0.0
    pos_sl: float = 0.0
    pos_tp: float = 0.0
    pos_qty: float = 0.0
    pos_orig_qty: float = 0.0
    pos_risk: float = 0.0
    pos_rev_count: int = 0
    pos_entry_bar: int = 0
    pos_locked: bool = False
    pos_depletion_done: int = 0
    pos_depletion_last_bar: int = 0
    in_position: bool = False

    wins = 0
    total_trades = 0

    min_balance_threshold = initial_balance * min_balance_pct
    max_daily_loss_amount = initial_balance * max_daily_loss_pct
    n = len(arr_close)
    start_i = max(slope_bars + 2, 6)

    for i in range(start_i, n):
        if balance <= 0 or trading_halted:
            break

        price = arr_close[i]

        if not in_position:
            trend_up   = price > arr_ema200_m15[i] and price > arr_ema200_h1[i]
            trend_down = price < arr_ema200_m15[i] and price < arr_ema200_h1[i]

            slope_ref  = arr_ema200[i - slope_bars]
            slope_up   = arr_ema200[i] > slope_ref
            slope_down = arr_ema200[i] < slope_ref

            if use_fast_ema:
                fast_a, fast_b = arr_ema8, arr_ema13
            else:
                fast_a, fast_b = arr_ema13, arr_ema21

            golden = fast_a[i - 1] <= fast_b[i - 1] and fast_a[i] > fast_b[i]
            death  = fast_a[i - 1] >= fast_b[i - 1] and fast_a[i] < fast_b[i]

            long_signal  = trend_up   and slope_up   and golden and price > arr_ema200[i]
            short_signal = trend_down and slope_down and death  and price < arr_ema200[i]

            if m15_trend_filter:
                m15_bull = arr_ema13_m15[i] > arr_ema21_m15[i]
                m15_bear = arr_ema13_m15[i] < arr_ema21_m15[i]
                if long_signal  and not m15_bull:
                    long_signal = False
                if short_signal and not m15_bear:
                    short_signal = False

            if adx_min > 0:
                if arr_adx14[i] < adx_min:
                    long_signal = False
                    short_signal = False

            if entry_confirm_bar and i >= 2:
                if use_fast_ema:
                    conf_a_pp, conf_b_pp = arr_ema8[i - 2],  arr_ema13[i - 2]
                    conf_a_p,  conf_b_p  = arr_ema8[i - 1],  arr_ema13[i - 1]
                    conf_a_c,  conf_b_c  = arr_ema8[i],      arr_ema13[i]
                else:
                    conf_a_pp, conf_b_pp = arr_ema13[i - 2], arr_ema21[i - 2]
                    conf_a_p,  conf_b_p  = arr_ema13[i - 1], arr_ema21[i - 1]
                    conf_a_c,  conf_b_c  = arr_ema13[i],     arr_ema21[i]
                prev_golden  = conf_a_pp <= conf_b_pp and conf_a_p > conf_b_p
                prev_death   = conf_a_pp >= conf_b_pp and conf_a_p < conf_b_p
                still_golden = conf_a_p > conf_b_p and conf_a_c > conf_b_c
                still_death  = conf_a_p < conf_b_p and conf_a_c < conf_b_c
                if long_signal  and not (prev_golden and still_golden):
                    long_signal = False
                if short_signal and not (prev_death  and still_death):
                    short_signal = False

            if rsi_filter:
                h1_rsi = arr_rsi14_h1[i]
                if long_signal  and h1_rsi > 65:
                    long_signal = False
                if short_signal and h1_rsi < 35:
                    short_signal = False

            if not long_signal and not short_signal:
                continue

            today = arr_dates[i]
            if daily_loss_date != today:
                daily_loss_date = today
                daily_loss = 0.0

            if balance < min_balance_threshold:
                trading_halted = True
                break

            if daily_loss >= max_daily_loss_amount:
                continue

            side = "LONG" if long_signal else "SHORT"
            sl_ema_mult = 1.0 - sl_ema_pct if side == "LONG" else 1.0 + sl_ema_pct

            if side == "LONG":
                i_start = max(i - 2, 0)
                sl_last3 = float(arr_low[i_start: i + 1].min())
                sl_ema_val = arr_ema21[i] * sl_ema_mult
                sl_ema_based = max(sl_last3, sl_ema_val)
                if atr_sl_mult > 0:
                    sl_atr = price - arr_atr14[i] * atr_sl_mult
                    sl = max(sl_ema_based, sl_atr)
                else:
                    sl = sl_ema_based
                risk = price - sl
                if risk <= 0:
                    continue
                tp = price + risk * rr_ratio
            else:
                i_start = max(i - 2, 0)
                sl_last3 = float(arr_high[i_start: i + 1].max())
                sl_ema_val = arr_ema21[i] * sl_ema_mult
                sl_ema_based = min(sl_last3, sl_ema_val)
                if atr_sl_mult > 0:
                    sl_atr = price + arr_atr14[i] * atr_sl_mult
                    sl = min(sl_ema_based, sl_atr)
                else:
                    sl = sl_ema_based
                risk = sl - price
                if risk <= 0:
                    continue
                tp = price - risk * rr_ratio

            sl_distance = abs(price - sl)
            risk_amount = balance * risk_per_trade_pct
            qty = (risk_amount * leverage) / sl_distance
            max_notional = balance * 0.75 * leverage
            qty = min(qty, max_notional / price)
            if qty <= 0:
                continue

            notional = qty * price
            entry_fee = notional * fee_rate
            balance -= entry_fee
            if balance <= 0:
                break

            pos_side = side
            pos_entry = price
            pos_sl = sl
            pos_tp = tp
            pos_qty = qty
            pos_orig_qty = qty
            pos_risk = abs(price - sl)
            pos_rev_count = 0
            pos_entry_bar = i
            pos_locked = False
            pos_depletion_done = 0
            pos_depletion_last_bar = i
            in_position = True
            continue

        # --- Kilitli pozisyon: parça parça eritme ---
        if pos_locked:
            if pos_depletion_done < depletion_parts:
                bars_since = i - pos_depletion_last_bar
                if bars_since >= depletion_interval_bars:
                    part_qty = pos_orig_qty / depletion_parts
                    part_qty = min(part_qty, pos_qty)
                    part_pnl = (
                        (price - pos_entry) * part_qty
                        if pos_side == "LONG"
                        else (pos_entry - price) * part_qty
                    )
                    part_fee = price * part_qty * fee_rate
                    part_net = part_pnl - part_fee
                    balance += part_net
                    min_balance_seen = min(min_balance_seen, balance)
                    if part_net < 0:
                        daily_loss += abs(part_net)
                    pos_qty -= part_qty
                    pos_depletion_done += 1
                    pos_depletion_last_bar = i
                    total_trades += 1
                    if part_net > 0:
                        wins += 1
                    if pos_depletion_done >= depletion_parts or pos_qty <= 0:
                        in_position = False
                        equity_peak = max(equity_peak, balance)
                        dd = (equity_peak - balance) / equity_peak if equity_peak > 0 else 0.0
                        max_dd = max(max_dd, dd)
                    if balance <= 0:
                        break
            else:
                in_position = False
            continue

        # --- Kilit tetikleme ---
        unrealized = (
            (price - pos_entry) * pos_qty
            if pos_side == "LONG"
            else (pos_entry - price) * pos_qty
        )
        lock_threshold = -pos_risk * pos_qty * lock_loss_pct
        if unrealized <= lock_threshold:
            pos_locked = True
            pos_depletion_last_bar = i
            continue

        # --- 24 saat (max_trade_bars bar) zorla kapama ---
        if i - pos_entry_bar >= max_trade_bars:
            pnl = (price - pos_entry) * pos_qty if pos_side == "LONG" else (pos_entry - price) * pos_qty
            exit_fee = price * pos_qty * fee_rate
            net_pnl = pnl - exit_fee
            balance += net_pnl
            min_balance_seen = min(min_balance_seen, balance)
            if net_pnl < 0:
                daily_loss += abs(net_pnl)
            total_trades += 1
            if net_pnl > 0:
                wins += 1
            in_position = False
            equity_peak = max(equity_peak, balance)
            dd = (equity_peak - balance) / equity_peak if equity_peak > 0 else 0.0
            max_dd = max(max_dd, dd)
            if balance <= 0:
                break
            continue

        # --- Normal çıkış: SL / TP / ters kesişim ---
        hit_sl = price <= pos_sl if pos_side == "LONG" else price >= pos_sl
        hit_tp = price >= pos_tp if pos_side == "LONG" else price <= pos_tp

        if pos_side == "LONG":
            raw_reverse = arr_ema13[i - 1] >= arr_ema21[i - 1] and arr_ema13[i] < arr_ema21[i]
        else:
            raw_reverse = arr_ema13[i - 1] <= arr_ema21[i - 1] and arr_ema13[i] > arr_ema21[i]

        pos_rev_count = pos_rev_count + 1 if raw_reverse else 0

        price_diff = (price - pos_entry) if pos_side == "LONG" else (pos_entry - price)
        profit_ratio = price_diff / pos_risk if pos_risk > 0 else 0.0
        confirmed_reverse = pos_rev_count >= reverse_confirm_bars
        enough_profit = profit_ratio >= rr_exit_ratio

        if not (hit_sl or hit_tp or (confirmed_reverse and enough_profit)):
            continue

        pnl = (price - pos_entry) * pos_qty if pos_side == "LONG" else (pos_entry - price) * pos_qty
        exit_fee = price * pos_qty * fee_rate
        net_pnl = pnl - exit_fee
        balance += net_pnl
        min_balance_seen = min(min_balance_seen, balance)

        if net_pnl < 0:
            daily_loss += abs(net_pnl)

        total_trades += 1
        if net_pnl > 0:
            wins += 1

        in_position = False

        equity_peak = max(equity_peak, balance)
        dd = (equity_peak - balance) / equity_peak if equity_peak > 0 else 0.0
        max_dd = max(max_dd, dd)

        if balance <= 0:
            break

    win_rate = wins / total_trades * 100 if total_trades > 0 else 0.0
    total_return = (balance - initial_balance) / initial_balance * 100

    return {
        "trades": total_trades,
        "win_rate": win_rate,
        "total_return": total_return,
        "final_balance": balance,
        "max_dd": max_dd * 100,
        "min_balance": min_balance_seen,
    }


def _worker(combo, keys, arrays):
    (arr_close, arr_high, arr_low,
     arr_ema8, arr_ema13, arr_ema21, arr_ema200,
     arr_ema200_m15, arr_ema13_m15, arr_ema21_m15,
     arr_ema200_h1, arr_rsi14_h1, arr_atr14, arr_adx14, arr_dates) = arrays
    params = dict(zip(keys, combo))
    res = run_single(
        arr_close, arr_high, arr_low,
        arr_ema8, arr_ema13, arr_ema21, arr_ema200,
        arr_ema200_m15, arr_ema13_m15, arr_ema21_m15,
        arr_ema200_h1, arr_rsi14_h1, arr_atr14, arr_adx14, arr_dates,
        **params,
    )
    res.update(params)
    return res


def main():
    api_key    = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")
    symbol     = os.getenv("SYMBOL", "BTCUSDT")
    market_type = os.getenv("MARKET_TYPE", "futures").lower()
    start_time = os.getenv("BT_START", "2024-01-01")
    end_time   = os.getenv("BT_END", "2025-04-27")

    client = Client(api_key, api_secret)
    df = fetch_merged_frame(client, symbol, start_time, end_time, market_type)

    param_grid = {
        "leverage":              [3, 5, 10, 15, 20, 25, 30, 35, 40],
        "risk_per_trade_pct":    [0.003, 0.005, 0.01],
        "rr_ratio":              [1.5, 2.0, 3.0],
        "slope_bars":            [3, 5],
        "reverse_confirm_bars":  [2, 3],
        "rr_exit_ratio":         [0.3, 0.5],
        "sl_ema_pct":            [0.003, 0.005],
        "use_fast_ema":          [False, True],
        "rsi_filter":            [False, True],
        "atr_sl_mult":           [0.0, 1.5],
        "m15_trend_filter":      [False, True],
        "adx_min":               [0.0, 20.0, 25.0],
        "entry_confirm_bar":     [False, True],
    }

    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    total = len(combos)
    print(f"\nToplam kombinasyon: {total}")
    print("Optimizasyon başlıyor...\n")

    arr_close      = df["close"].to_numpy(dtype=float)
    arr_high       = df["high"].to_numpy(dtype=float)
    arr_low        = df["low"].to_numpy(dtype=float)
    arr_ema8        = df["ema8"].to_numpy(dtype=float)
    arr_ema13       = df["ema13"].to_numpy(dtype=float)
    arr_ema21       = df["ema21"].to_numpy(dtype=float)
    arr_ema200      = df["ema200"].to_numpy(dtype=float)
    arr_ema200_m15  = df["ema200_m15"].to_numpy(dtype=float)
    arr_ema13_m15   = df["ema13_m15"].to_numpy(dtype=float)
    arr_ema21_m15   = df["ema21_m15"].to_numpy(dtype=float)
    arr_ema200_h1   = df["ema200_h1"].to_numpy(dtype=float)
    arr_rsi14_h1    = df["rsi14_h1"].to_numpy(dtype=float)
    arr_atr14       = df["atr14"].to_numpy(dtype=float)
    arr_adx14       = df["adx14"].to_numpy(dtype=float)
    arr_dates       = [t.strftime("%Y-%m-%d") for t in df["close_time"]]

    arrays = (
        arr_close, arr_high, arr_low,
        arr_ema8, arr_ema13, arr_ema21, arr_ema200,
        arr_ema200_m15, arr_ema13_m15, arr_ema21_m15,
        arr_ema200_h1, arr_rsi14_h1, arr_atr14, arr_adx14, arr_dates,
    )

    worker = partial(_worker, keys=keys, arrays=arrays)

    cpu_count = max(1, multiprocessing.cpu_count() - 1)
    print(f"  CPU sayısı: {cpu_count} çekirdek kullanılıyor")

    results = []
    chunk = max(1, total // 100)
    with multiprocessing.Pool(processes=cpu_count) as pool:
        for idx, res in enumerate(pool.imap_unordered(worker, combos, chunksize=chunk), 1):
            results.append(res)
            if idx % 2000 == 0 or idx == total:
                print(f"  [{idx}/{total}] tamamlandı...", flush=True)

    results_df = pd.DataFrame(results)

    results_df = results_df[results_df["trades"] >= 30]

    # Risk-adjusted score: total_return / (max_dd + 1) — yüksek kazanç, düşük drawdown
    results_df["score"] = results_df["total_return"] / (results_df["max_dd"] + 1.0)

    results_df = results_df.sort_values(
        ["score", "total_return"],
        ascending=[False, False]
    ).reset_index(drop=True)

    display_cols = ["score", "win_rate", "total_return", "trades", "max_dd", "final_balance",
                    "leverage", "risk_per_trade_pct", "rr_ratio", "slope_bars",
                    "reverse_confirm_bars", "rr_exit_ratio", "sl_ema_pct",
                    "use_fast_ema", "rsi_filter", "m15_trend_filter", "adx_min", "entry_confirm_bar"]

    print("\n=== TOP 15 SONUÇ (risk-adjusted score, min 30 işlem) ===")
    print(results_df[display_cols].head(15).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    best = results_df.iloc[0]
    print("\n=== EN İYİ PARAMETRE SETİ ===")
    for col in display_cols:
        print(f"  {col:30s}: {best[col]:.4f}" if isinstance(best[col], float) else f"  {col:30s}: {best[col]}")

    results_df.to_csv("optimization_results.csv", index=False)
    print("\nTüm sonuçlar: optimization_results.csv")


if __name__ == "__main__":
    main()
