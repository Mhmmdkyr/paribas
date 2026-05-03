import argparse
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from binance.client import Client
from dotenv import load_dotenv


load_dotenv()


@dataclass
class BacktestConfig:
    symbol: str = os.getenv("SYMBOL", "BTCUSDT")
    market_type: str = os.getenv("MARKET_TYPE", "futures").lower()  # futures | spot
    balance_usage_pct: float = float(os.getenv("BALANCE_USAGE_PCT", "0.75"))
    leverage: str = os.getenv("LEVERAGE", "max")
    max_leverage_cap: int = int(os.getenv("MAX_LEVERAGE_CAP", "10"))
    risk_per_trade_pct: float = float(os.getenv("RISK_PER_TRADE_PCT", "0.01"))
    max_daily_loss_pct: float = float(os.getenv("MAX_DAILY_LOSS_PCT", "0.05"))
    min_balance_pct: float = float(os.getenv("MIN_BALANCE_PCT", "0.20"))
    reverse_cross_confirm_bars: int = int(os.getenv("REVERSE_CROSS_CONFIRM_BARS", "2"))
    reverse_cross_min_profit_ratio: float = float(os.getenv("REVERSE_CROSS_MIN_PROFIT_RATIO", "0.3"))
    slope_bars: int = int(os.getenv("SLOPE_BARS", "3"))
    rr_ratio: float = float(os.getenv("RR_RATIO", "1.5"))
    sl_ema_pct: float = float(os.getenv("SL_EMA_PCT", "0.005"))
    use_fast_ema: bool = os.getenv("USE_FAST_EMA", "true").lower() == "true"
    rsi_filter: bool = os.getenv("RSI_FILTER", "true").lower() == "true"
    m15_trend_filter: bool = os.getenv("M15_TREND_FILTER", "true").lower() == "true"
    adx_min: float = float(os.getenv("ADX_MIN", "25"))
    entry_confirm_bar: bool = os.getenv("ENTRY_CONFIRM_BAR", "false").lower() == "true"
    max_trade_hours: int = int(os.getenv("MAX_TRADE_HOURS", "24"))
    lock_loss_pct: float = float(os.getenv("LOCK_LOSS_PCT", "0.50"))
    depletion_parts: int = int(os.getenv("DEPLETION_PARTS", "4"))
    depletion_interval_bars: int = int(os.getenv("DEPLETION_INTERVAL_BARS", "12"))
    initial_balance: float = float(os.getenv("BT_INITIAL_BALANCE", "1000"))
    fee_rate: float = float(os.getenv("BT_FEE_RATE", "0.0004"))
    start_time: str = os.getenv("BT_START", "2025-01-01")
    end_time: str = os.getenv("BT_END", "2025-12-31")
    testnet: bool = os.getenv("TESTNET", "false").lower() == "true"


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema8"]   = out["close"].ewm(span=8,   adjust=False).mean()
    out["ema13"]  = out["close"].ewm(span=13,  adjust=False).mean()
    out["ema21"]  = out["close"].ewm(span=21,  adjust=False).mean()
    out["ema200"] = out["close"].ewm(span=200, adjust=False).mean()
    delta    = out["close"].diff()
    avg_gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    avg_loss = (-delta).clip(lower=0).ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    out["rsi14"] = 100 - 100 / (1 + rs)
    hl   = out["high"] - out["low"]
    hc   = (out["high"] - out["close"].shift()).abs()
    lc   = (out["low"]  - out["close"].shift()).abs()
    tr   = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    out["atr14"] = tr.ewm(com=13, adjust=False).mean()
    up_move   = out["high"].diff()
    down_move = (-out["low"].diff())
    plus_dm   = np.where((up_move > down_move) & (up_move > 0),   up_move,   0.0)
    minus_dm  = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr_s     = pd.Series(tr.values, index=out.index)
    plus_di   = 100 * pd.Series(plus_dm,  index=out.index).ewm(com=13, adjust=False).mean() / atr_s.ewm(com=13, adjust=False).mean()
    minus_di  = 100 * pd.Series(minus_dm, index=out.index).ewm(com=13, adjust=False).mean() / atr_s.ewm(com=13, adjust=False).mean()
    dx        = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float("nan"))
    out["adx14"] = dx.ewm(com=13, adjust=False).mean()
    return out


def fetch_klines_df(
    client: Client,
    symbol: str,
    interval: str,
    start_time: str,
    end_time: str,
    market_type: str,
) -> pd.DataFrame:
    if market_type == "futures":
        raw = client.futures_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str=start_time,
            end_str=end_time,
        )
    else:
        raw = client.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str=start_time,
            end_str=end_time,
        )

    df = pd.DataFrame(
        raw,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ],
    )
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    return df[["open_time", "high", "low", "close", "close_time"]]


def get_max_futures_leverage(client: Client, symbol: str) -> int:
    try:
        bracket_data = client.futures_leverage_bracket(symbol=symbol)
        if not bracket_data:
            return 1
        symbol_brackets = bracket_data[0].get("brackets", [])
        max_lev = max(int(b.get("initialLeverage", 1)) for b in symbol_brackets)
        return max_lev if max_lev > 0 else 1
    except Exception as exc:
        logging.warning("Maksimum kaldıraç alınamadı, varsayılan 20x kullanılacak: %s", exc)
        return 20


def resolve_leverage(client: Client, cfg: BacktestConfig) -> int:
    if cfg.market_type != "futures":
        return 1

    max_lev = get_max_futures_leverage(client, cfg.symbol)
    cap = max(1, cfg.max_leverage_cap)
    lev_raw = str(cfg.leverage).strip().lower()

    if lev_raw == "max":
        resolved = min(max_lev, cap)
        logging.info("Kaldıraç: exchange_max=%s, cap=%s, kullanılan=%s", max_lev, cap, resolved)
        return resolved

    try:
        requested = int(lev_raw)
    except ValueError:
        logging.warning("LEVERAGE geçersiz (%s), cap uygulanarak max kullanılacak.", cfg.leverage)
        return min(max_lev, cap)

    if requested < 1:
        return 1
    return min(requested, max_lev, cap)


def prepare_merged_frame(client: Client, cfg: BacktestConfig) -> pd.DataFrame:
    m5  = fetch_klines_df(client, cfg.symbol, Client.KLINE_INTERVAL_5MINUTE,  cfg.start_time, cfg.end_time, cfg.market_type)
    m15 = fetch_klines_df(client, cfg.symbol, Client.KLINE_INTERVAL_15MINUTE, cfg.start_time, cfg.end_time, cfg.market_type)
    h1  = fetch_klines_df(client, cfg.symbol, Client.KLINE_INTERVAL_1HOUR,    cfg.start_time, cfg.end_time, cfg.market_type)

    m5  = add_indicators(m5).dropna().reset_index(drop=True)
    m15 = add_indicators(m15).dropna().reset_index(drop=True)
    h1  = add_indicators(h1).dropna().reset_index(drop=True)

    m15_view = m15[["close_time", "ema200", "ema13", "ema21"]].rename(
        columns={"ema200": "ema200_m15", "ema13": "ema13_m15", "ema21": "ema21_m15"}
    )
    h1_view = h1[["close_time", "ema200", "rsi14"]].rename(
        columns={"ema200": "ema200_h1", "rsi14": "rsi14_h1"}
    )

    merged = pd.merge_asof(
        m5.sort_values("close_time"),
        m15_view.sort_values("close_time"),
        on="close_time",
        direction="backward",
    )
    merged = pd.merge_asof(
        merged.sort_values("close_time"),
        h1_view.sort_values("close_time"),
        on="close_time",
        direction="backward",
    )

    return merged.dropna().reset_index(drop=True)


def run_backtest(cfg: BacktestConfig):
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")
    client = Client(api_key, api_secret, testnet=cfg.testnet)

    leverage = resolve_leverage(client, cfg)
    usage_pct = max(0.0, min(cfg.balance_usage_pct, 0.75))

    df = prepare_merged_frame(client, cfg)
    if len(df) < 210:
        raise ValueError("Backtest için yeterli mum verisi yok.")

    balance = cfg.initial_balance
    equity_peak = balance
    max_dd = 0.0
    min_balance = balance

    daily_loss: float = 0.0
    daily_loss_date: str = ""
    trading_halted: bool = False

    position: Optional[Dict] = None
    trades: List[Dict] = []
    depletion_events: List[Dict] = []

    for i in range(6, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]

        if balance <= 0:
            depletion_events.append(
                {
                    "time": curr["close_time"],
                    "stage": "LOOP_START",
                    "reason": "Bakiye sıfır veya negatif. Backtest durduruldu.",
                    "balance": balance,
                }
            )
            break

        price = float(curr["close"])
        trend_up   = price > float(curr["ema200_m15"]) and price > float(curr["ema200_h1"])
        trend_down = price < float(curr["ema200_m15"]) and price < float(curr["ema200_h1"])

        slope_ref  = float(df.iloc[i - cfg.slope_bars]["ema200"])
        slope_up   = float(curr["ema200"]) > slope_ref
        slope_down = float(curr["ema200"]) < slope_ref

        if cfg.use_fast_ema:
            fast_prev_a, fast_prev_b = float(prev["ema8"]),  float(prev["ema13"])
            fast_curr_a, fast_curr_b = float(curr["ema8"]),  float(curr["ema13"])
        else:
            fast_prev_a, fast_prev_b = float(prev["ema13"]), float(prev["ema21"])
            fast_curr_a, fast_curr_b = float(curr["ema13"]), float(curr["ema21"])

        golden = fast_prev_a <= fast_prev_b and fast_curr_a > fast_curr_b
        death  = fast_prev_a >= fast_prev_b and fast_curr_a < fast_curr_b

        if cfg.entry_confirm_bar and i >= 2:
            pprev = df.iloc[i - 2]
            if cfg.use_fast_ema:
                pp_a, pp_b = float(pprev["ema8"]),  float(pprev["ema13"])
            else:
                pp_a, pp_b = float(pprev["ema13"]), float(pprev["ema21"])
            # cross on bar i-2->i-1, bar i still in that direction
            prev_golden  = pp_a  <= pp_b  and fast_prev_a > fast_prev_b
            prev_death   = pp_a  >= pp_b  and fast_prev_a < fast_prev_b
            still_golden = fast_prev_a > fast_prev_b and fast_curr_a > fast_curr_b
            still_death  = fast_prev_a < fast_prev_b and fast_curr_a < fast_curr_b
            long_signal  = trend_up   and slope_up   and prev_golden  and still_golden and price > float(curr["ema200"])
            short_signal = trend_down and slope_down and prev_death   and still_death  and price < float(curr["ema200"])
        else:
            long_signal  = trend_up   and slope_up   and golden and price > float(curr["ema200"])
            short_signal = trend_down and slope_down and death  and price < float(curr["ema200"])

        if cfg.m15_trend_filter:
            m15_bull = float(curr["ema13_m15"]) > float(curr["ema21_m15"])
            m15_bear = float(curr["ema13_m15"]) < float(curr["ema21_m15"])
            if long_signal  and not m15_bull:
                long_signal = False
            if short_signal and not m15_bear:
                short_signal = False

        if cfg.adx_min > 0 and float(curr["adx14"]) < cfg.adx_min:
            long_signal  = False
            short_signal = False

        if cfg.rsi_filter:
            h1_rsi = float(curr["rsi14_h1"])
            if long_signal  and h1_rsi > 65:
                long_signal = False
            if short_signal and h1_rsi < 35:
                short_signal = False

        if position is None:
            if short_signal and cfg.market_type == "spot":
                continue
            if not long_signal and not short_signal:
                continue

            if trading_halted:
                continue

            today = curr["close_time"].strftime("%Y-%m-%d")
            if daily_loss_date != today:
                daily_loss_date = today
                daily_loss = 0.0

            min_balance_threshold = cfg.initial_balance * cfg.min_balance_pct
            if balance < min_balance_threshold:
                if not trading_halted:
                    logging.warning(
                        "MİNİMUM BAKİYE KORUMASI: balance=%.4f < threshold=%.4f. Backtest durduruldu.",
                        balance, min_balance_threshold,
                    )
                    trading_halted = True
                continue

            max_daily_loss_amount = cfg.initial_balance * cfg.max_daily_loss_pct
            if daily_loss >= max_daily_loss_amount:
                logging.warning(
                    "GÜNLÜK ZARAR LİMİTİ: daily_loss=%.4f >= limit=%.4f. %s tarihinde işlem yok.",
                    daily_loss, max_daily_loss_amount, today,
                )
                continue

            side = "LONG" if long_signal else "SHORT"
            sl_ema_mult = 1.0 - cfg.sl_ema_pct if side == "LONG" else 1.0 + cfg.sl_ema_pct
            if side == "LONG":
                sl_last3 = float(df.iloc[max(i - 2, 0) : i + 1]["low"].min())
                sl_ema   = float(curr["ema21"]) * sl_ema_mult
                sl = max(sl_last3, sl_ema)
                risk = price - sl
                if risk <= 0:
                    continue
                tp = price + (risk * cfg.rr_ratio)
            else:
                sl_last3 = float(df.iloc[max(i - 2, 0) : i + 1]["high"].max())
                sl_ema   = float(curr["ema21"]) * sl_ema_mult
                sl = min(sl_last3, sl_ema)
                risk = sl - price
                if risk <= 0:
                    continue
                tp = price - (risk * cfg.rr_ratio)

            sl_distance = abs(price - sl)
            risk_amount = balance * cfg.risk_per_trade_pct
            qty = (risk_amount * leverage) / sl_distance
            max_notional = balance * min(usage_pct, 0.75) * leverage if cfg.market_type == "futures" else balance * min(usage_pct, 0.75)
            qty = min(qty, max_notional / price)
            notional = qty * price
            if qty <= 0:
                depletion_events.append(
                    {
                        "time": curr["close_time"],
                        "stage": "ENTRY_REJECTED",
                        "reason": "Hesaplanan miktar 0 veya negatif.",
                        "balance": balance,
                        "price": price,
                    }
                )
                continue

            entry_fee = notional * cfg.fee_rate
            balance -= entry_fee
            min_balance = min(min_balance, balance)

            if balance <= 0:
                depletion_events.append(
                    {
                        "time": curr["close_time"],
                        "stage": "ENTRY_FEE",
                        "reason": "Giriş komisyonu sonrası bakiye tükendi.",
                        "balance": balance,
                        "entry_fee": entry_fee,
                        "side": side,
                    }
                )
                break

            position = {
                "side": side,
                "entry": price,
                "sl": sl,
                "tp": tp,
                "qty": qty,
                "original_qty": qty,
                "notional": notional,
                "entry_time": curr["close_time"],
                "entry_fee": entry_fee,
                "risk": abs(price - sl),
                "reverse_cross_count": 0,
                "locked": False,
                "depletion_done": 0,
                "depletion_last_bar": i,
            }
            continue

        side = position["side"]

        # --- Kilitli pozisyon: parça parça eritme ---
        if position["locked"]:
            parts_done = position["depletion_done"]
            total_parts = cfg.depletion_parts
            if parts_done < total_parts:
                bars_since = i - position["depletion_last_bar"]
                if bars_since >= cfg.depletion_interval_bars:
                    part_qty = position["original_qty"] / total_parts
                    part_notional = part_qty * price
                    part_exit_fee = part_notional * cfg.fee_rate
                    part_pnl = (
                        (price - position["entry"]) * part_qty
                        if side == "LONG"
                        else (position["entry"] - price) * part_qty
                    )
                    part_net = part_pnl - part_exit_fee
                    balance += part_net
                    min_balance = min(min_balance, balance)
                    if part_net < 0:
                        daily_loss += abs(part_net)
                    position["depletion_done"] += 1
                    position["depletion_last_bar"] = i
                    position["qty"] -= part_qty
                    trades.append({
                        "entry_time": position["entry_time"],
                        "exit_time": curr["close_time"],
                        "side": side,
                        "entry": position["entry"],
                        "exit": price,
                        "reason": f"ERIME_{parts_done + 1}/{total_parts}",
                        "gross_pnl": part_pnl,
                        "net_pnl": part_net,
                        "entry_fee": 0.0,
                        "exit_fee": part_exit_fee,
                        "balance_after": balance,
                    })
                    if position["depletion_done"] >= total_parts:
                        position = None
                        equity_peak = max(equity_peak, balance)
                        dd = (equity_peak - balance) / equity_peak if equity_peak > 0 else 0
                        max_dd = max(max_dd, dd)
            else:
                position = None
            continue

        # --- Kilit tetikleme: zarar eşiği aşıldı mı? ---
        unrealized_pnl = (
            (price - position["entry"]) * position["qty"]
            if side == "LONG"
            else (position["entry"] - price) * position["qty"]
        )
        lock_threshold = -position["risk"] * position["qty"] * cfg.lock_loss_pct
        if unrealized_pnl <= lock_threshold:
            logging.warning(
                "POZİSYON KİLİTLENDİ: unrealized=%.2f <= threshold=%.2f | %s | entry=%.2f",
                unrealized_pnl, lock_threshold, side, position["entry"],
            )
            position["locked"] = True
            position["depletion_last_bar"] = i
            continue

        # --- 24 saat zorla kapatma ---
        hours_open = (curr["close_time"] - position["entry_time"]).total_seconds() / 3600
        if hours_open >= cfg.max_trade_hours:
            reason = "ZAMAN_ASIMI"
            pnl = (
                (price - position["entry"]) * position["qty"]
                if side == "LONG"
                else (position["entry"] - price) * position["qty"]
            )
            exit_notional = price * position["qty"]
            exit_fee = exit_notional * cfg.fee_rate
            net_pnl = pnl - exit_fee
            balance += net_pnl
            min_balance = min(min_balance, balance)
            if net_pnl < 0:
                daily_loss += abs(net_pnl)
            trades.append({
                "entry_time": position["entry_time"],
                "exit_time": curr["close_time"],
                "side": side,
                "entry": position["entry"],
                "exit": price,
                "reason": reason,
                "gross_pnl": pnl,
                "net_pnl": net_pnl,
                "entry_fee": position["entry_fee"],
                "exit_fee": exit_fee,
                "balance_after": balance,
            })
            position = None
            equity_peak = max(equity_peak, balance)
            dd = (equity_peak - balance) / equity_peak if equity_peak > 0 else 0
            max_dd = max(max_dd, dd)
            continue

        # --- Normal çıkış: SL / TP / ters kesişim ---
        hit_sl = price <= position["sl"] if side == "LONG" else price >= position["sl"]
        hit_tp = price >= position["tp"] if side == "LONG" else price <= position["tp"]
        if cfg.use_fast_ema:
            rev_prev_a, rev_prev_b = float(prev["ema8"]),  float(prev["ema13"])
            rev_curr_a, rev_curr_b = float(curr["ema8"]),  float(curr["ema13"])
        else:
            rev_prev_a, rev_prev_b = float(prev["ema13"]), float(prev["ema21"])
            rev_curr_a, rev_curr_b = float(curr["ema13"]), float(curr["ema21"])
        raw_reverse = (
            (rev_prev_a >= rev_prev_b and rev_curr_a < rev_curr_b)
            if side == "LONG"
            else (rev_prev_a <= rev_prev_b and rev_curr_a > rev_curr_b)
        )

        if raw_reverse:
            position["reverse_cross_count"] += 1
        else:
            position["reverse_cross_count"] = 0

        price_diff = (price - position["entry"]) if side == "LONG" else (position["entry"] - price)
        profit_in_risk_units = price_diff / position["risk"] if position["risk"] > 0 else 0
        confirmed_reverse = position["reverse_cross_count"] >= cfg.reverse_cross_confirm_bars
        enough_profit = profit_in_risk_units >= cfg.reverse_cross_min_profit_ratio

        if not (hit_sl or hit_tp or (confirmed_reverse and enough_profit)):
            continue

        reason = "SL" if hit_sl else ("TP" if hit_tp else "TERS_KESISIM")
        pnl = (price - position["entry"]) * position["qty"] if side == "LONG" else (position["entry"] - price) * position["qty"]
        exit_notional = price * position["qty"]
        exit_fee = exit_notional * cfg.fee_rate
        net_pnl = pnl - exit_fee
        balance += net_pnl
        min_balance = min(min_balance, balance)

        if net_pnl < 0:
            daily_loss += abs(net_pnl)

        trades.append(
            {
                "entry_time": position["entry_time"],
                "exit_time": curr["close_time"],
                "side": side,
                "entry": position["entry"],
                "exit": price,
                "reason": reason,
                "gross_pnl": pnl,
                "net_pnl": net_pnl,
                "entry_fee": position["entry_fee"],
                "exit_fee": exit_fee,
                "balance_after": balance,
            }
        )
        position = None

        equity_peak = max(equity_peak, balance)
        dd = (equity_peak - balance) / equity_peak if equity_peak > 0 else 0
        max_dd = max(max_dd, dd)

        if balance <= 0:
            depletion_events.append(
                {
                    "time": curr["close_time"],
                    "stage": "POSITION_CLOSE",
                    "reason": "Pozisyon kapanışı sonrası bakiye tükendi.",
                    "balance": balance,
                    "side": side,
                    "close_reason": reason,
                    "net_pnl": net_pnl,
                }
            )
            break

    total_return = ((balance - cfg.initial_balance) / cfg.initial_balance) * 100
    wins = sum(1 for t in trades if t["net_pnl"] > 0)
    win_rate = (wins / len(trades) * 100) if trades else 0.0
    best_trade_net = max(trades, key=lambda t: t["net_pnl"]) if trades else None
    worst_trade_net = min(trades, key=lambda t: t["net_pnl"]) if trades else None
    best_trade_gross = max(trades, key=lambda t: t["gross_pnl"]) if trades else None
    worst_trade_gross = min(trades, key=lambda t: t["gross_pnl"]) if trades else None

    print("\n=== BACKTEST SONUCU ===")
    print(f"Sembol            : {cfg.symbol}")
    print(f"Piyasa            : {cfg.market_type}")
    print(f"Dönem             : {cfg.start_time} -> {cfg.end_time}")
    print(f"Kaldıraç          : {leverage}x")
    print(f"Kullanım Oranı    : {usage_pct * 100:.2f}%")
    print(f"Başlangıç Bakiye  : {cfg.initial_balance:.2f}")
    print(f"Bitiş Bakiye      : {balance:.2f}")
    print(f"Toplam Getiri     : {total_return:.2f}%")
    print(f"İşlem Sayısı      : {len(trades)}")
    print(f"Win Rate          : {win_rate:.2f}%")
    print(f"Max Drawdown      : {max_dd * 100:.2f}%")
    print(f"Min Bakiye        : {min_balance:.2f}")

    if best_trade_net and worst_trade_net:
        print(
            "Maks Net Kar      : "
            f"{best_trade_net['net_pnl']:.2f} | {best_trade_net['side']} | "
            f"{best_trade_net['entry_time']} -> {best_trade_net['exit_time']} | {best_trade_net['reason']}"
        )
        print(
            "Maks Net Zarar    : "
            f"{worst_trade_net['net_pnl']:.2f} | {worst_trade_net['side']} | "
            f"{worst_trade_net['entry_time']} -> {worst_trade_net['exit_time']} | {worst_trade_net['reason']}"
        )
        print(
            "Maks Brüt Kar     : "
            f"{best_trade_gross['gross_pnl']:.2f} | {best_trade_gross['side']} | "
            f"{best_trade_gross['entry_time']} -> {best_trade_gross['exit_time']} | {best_trade_gross['reason']}"
        )
        print(
            "Maks Brüt Zarar   : "
            f"{worst_trade_gross['gross_pnl']:.2f} | {worst_trade_gross['side']} | "
            f"{worst_trade_gross['entry_time']} -> {worst_trade_gross['exit_time']} | {worst_trade_gross['reason']}"
        )

    if depletion_events:
        print("\n=== BAKIYE TUKENME OLAYLARI ===")
        for idx, event in enumerate(depletion_events, start=1):
            print(f"{idx}. {event}")

    if trades:
        trades_df = pd.DataFrame(trades)
        out_file = "backtest_trades.csv"
        trades_df.to_csv(out_file, index=False)
        print(f"İşlem detayları   : {out_file}")

    if depletion_events:
        depletion_df = pd.DataFrame(depletion_events)
        dep_file = "backtest_depletion_events.csv"
        depletion_df.to_csv(dep_file, index=False)
        print(f"Tükenme detayları : {dep_file}")


def parse_args() -> BacktestConfig:
    parser = argparse.ArgumentParser(description="EMA Crossover multi-timeframe strategy backtest")
    parser.add_argument("--symbol", default=os.getenv("SYMBOL", "BTCUSDT"))
    parser.add_argument("--market-type", default=os.getenv("MARKET_TYPE", "futures"), choices=["futures", "spot"])
    parser.add_argument("--start", default=os.getenv("BT_START", "2025-01-01"))
    parser.add_argument("--end", default=os.getenv("BT_END", "2025-12-31"))
    parser.add_argument("--initial-balance", type=float, default=float(os.getenv("BT_INITIAL_BALANCE", "1000")))
    parser.add_argument("--fee-rate", type=float, default=float(os.getenv("BT_FEE_RATE", "0.0004")))
    parser.add_argument("--balance-usage", type=float, default=float(os.getenv("BALANCE_USAGE_PCT", "0.75")))
    parser.add_argument("--leverage", default=os.getenv("LEVERAGE", "max"))
    parser.add_argument("--max-leverage-cap", type=int, default=int(os.getenv("MAX_LEVERAGE_CAP", "10")))
    parser.add_argument("--risk-per-trade", type=float, default=float(os.getenv("RISK_PER_TRADE_PCT", "0.01")))
    parser.add_argument("--max-daily-loss", type=float, default=float(os.getenv("MAX_DAILY_LOSS_PCT", "0.05")))
    parser.add_argument("--min-balance-pct", type=float, default=float(os.getenv("MIN_BALANCE_PCT", "0.20")))
    parser.add_argument("--reverse-confirm-bars", type=int, default=int(os.getenv("REVERSE_CROSS_CONFIRM_BARS", "2")))
    parser.add_argument("--reverse-min-profit", type=float, default=float(os.getenv("REVERSE_CROSS_MIN_PROFIT_RATIO", "0.3")))
    parser.add_argument("--testnet", action="store_true", default=os.getenv("TESTNET", "false").lower() == "true")
    args = parser.parse_args()

    return BacktestConfig(
        symbol=args.symbol,
        market_type=args.market_type,
        balance_usage_pct=args.balance_usage,
        leverage=args.leverage,
        max_leverage_cap=args.max_leverage_cap,
        risk_per_trade_pct=args.risk_per_trade,
        max_daily_loss_pct=args.max_daily_loss,
        min_balance_pct=args.min_balance_pct,
        reverse_cross_confirm_bars=args.reverse_confirm_bars,
        reverse_cross_min_profit_ratio=args.reverse_min_profit,
        initial_balance=args.initial_balance,
        fee_rate=args.fee_rate,
        start_time=args.start,
        end_time=args.end,
        testnet=args.testnet,
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = parse_args()
    run_backtest(cfg)


if __name__ == "__main__":
    main()
