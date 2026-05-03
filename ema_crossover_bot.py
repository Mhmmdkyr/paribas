import logging
import math
import os
import sys
import time
import traceback
import urllib.parse
import urllib.request
from dataclasses import dataclass
from threading import Lock
from typing import Dict, Optional

import numpy as np
import pandas as pd
from binance import ThreadedWebsocketManager
from binance.client import Client
from binance.enums import ORDER_TYPE_MARKET, SIDE_BUY, SIDE_SELL
from dotenv import load_dotenv


load_dotenv()


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str):
        self.token = token.strip()
        self.chat_id = chat_id.strip()
        self.enabled = bool(self.token and self.chat_id)
        if not self.enabled:
            logging.info("Telegram bildirimi devre dışı (TELEGRAM_TOKEN veya TELEGRAM_CHAT_ID eksik).")

    def send(self, text: str):
        if not self.enabled:
            return
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            data = urllib.parse.urlencode({"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"}).encode()
            req = urllib.request.Request(url, data=data, method="POST")
            with urllib.request.urlopen(req, timeout=5):
                pass
        except Exception as exc:
            logging.warning("Telegram gönderim hatası: %s", exc)


@dataclass
class BotConfig:
    symbol: str = os.getenv("SYMBOL", "BTCUSDT")
    market_type: str = os.getenv("MARKET_TYPE", "futures").lower()  # futures | spot
    balance_usage_pct: float = float(os.getenv("BALANCE_USAGE_PCT", "0.75"))
    rest_limit: int = int(os.getenv("REST_LIMIT", "2000"))
    dry_run: bool = os.getenv("DRY_RUN", "true").lower() == "true"
    testnet: bool = os.getenv("TESTNET", "false").lower() == "true"
    leverage: str = os.getenv("LEVERAGE", "max")
    max_leverage_cap: int = int(os.getenv("MAX_LEVERAGE_CAP", "20"))
    risk_per_trade_pct: float = float(os.getenv("RISK_PER_TRADE_PCT", "0.003"))  # Bakiyenin %0.3'ü
    max_daily_loss_pct: float = float(os.getenv("MAX_DAILY_LOSS_PCT", "0.10"))  # Günlük %10
    min_balance_pct: float = float(os.getenv("MIN_BALANCE_PCT", "0.20"))  # Başlangıç bakiyesinin %20'si
    reverse_cross_confirm_bars: int = int(os.getenv("REVERSE_CROSS_CONFIRM_BARS", "2"))  # Kaç ardarda mum teyit
    reverse_cross_min_profit_ratio: float = float(os.getenv("REVERSE_CROSS_MIN_PROFIT_RATIO", "0.3"))  # Riskin %30'u kazanılmadan çıkma
    slope_bars: int = int(os.getenv("SLOPE_BARS", "3"))
    rr_ratio: float = float(os.getenv("RR_RATIO", "1.5"))
    sl_ema_pct: float = float(os.getenv("SL_EMA_PCT", "0.003"))
    use_fast_ema: bool = os.getenv("USE_FAST_EMA", "true").lower() == "true"
    rsi_filter: bool = os.getenv("RSI_FILTER", "true").lower() == "true"  # optimal: true
    m15_trend_filter: bool = os.getenv("M15_TREND_FILTER", "true").lower() == "true"
    adx_min: float = float(os.getenv("ADX_MIN", "25"))
    entry_confirm_bar: bool = os.getenv("ENTRY_CONFIRM_BAR", "false").lower() == "true"
    max_trade_hours: int = int(os.getenv("MAX_TRADE_HOURS", "24"))
    lock_loss_pct: float = float(os.getenv("LOCK_LOSS_PCT", "0.50"))
    depletion_parts: int = int(os.getenv("DEPLETION_PARTS", "4"))
    depletion_interval_bars: int = int(os.getenv("DEPLETION_INTERVAL_BARS", "12"))
    tp1_ratio: float = float(os.getenv("TP1_RATIO", "0.75"))
    tp1_qty_pct: float = float(os.getenv("TP1_QTY_PCT", "0.50"))
    breakeven_after_tp1: bool = os.getenv("BREAKEVEN_AFTER_TP1", "true").lower() == "true"

    telegram_token: str = os.getenv("TELEGRAM_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")

    m5: str = Client.KLINE_INTERVAL_5MINUTE
    m15: str = Client.KLINE_INTERVAL_15MINUTE
    h1: str = Client.KLINE_INTERVAL_1HOUR


class MultiTimeframeEMABot:
    def __init__(self, config: BotConfig):
        self.config = config
        self.api_key = os.getenv("BINANCE_API_KEY", "")
        self.api_secret = os.getenv("BINANCE_API_SECRET", "")

        if not self.api_key or not self.api_secret:
            raise ValueError("BINANCE_API_KEY ve BINANCE_API_SECRET ortam değişkenleri zorunludur.")

        self.client = Client(self.api_key, self.api_secret, testnet=self.config.testnet)
        self.ws_manager: Optional[ThreadedWebsocketManager] = None
        self.tg = TelegramNotifier(self.config.telegram_token, self.config.telegram_chat_id)

        self.frames: Dict[str, pd.DataFrame] = {}
        self.position: Optional[Dict] = None
        self.lock = Lock()
        self.reconnect_requested = False
        self.quote_asset: str = "USDT"
        self.step_size: float = 0.0
        self.min_qty: float = 0.0
        self.active_leverage: int = 1
        self.initial_balance: float = 0.0
        self.daily_loss: float = 0.0
        self.daily_loss_date: Optional[str] = None
        self.trading_halted: bool = False

        self._signal_count_hour: int = 0
        self._signal_long_count: int = 0
        self._signal_short_count: int = 0
        self._last_report_hour: int = -1

        self._load_symbol_filters()

        if self.config.market_type == "futures":
            self._configure_futures()

    def _configure_futures(self):
        try:
            self.active_leverage = self._resolve_futures_leverage()
            self.client.futures_change_leverage(symbol=self.config.symbol, leverage=self.active_leverage)
            logging.info("Futures leverage ayarlandı: %sx", self.active_leverage)
        except Exception as exc:
            logging.warning("Leverage ayarlanamadı: %s", exc)

    def _get_max_futures_leverage(self) -> int:
        try:
            bracket_data = self.client.futures_leverage_bracket(symbol=self.config.symbol)
            if not bracket_data:
                return 1
            symbol_brackets = bracket_data[0].get("brackets", [])
            max_lev = max(int(b.get("initialLeverage", 1)) for b in symbol_brackets)
            return max_lev if max_lev > 0 else 1
        except Exception as exc:
            logging.warning("Maksimum kaldıraç alınamadı: %s", exc)
            return 1

    def _resolve_futures_leverage(self) -> int:
        lev_raw = str(self.config.leverage).strip().lower()
        max_lev = self._get_max_futures_leverage()
        cap = max(1, self.config.max_leverage_cap)

        if lev_raw == "max":
            resolved = min(max_lev, cap)
            logging.info("Kaldıraç: exchange_max=%s, cap=%s, kullanılan=%s", max_lev, cap, resolved)
            return resolved

        try:
            requested = int(lev_raw)
        except ValueError:
            logging.warning("LEVERAGE değeri geçersiz (%s), cap uygulanarak max kullanılacak.", self.config.leverage)
            return min(max_lev, cap)

        if requested < 1:
            return 1
        return min(requested, max_lev, cap)

    def _load_symbol_filters(self):
        try:
            if self.config.market_type == "futures":
                exchange_info = self.client.futures_exchange_info()
                symbols = exchange_info.get("symbols", [])
            else:
                exchange_info = self.client.get_exchange_info()
                symbols = exchange_info.get("symbols", [])

            symbol_info = next((s for s in symbols if s.get("symbol") == self.config.symbol), None)
            if not symbol_info:
                logging.warning("Sembol bilgisi bulunamadı: %s", self.config.symbol)
                return

            self.quote_asset = symbol_info.get("quoteAsset", "USDT")
            lot_filter = next((f for f in symbol_info.get("filters", []) if f.get("filterType") == "LOT_SIZE"), None)
            if lot_filter:
                self.step_size = float(lot_filter.get("stepSize", "0"))
                self.min_qty = float(lot_filter.get("minQty", "0"))

            logging.info(
                "Sembol filtreleri yüklendi | quote_asset=%s step_size=%s min_qty=%s",
                self.quote_asset,
                self.step_size,
                self.min_qty,
            )
        except Exception as exc:
            logging.warning("Sembol filtreleri yüklenemedi: %s", exc)

    def _get_available_quote_balance(self) -> float:
        try:
            if self.config.market_type == "futures":
                balances = self.client.futures_account_balance()
                for row in balances:
                    if row.get("asset") == self.quote_asset:
                        return float(row.get("availableBalance", 0.0))
                return 0.0

            balance = self.client.get_asset_balance(asset=self.quote_asset)
            if not balance:
                return 0.0
            return float(balance.get("free", 0.0))
        except Exception as exc:
            logging.error("Bakiye çekilemedi: %s", exc)
            return 0.0

    def _normalize_quantity(self, quantity: float) -> float:
        if quantity <= 0:
            return 0.0

        normalized = quantity
        if self.step_size > 0:
            normalized = math.floor(quantity / self.step_size) * self.step_size

        if self.min_qty > 0 and normalized < self.min_qty:
            return 0.0

        return float(normalized)

    def _calculate_order_quantity(self, entry_price: float, sl_price: float, side: str) -> float:
        available_quote = self._get_available_quote_balance()
        if available_quote <= 0:
            logging.warning("Kullanılabilir bakiye bulunamadı.")
            return 0.0

        risk_amount = available_quote * self.config.risk_per_trade_pct
        sl_distance = abs(entry_price - sl_price)
        if sl_distance <= 0:
            logging.warning("SL mesafesi sıfır, pozisyon açılmadı.")
            return 0.0

        raw_qty = (risk_amount * self.active_leverage) / sl_distance

        max_notional = available_quote * min(self.config.balance_usage_pct, 0.75) * self.active_leverage
        max_qty_by_notional = max_notional / entry_price
        raw_qty = min(raw_qty, max_qty_by_notional)

        qty = self._normalize_quantity(raw_qty)

        logging.info(
            "Pozisyon boyutu | available=%.4f risk_pct=%.2f%% risk_amount=%.4f sl_dist=%.4f leverage=%sx raw_qty=%.8f final_qty=%.8f",
            available_quote,
            self.config.risk_per_trade_pct * 100,
            risk_amount,
            sl_distance,
            self.active_leverage,
            raw_qty,
            qty,
        )
        return qty

    def _fetch_klines_rest(self, interval: str, limit: int) -> pd.DataFrame:
        retries = 0
        while retries < 5:
            try:
                if self.config.market_type == "futures":
                    raw = self.client.futures_klines(symbol=self.config.symbol, interval=interval, limit=limit)
                else:
                    raw = self.client.get_klines(symbol=self.config.symbol, interval=interval, limit=limit)

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
                df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
                df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
                return df[["open_time", "high", "low", "close", "close_time"]]
            except Exception as exc:
                retries += 1
                logging.error("REST kline çekim hatası (%s, deneme %s): %s", interval, retries, exc)
                time.sleep(1.5 * retries)

        raise ConnectionError(f"REST verisi alınamadı: interval={interval}")

    @staticmethod
    def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
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
        hl  = out["high"] - out["low"]
        hc  = (out["high"] - out["close"].shift()).abs()
        lc  = (out["low"]  - out["close"].shift()).abs()
        tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
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

    def _check_risk_guards(self) -> bool:
        if self.trading_halted:
            logging.warning("İşlem durduruldu: Güvenlik limiti devreye girdi, yeni pozisyon açılmayacak.")
            return False

        available = self._get_available_quote_balance()

        if self.initial_balance > 0:
            min_balance = self.initial_balance * self.config.min_balance_pct
            if available < min_balance:
                logging.warning(
                    "MİNİMUM BAKİYE KORUMASI: available=%.4f < min=%.4f (%%%.1f). İşlemler durduruluyor!",
                    available,
                    min_balance,
                    self.config.min_balance_pct * 100,
                )
                self.trading_halted = True
                return False

        today = pd.Timestamp.now("UTC").strftime("%Y-%m-%d")
        if self.daily_loss_date != today:
            self.daily_loss_date = today
            self.daily_loss = 0.0

        if self.initial_balance > 0:
            max_daily_loss_amount = self.initial_balance * self.config.max_daily_loss_pct
            if self.daily_loss >= max_daily_loss_amount:
                logging.warning(
                    "GÜNLÜK ZARAR LİMİTİ: daily_loss=%.4f >= limit=%.4f (%%%.1f). Bugün işlem yapılmayacak!",
                    self.daily_loss,
                    max_daily_loss_amount,
                    self.config.max_daily_loss_pct * 100,
                )
                return False

        return True

    def _record_pnl(self, net_pnl: float):
        if net_pnl < 0:
            self.daily_loss += abs(net_pnl)

    def bootstrap(self):
        warmup_limit = min(self.config.rest_limit, 1500)
        logging.info("Başlangıç verileri çekiliyor (limit=%s)...", warmup_limit)

        self.initial_balance = self._get_available_quote_balance()
        logging.info("Başlangıç bakiyesi kaydedildi: %.4f", self.initial_balance)

        for interval in [self.config.m5, self.config.m15, self.config.h1]:
            df = self._fetch_klines_rest(interval=interval, limit=warmup_limit)
            df = self._add_indicators(df)
            df = df.dropna().reset_index(drop=True)
            self.frames[interval] = df
            logging.info(
                "%s yüklendi -> son close=%.4f, EMA13=%.4f, EMA21=%.4f, EMA200=%.4f",
                interval,
                df.iloc[-1]["close"],
                df.iloc[-1]["ema13"],
                df.iloc[-1]["ema21"],
                df.iloc[-1]["ema200"],
            )

        self._load_open_position()

        dry_tag = " [DRY RUN]" if self.config.dry_run else ""
        pos_info = ""
        if self.position:
            pos_info = f"\n⚠️ Açık Pozisyon Yüklendi: {self.position['side']} @ {self.position['entry']:.4f}"
        self.tg.send(
            f"🤖 <b>Bot Başlatıldı{dry_tag}</b>\n"
            f"Sembol   : {self.config.symbol}\n"
            f"Bakiye   : {self.initial_balance:.4f} USDT\n"
            f"Kaldıraç : {self.active_leverage}x\n"
            f"Risk/İşlem: %{self.config.risk_per_trade_pct * 100:.2f}{pos_info}"
        )

    def _load_open_position(self):
        try:
            positions = self.client.futures_position_information(symbol=self.config.symbol)
            for p in positions:
                amt = float(p.get("positionAmt", 0))
                if amt == 0:
                    continue
                side = "LONG" if amt > 0 else "SHORT"
                entry = float(p.get("entryPrice", 0))
                qty = abs(amt)
                risk = entry * self.config.sl_ema_pct
                sl_mult = 1.0 - self.config.sl_ema_pct if side == "LONG" else 1.0 + self.config.sl_ema_pct
                sl_price = entry * sl_mult
                tp_price = (
                    entry + risk * self.config.rr_ratio
                    if side == "LONG"
                    else entry - risk * self.config.rr_ratio
                )
                tp1_price = (
                    entry + risk * self.config.tp1_ratio
                    if side == "LONG"
                    else entry - risk * self.config.tp1_ratio
                )
                self.position = {
                    "side": side,
                    "entry": entry,
                    "sl": sl_price,
                    "tp": tp_price,
                    "tp1": tp1_price,
                    "tp1_done": False,
                    "qty": qty,
                    "original_qty": qty,
                    "risk": risk,
                    "opened_at": pd.Timestamp.now("UTC") - pd.Timedelta(hours=1),
                    "reverse_cross_count": 0,
                    "locked": False,
                    "depletion_done": 0,
                    "depletion_last_bar": 0,
                    "depletion_bar_counter": 0,
                    "sl_order_id": None,
                }
                logging.warning(
                    "Açık pozisyon yüklendi | side=%s entry=%.4f qty=%s sl=%.4f tp=%.4f",
                    side, entry, qty, sl_price, tp_price,
                )
                return
        except Exception as exc:
            logging.warning("Açık pozisyon yüklenemedi: %s", exc)

    def _update_frame_from_kline(self, interval: str, kline: Dict):
        row = {
            "open_time": pd.to_datetime(int(kline["t"]), unit="ms"),
            "high": float(kline["h"]),
            "low": float(kline["l"]),
            "close": float(kline["c"]),
            "close_time": pd.to_datetime(int(kline["T"]), unit="ms"),
        }

        df = self.frames.get(interval)
        if df is None or df.empty:
            self.frames[interval] = pd.DataFrame([row])
        else:
            last_close_time = df.iloc[-1]["close_time"]
            if last_close_time == row["close_time"]:
                df.iloc[-1, df.columns.get_loc("open_time")] = row["open_time"]
                df.iloc[-1, df.columns.get_loc("high")] = row["high"]
                df.iloc[-1, df.columns.get_loc("low")] = row["low"]
                df.iloc[-1, df.columns.get_loc("close")] = row["close"]
                df.iloc[-1, df.columns.get_loc("close_time")] = row["close_time"]
            else:
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            self.frames[interval] = df.tail(max(self.config.rest_limit, 2200)).reset_index(drop=True)

        self.frames[interval] = self._add_indicators(self.frames[interval])
        self.frames[interval] = self.frames[interval].dropna().reset_index(drop=True)

    def _crosses(self, df: pd.DataFrame):
        prev = df.iloc[-2]
        curr = df.iloc[-1]
        if self.config.use_fast_ema:
            golden = prev["ema8"] <= prev["ema13"] and curr["ema8"] > curr["ema13"]
            death  = prev["ema8"] >= prev["ema13"] and curr["ema8"] < curr["ema13"]
        else:
            golden = prev["ema13"] <= prev["ema21"] and curr["ema13"] > curr["ema21"]
            death  = prev["ema13"] >= prev["ema21"] and curr["ema13"] < curr["ema21"]
        return golden, death

    def _ema200_slope(self, df: pd.DataFrame):
        curr = df.iloc[-1]["ema200"]
        old  = df.iloc[-1 - self.config.slope_bars]["ema200"]
        return curr > old, curr < old

    def _log_signal_snapshot(self, long_signal: bool, short_signal: bool):
        m5_df = self.frames[self.config.m5]
        m5    = m5_df.iloc[-1]
        m5p   = m5_df.iloc[-2]
        m15   = self.frames[self.config.m15].iloc[-1]
        h1    = self.frames[self.config.h1].iloc[-1]

        price = float(m5["close"])
        slope_up = float(m5["ema200"]) > float(m5_df.iloc[-1 - self.config.slope_bars]["ema200"])

        if self.config.use_fast_ema:
            cross_a_prev, cross_b_prev = float(m5p["ema8"]),  float(m5p["ema13"])
            cross_a_curr, cross_b_curr = float(m5["ema8"]),   float(m5["ema13"])
            cross_label = "EMA8/13"
        else:
            cross_a_prev, cross_b_prev = float(m5p["ema13"]), float(m5p["ema21"])
            cross_a_curr, cross_b_curr = float(m5["ema13"]),  float(m5["ema21"])
            cross_label = "EMA13/21"

        golden = cross_a_prev <= cross_b_prev and cross_a_curr > cross_b_curr
        death  = cross_a_prev >= cross_b_prev and cross_a_curr < cross_b_curr

        trend_up   = price > float(m15["ema200"]) and price > float(h1["ema200"])
        m15_bull   = float(m15["ema13"]) > float(m15["ema21"])
        adx_val    = float(m5["adx14"])

        logging.info(
            "Sinyal | close=%.4f | %s prev=%.4f/%.4f curr=%.4f/%.4f golden=%s death=%s"
            " | slope_up=%s trend_up=%s m15_bull=%s ADX=%.1f"
            " | EMA200=%.4f M15EMA200=%.4f H1EMA200=%.4f | LONG=%s SHORT=%s",
            price,
            cross_label,
            cross_a_prev, cross_b_prev,
            cross_a_curr, cross_b_curr,
            golden, death,
            slope_up, trend_up, m15_bull, adx_val,
            m5["ema200"], m15["ema200"], h1["ema200"],
            long_signal, short_signal,
        )

    def _update_hourly_report(self, long_signal: bool, short_signal: bool):
        now = pd.Timestamp.now("UTC")
        current_hour = now.hour

        self._signal_count_hour += 1
        if long_signal:
            self._signal_long_count += 1
        if short_signal:
            self._signal_short_count += 1

        if self._last_report_hour == -1:
            self._last_report_hour = current_hour

        if current_hour != self._last_report_hour:
            pos_text = "Yok"
            if self.position:
                pos_text = (
                    f"{self.position['side']} @ {self.position['entry']:.2f}"
                    f" | SL={self.position['sl']:.2f} TP={self.position['tp']:.2f}"
                    + (" [KİLİTLİ]" if self.position["locked"] else "")
                )
            balance = self._get_available_quote_balance()
            self.tg.send(
                f"📊 <b>Saatlik Rapor</b> ({self._last_report_hour:02d}:00 UTC)\n"
                f"Sorgu sayısı : {self._signal_count_hour} "
                f"(LONG={self._signal_long_count} SHORT={self._signal_short_count})\n"
                f"Bakiye       : {balance:.4f} USDT\n"
                f"Pozisyon     : {pos_text}\n"
                f"Günlük zarar : {self.daily_loss:.4f} USDT"
            )
            self._signal_count_hour = 0
            self._signal_long_count = 0
            self._signal_short_count = 0
            self._last_report_hour = current_hour

    def evaluate_signals(self):
        m5 = self.frames.get(self.config.m5)
        m15 = self.frames.get(self.config.m15)
        h1 = self.frames.get(self.config.h1)

        if m5 is None or m15 is None or h1 is None:
            return
        if len(m5) < 210 or len(m15) < 210 or len(h1) < 210:
            return

        curr = m5.iloc[-1]
        price = curr["close"]

        trend_up   = price > m15.iloc[-1]["ema200"] and price > h1.iloc[-1]["ema200"]
        trend_down = price < m15.iloc[-1]["ema200"] and price < h1.iloc[-1]["ema200"]

        slope_up, slope_down = self._ema200_slope(m5)
        golden, death = self._crosses(m5)

        candle_above_ema200 = price > curr["ema200"]
        candle_below_ema200 = price < curr["ema200"]

        long_signal  = trend_up   and slope_up   and golden and candle_above_ema200
        short_signal = trend_down and slope_down and death  and candle_below_ema200

        if self.config.m15_trend_filter:
            m15_bull = m15.iloc[-1]["ema13"] > m15.iloc[-1]["ema21"]
            m15_bear = m15.iloc[-1]["ema13"] < m15.iloc[-1]["ema21"]
            if long_signal  and not m15_bull:
                long_signal = False
            if short_signal and not m15_bear:
                short_signal = False

        if self.config.adx_min > 0:
            if float(curr["adx14"]) < self.config.adx_min:
                long_signal  = False
                short_signal = False

        if self.config.entry_confirm_bar and len(m5) >= 3:
            pprev = m5.iloc[-3]
            prev_bar = m5.iloc[-2]
            curr_bar = m5.iloc[-1]
            if self.config.use_fast_ema:
                pg_a, pg_b = pprev["ema8"],    pprev["ema13"]
                cg_a, cg_b = prev_bar["ema8"], prev_bar["ema13"]
                cc_a, cc_b = curr_bar["ema8"], curr_bar["ema13"]
            else:
                pg_a, pg_b = pprev["ema13"],    pprev["ema21"]
                cg_a, cg_b = prev_bar["ema13"], prev_bar["ema21"]
                cc_a, cc_b = curr_bar["ema13"], curr_bar["ema21"]
            prev_golden  = pg_a <= pg_b and cg_a > cg_b
            prev_death   = pg_a >= pg_b and cg_a < cg_b
            still_golden = cg_a > cg_b and cc_a > cc_b
            still_death  = cg_a < cg_b and cc_a < cc_b
            if long_signal  and not (prev_golden and still_golden):
                long_signal = False
            if short_signal and not (prev_death and still_death):
                short_signal = False

        if self.config.rsi_filter:
            h1_rsi = float(h1.iloc[-1]["rsi14"])
            if long_signal  and h1_rsi > 65:
                long_signal = False
            if short_signal and h1_rsi < 35:
                short_signal = False

        self._log_signal_snapshot(long_signal, short_signal)
        self._update_hourly_report(long_signal, short_signal)

        if self.position is None:
            if long_signal or short_signal:
                if not self._check_risk_guards():
                    return
            if long_signal:
                self.open_position("LONG")
            elif short_signal:
                self.open_position("SHORT")
        else:
            self.manage_open_position()

    def open_position(self, side: str):
        m5 = self.frames[self.config.m5]
        curr = m5.iloc[-1]
        entry_price = float(curr["close"])

        if side == "SHORT" and self.config.market_type == "spot":
            logging.warning("Spot modunda SHORT desteklenmediği için sinyal atlandı.")
            return

        sl_ema_mult = 1.0 - self.config.sl_ema_pct if side == "LONG" else 1.0 + self.config.sl_ema_pct
        if side == "LONG":
            sl_last3 = float(m5.tail(3)["low"].min())
            sl_ema   = float(curr["ema21"]) * sl_ema_mult
            sl_price = max(sl_last3, sl_ema)
            risk = entry_price - sl_price
            if risk <= 0:
                logging.warning("LONG risk hesaplaması geçersiz, işlem açılmadı.")
                return
            tp_price = entry_price + (risk * self.config.rr_ratio)
            order_side = SIDE_BUY
        else:
            sl_last3 = float(m5.tail(3)["high"].max())
            sl_ema   = float(curr["ema21"]) * sl_ema_mult
            sl_price = min(sl_last3, sl_ema)
            risk = sl_price - entry_price
            if risk <= 0:
                logging.warning("SHORT risk hesaplaması geçersiz, işlem açılmadı.")
                return
            tp_price = entry_price - (risk * self.config.rr_ratio)
            order_side = SIDE_SELL

        quantity = self._calculate_order_quantity(entry_price, sl_price, side)

        if quantity <= 0:
            logging.warning("Hesaplanan miktar geçersiz (<=0). İşlem açılmadı.")
            return

        try:
            if not self.config.dry_run:
                self._place_market_order(order_side, quantity)

            pos_risk = abs(entry_price - sl_price)
            tp1_price = (
                entry_price + pos_risk * self.config.tp1_ratio
                if side == "LONG"
                else entry_price - pos_risk * self.config.tp1_ratio
            )
            self.position = {
                "side": side,
                "entry": entry_price,
                "sl": sl_price,
                "tp": tp_price,
                "tp1": tp1_price,
                "tp1_done": False,
                "qty": quantity,
                "original_qty": quantity,
                "risk": pos_risk,
                "opened_at": pd.Timestamp.now("UTC"),
                "reverse_cross_count": 0,
                "locked": False,
                "depletion_done": 0,
                "depletion_last_bar": 0,
                "depletion_bar_counter": 0,
                "sl_order_id": None,
            }
            self._place_sl_order(sl_price, side)
            logging.info(
                "POZISYON AÇILDI | side=%s entry=%.4f sl=%.4f tp1=%.4f tp=%.4f qty=%s dry_run=%s",
                side,
                entry_price,
                sl_price,
                tp1_price,
                tp_price,
                quantity,
                self.config.dry_run,
            )
            dry_tag = " [DRY RUN]" if self.config.dry_run else ""
            self.tg.send(
                f"🟢 <b>POZİSYON AÇILDI{dry_tag}</b>\n"
                f"Sembol : {self.config.symbol}\n"
                f"Yön    : {side}\n"
                f"Giriş  : {entry_price:.4f}\n"
                f"SL     : {sl_price:.4f}\n"
                f"TP1    : {tp1_price:.4f} (%{self.config.tp1_qty_pct*100:.0f} kapatılır)\n"
                f"TP2    : {tp_price:.4f} (kalan kapatılır)"
            )
        except Exception as exc:
            logging.error("Pozisyon açma hatası: %s", exc)

    def manage_open_position(self):
        if self.position is None:
            return

        m5 = self.frames[self.config.m5]
        prev = m5.iloc[-2]
        curr = m5.iloc[-1]
        price = float(curr["close"])
        side = self.position["side"]

        # --- Kilitli pozisyon: parça parça eritme ---
        if self.position["locked"]:
            self.position["depletion_bar_counter"] += 1
            parts_done = self.position["depletion_done"]
            total_parts = self.config.depletion_parts
            if parts_done < total_parts:
                if self.position["depletion_bar_counter"] >= self.config.depletion_interval_bars:
                    original_qty = self.position["original_qty"]
                    part_qty = original_qty / total_parts
                    part_qty = min(part_qty, self.position["qty"])
                    part_qty = self._normalize_quantity(part_qty)
                    if part_qty <= 0:
                        self.position["depletion_done"] += 1
                        self.position["depletion_bar_counter"] = 0
                        return
                    self._close_partial(part_qty, price, f"ERIME_{parts_done + 1}/{total_parts}")
                    self.position["depletion_done"] += 1
                    self.position["depletion_bar_counter"] = 0
                    self.position["qty"] -= part_qty
                    if self.position["depletion_done"] >= total_parts or self.position["qty"] <= 0:
                        self.position = None
            else:
                self.position = None
            return

        # --- TP1: kısmi kar alma + breakeven SL ---
        if not self.position["tp1_done"] and not self.position["locked"]:
            hit_tp1 = price >= self.position["tp1"] if side == "LONG" else price <= self.position["tp1"]
            if hit_tp1:
                tp1_qty = self._normalize_quantity(self.position["qty"] * self.config.tp1_qty_pct)
                if tp1_qty > 0:
                    self._close_partial(tp1_qty, price, "TP1")
                    self.position["qty"] -= tp1_qty
                self.position["tp1_done"] = True
                if self.config.breakeven_after_tp1:
                    self.position["sl"] = self.position["entry"]
                    self._cancel_sl_order()
                    self._place_sl_order(self.position["entry"], side)
                    logging.info(
                        "BREAKEVEN SL | SL giriş fiyatına çekildi: %.4f",
                        self.position["entry"],
                    )
                return

        # --- Kilit tetikleme: zarar eşiği aşıldı mı? ---
        unrealized_pnl = (
            (price - self.position["entry"]) * self.position["qty"]
            if side == "LONG"
            else (self.position["entry"] - price) * self.position["qty"]
        )
        lock_threshold = -self.position["risk"] * self.position["qty"] * self.config.lock_loss_pct
        if unrealized_pnl <= lock_threshold:
            logging.warning(
                "POZİSYON KİLİTLENDİ: unrealized=%.2f <= threshold=%.2f | %s | entry=%.4f",
                unrealized_pnl, lock_threshold, side, self.position["entry"],
            )
            self.position["locked"] = True
            self.position["depletion_bar_counter"] = 0
            self._cancel_sl_order()
            return

        # --- 24 saat zorla kapatma ---
        hours_open = (pd.Timestamp.now("UTC") - self.position["opened_at"]).total_seconds() / 3600
        if hours_open >= self.config.max_trade_hours:
            logging.warning(
                "ZAMAN ASIMI: pozisyon %.1f saat açık. Zorla kapatılıyor.",
                hours_open,
            )
            self.close_position("ZAMAN_ASIMI")
            return

        # --- Normal çıkış: SL / TP / ters kesişim ---
        hit_sl = price <= self.position["sl"] if side == "LONG" else price >= self.position["sl"]
        hit_tp = price >= self.position["tp"] if side == "LONG" else price <= self.position["tp"]

        if self.config.use_fast_ema:
            rev_prev_a, rev_prev_b = prev["ema8"],  prev["ema13"]
            rev_curr_a, rev_curr_b = curr["ema8"],  curr["ema13"]
        else:
            rev_prev_a, rev_prev_b = prev["ema13"], prev["ema21"]
            rev_curr_a, rev_curr_b = curr["ema13"], curr["ema21"]
        raw_reverse = False
        if side == "LONG":
            raw_reverse = rev_prev_a >= rev_prev_b and rev_curr_a < rev_curr_b
        elif side == "SHORT":
            raw_reverse = rev_prev_a <= rev_prev_b and rev_curr_a > rev_curr_b

        if raw_reverse:
            self.position["reverse_cross_count"] += 1
        else:
            self.position["reverse_cross_count"] = 0

        confirm_bars = self.config.reverse_cross_confirm_bars
        min_profit_ratio = self.config.reverse_cross_min_profit_ratio
        price_diff = (price - self.position["entry"]) if side == "LONG" else (self.position["entry"] - price)
        profit_in_risk_units = price_diff / self.position["risk"] if self.position["risk"] > 0 else 0
        enough_profit = profit_in_risk_units >= min_profit_ratio
        confirmed_reverse = self.position["reverse_cross_count"] >= confirm_bars

        if hit_sl:
            self.close_position("SL")
        elif hit_tp:
            self.close_position("TP")
        elif confirmed_reverse and enough_profit:
            self.close_position("TERS_KESISIM")
        elif confirmed_reverse and not enough_profit:
            logging.info(
                "TERS_KESİSİM bekletiliyor | profit_ratio=%.2f < min=%.2f | reverse_count=%d",
                profit_in_risk_units,
                min_profit_ratio,
                self.position["reverse_cross_count"],
            )

    def close_position(self, reason: str):
        if self.position is None:
            return

        side = self.position["side"]
        exit_side = SIDE_SELL if side == "LONG" else SIDE_BUY
        exit_price = float(self.frames[self.config.m5].iloc[-1]["close"])
        qty = self.position["qty"]
        entry_price = self.position["entry"]

        try:
            self._cancel_sl_order()
            if not self.config.dry_run:
                self._place_market_order(exit_side, qty)

            price_diff = (exit_price - entry_price) if side == "LONG" else (entry_price - exit_price)
            gross_pnl = price_diff * qty
            fee_rate = 0.0004
            fee = exit_price * qty * fee_rate
            net_pnl = gross_pnl - fee

            self._record_pnl(net_pnl)

            logging.info(
                "POZISYON KAPATILDI | reason=%s side=%s exit=%.4f gross_pnl=%.4f net_pnl=%.4f daily_loss=%.4f",
                reason,
                side,
                exit_price,
                gross_pnl,
                net_pnl,
                self.daily_loss,
            )
            emoji = "✅" if net_pnl >= 0 else "🔴"
            dry_tag = " [DRY RUN]" if self.config.dry_run else ""
            self.tg.send(
                f"{emoji} <b>POZİSYON KAPATILDI{dry_tag}</b>\n"
                f"Sebep  : {reason}\n"
                f"Yön    : {side}\n"
                f"Çıkış  : {exit_price:.4f}\n"
                f"Net PnL: {net_pnl:+.4f} USDT\n"
                f"Günlük Zarar: {self.daily_loss:.4f} USDT"
            )
            self.position = None
        except Exception as exc:
            logging.error("Pozisyon kapatma hatası: %s", exc)

    def _close_partial(self, qty: float, price: float, reason: str):
        if self.position is None:
            return
        side = self.position["side"]
        exit_side = SIDE_SELL if side == "LONG" else SIDE_BUY
        entry_price = self.position["entry"]
        try:
            if not self.config.dry_run:
                self._place_market_order(exit_side, qty)
            price_diff = (price - entry_price) if side == "LONG" else (entry_price - price)
            gross_pnl = price_diff * qty
            fee_rate = 0.0004
            fee = price * qty * fee_rate
            net_pnl = gross_pnl - fee
            self._record_pnl(net_pnl)
            logging.info(
                "KISMİ KAPANIŞ | reason=%s side=%s qty=%s exit=%.4f gross=%.4f net=%.4f",
                reason, side, qty, price, gross_pnl, net_pnl,
            )
            emoji = "⚠️" if net_pnl < 0 else "🔸"
            dry_tag = " [DRY RUN]" if self.config.dry_run else ""
            self.tg.send(
                f"{emoji} <b>KISMİ KAPANIŞ{dry_tag}</b> ({reason})\n"
                f"Yön    : {side}\n"
                f"Çıkış  : {price:.4f}\n"
                f"Miktar : {qty}\n"
                f"Net PnL: {net_pnl:+.4f} USDT"
            )
        except Exception as exc:
            logging.error("Kısmi kapanış hatası: %s", exc)

    def _place_sl_order(self, sl_price: float, position_side: str):
        if self.config.dry_run or self.config.market_type != "futures":
            return
        try:
            sl_side = SIDE_SELL if position_side == "LONG" else SIDE_BUY
            resp = self.client.futures_create_order(
                symbol=self.config.symbol,
                side=sl_side,
                type="STOP_MARKET",
                stopPrice=round(sl_price, 1),
                closePosition=True,
                timeInForce="GTE_GTC",
            )
            order_id = resp.get("orderId")
            self.position["sl_order_id"] = order_id
            logging.info("Exchange SL emri yerleştirildi | orderId=%s stopPrice=%.4f", order_id, sl_price)
        except Exception as exc:
            logging.warning("Exchange SL emri yerleştirilemedi: %s", exc)

    def _cancel_sl_order(self):
        if self.config.dry_run or self.config.market_type != "futures":
            return
        if not self.position:
            return
        order_id = self.position.get("sl_order_id")
        if not order_id:
            return
        try:
            self.client.futures_cancel_order(symbol=self.config.symbol, orderId=order_id)
            self.position["sl_order_id"] = None
            logging.info("Exchange SL emri iptal edildi | orderId=%s", order_id)
        except Exception as exc:
            logging.warning("Exchange SL emri iptal edilemedi: %s", exc)

    def _place_market_order(self, side: str, quantity: float):
        if self.config.market_type == "futures":
            self.client.futures_create_order(
                symbol=self.config.symbol,
                side=side,
                type=ORDER_TYPE_MARKET,
                quantity=quantity,
            )
        else:
            self.client.create_order(
                symbol=self.config.symbol,
                side=side,
                type=ORDER_TYPE_MARKET,
                quantity=quantity,
            )

    def _socket_message_handler(self, msg: Dict):
        try:
            if not isinstance(msg, dict):
                return

            if msg.get("e") == "error":
                logging.error("WebSocket hata mesajı: %s", msg)
                self.reconnect_requested = True
                return

            kline = msg.get("k")
            if not kline:
                return

            if not kline.get("x", False):
                return

            interval = kline.get("i")
            if interval not in [self.config.m5, self.config.m15, self.config.h1]:
                return

            with self.lock:
                self._update_frame_from_kline(interval=interval, kline=kline)
                if interval == self.config.m5:
                    self.evaluate_signals()

        except Exception as exc:
            logging.error("Socket mesaj işleme hatası: %s", exc)
            logging.debug(traceback.format_exc())

    @staticmethod
    def _user_stream_handler(msg: Dict):
        if not isinstance(msg, dict):
            return
        event_type = msg.get("e")
        if event_type in {"executionReport", "ORDER_TRADE_UPDATE", "ACCOUNT_UPDATE"}:
            logging.info("User Stream | %s | %s", event_type, msg)

    def _start_kline_socket(self, interval: str):
        ws_symbol = self.config.symbol.lower()

        if self.config.market_type == "futures":
            start_fn = getattr(self.ws_manager, "start_kline_futures_socket", None)
            if start_fn:
                start_fn(callback=self._socket_message_handler, symbol=ws_symbol, interval=interval)
                return

        self.ws_manager.start_kline_socket(callback=self._socket_message_handler, symbol=ws_symbol, interval=interval)

    def _start_user_socket(self):
        if self.config.market_type == "futures":
            futures_user_fn = getattr(self.ws_manager, "start_futures_user_socket", None)
            if futures_user_fn:
                futures_user_fn(callback=self._user_stream_handler)
                return

        self.ws_manager.start_user_socket(callback=self._user_stream_handler)

    def start_websockets(self):
        self.ws_manager = ThreadedWebsocketManager(
            api_key=self.api_key,
            api_secret=self.api_secret,
            testnet=self.config.testnet,
        )
        self.ws_manager.start()

        self._start_user_socket()
        self._start_kline_socket(self.config.m5)
        self._start_kline_socket(self.config.m15)
        self._start_kline_socket(self.config.h1)

        logging.info("WebSocket streamleri başlatıldı (User Data + M5/M15/H1 Kline).")

    def stop_websockets(self):
        if self.ws_manager:
            try:
                self.ws_manager.stop()
                self.ws_manager.join(timeout=10)
            except Exception as exc:
                logging.warning("WebSocket durdurma hatası: %s", exc)
            self.ws_manager = None
            time.sleep(2)

    def run(self):
        self.bootstrap()

        try:
            self.reconnect_requested = False
            self.start_websockets()
            logging.info("Bot çalışıyor...")

            while True:
                if self.reconnect_requested:
                    logging.error("WebSocket bağlantısı kesildi. Bot yeniden başlatılmak üzere çıkıyor...")
                    self.stop_websockets()
                    sys.exit(1)
                time.sleep(1)

        except KeyboardInterrupt:
            logging.info("Bot kullanıcı tarafından durduruldu.")
            self.stop_websockets()
            sys.exit(0)
        except Exception as exc:
            logging.error("Çalışma hatası: %s", exc)
            self.stop_websockets()
            sys.exit(1)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    config = BotConfig()
    bot = MultiTimeframeEMABot(config)
    bot.run()


if __name__ == "__main__":
    main()
