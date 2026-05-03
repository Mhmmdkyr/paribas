import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BOT_SCRIPT = Path(__file__).parent / "ema_crossover_bot.py"
PYTHON = sys.executable

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

RESTART_DELAY = 10
MAX_RESTARTS_PER_HOUR = 10


def tg_send(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = urllib.parse.urlencode(
            {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
        ).encode()
        req = urllib.request.Request(url, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=5):
            pass
    except Exception as exc:
        print(f"[watchdog] Telegram gönderim hatası: {exc}")


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts} | WATCHDOG | {msg}", flush=True)


def fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}s {m}dk {s}sn"
    if m:
        return f"{m}dk {s}sn"
    return f"{s}sn"


def run():
    restarts_this_hour: list[float] = []
    total_restarts = 0
    watchdog_start = time.time()

    tg_send(
        f"🛡️ <b>Watchdog başlatıldı</b>\n"
        f"Zaman: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Bot izleniyor — çökerse otomatik yeniden başlatılacak.\n"
        f"Maks restart/saat: {MAX_RESTARTS_PER_HOUR}"
    )
    log("Watchdog başlatıldı.")

    while True:
        attempt = total_restarts + 1
        now = time.time()

        restarts_this_hour = [t for t in restarts_this_hour if now - t < 3600]
        if len(restarts_this_hour) >= MAX_RESTARTS_PER_HOUR:
            msg = (
                f"🚨 <b>Watchdog: Çok fazla restart!</b>\n"
                f"Son 1 saatte {MAX_RESTARTS_PER_HOUR} kez yeniden başlatıldı.\n"
                f"Toplam restart: {total_restarts}\n"
                f"Watchdog süresi: {fmt_duration(time.time() - watchdog_start)}\n"
                f"⛔ Watchdog durduruluyor — manuel müdahale gerekli."
            )
            log(msg)
            tg_send(msg)
            sys.exit(1)

        if total_restarts > 0:
            tg_send(
                f"🔄 <b>Bot yeniden başlatılıyor</b> (#{attempt})\n"
                f"Zaman: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Son 1 saatteki restart: {len(restarts_this_hour) + 1}/{MAX_RESTARTS_PER_HOUR}"
            )

        log(f"Bot başlatılıyor (deneme #{attempt})...")
        start_time = time.time()

        try:
            proc = subprocess.run(
                [PYTHON, str(BOT_SCRIPT)],
                cwd=str(BOT_SCRIPT.parent),
            )
            exit_code = proc.returncode
        except Exception as exc:
            exit_code = -1
            log(f"Bot başlatma hatası: {exc}")

        elapsed = time.time() - start_time

        if exit_code == 0:
            log("Bot normal şekilde sonlandı (exit 0). Watchdog duruyor.")
            tg_send(
                f"ℹ️ <b>Bot normal sonlandı.</b>\n"
                f"Toplam çalışma: {fmt_duration(time.time() - watchdog_start)}\n"
                f"Toplam restart: {total_restarts}\n"
                f"Watchdog durdu."
            )
            break

        total_restarts += 1
        restarts_this_hour.append(time.time())

        crash_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tg_send(
            f"💥 <b>Bot çöktü!</b>\n"
            f"Zaman       : {crash_time}\n"
            f"Çıkış kodu  : {exit_code}\n"
            f"Çalışma süresi: {fmt_duration(elapsed)}\n"
            f"Toplam crash: {total_restarts}\n"
            f"⏳ {RESTART_DELAY} saniye sonra yeniden başlatılıyor..."
        )
        log(f"Bot çöktü (exit={exit_code}, süre={fmt_duration(elapsed)}). {RESTART_DELAY}sn sonra restart.")

        time.sleep(RESTART_DELAY)


if __name__ == "__main__":
    run()
