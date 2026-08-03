from __future__ import annotations

import os
import time
from datetime import datetime, time as dtime
from typing import Dict

import pandas as pd
import requests
from dotenv import load_dotenv

from strategy import GerkoXAUModel, format_signal
from telegram_bot import TelegramBot

load_dotenv()

TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "").strip()
SYMBOL = os.getenv("SYMBOL", "XAU/USD")
CHECK_INTERVAL_SECONDS = int(os.getenv("CHECK_INTERVAL_SECONDS", "300"))
MIN_SCORE = int(os.getenv("MIN_SCORE", "80"))
MAX_SIGNALS_PER_DAY = int(os.getenv("MAX_SIGNALS_PER_DAY", "8"))

# London/UK local session windows. Render runs UTC by default, so set TZ=Europe/London in Render.
SCOUTING_WINDOWS = [
    (dtime(23, 0), dtime(3, 0)),
    (dtime(7, 0), dtime(11, 0)),
    (dtime(12, 0), dtime(17, 0)),
]


def now_local() -> datetime:
    return datetime.now()


def inside_window(current: dtime, start: dtime, end: dtime) -> bool:
    if start <= end:
        return start <= current <= end
    return current >= start or current <= end


def is_trading_time(dt: datetime) -> bool:
    # Sleep from Friday 22:00 to Sunday 22:00.
    # weekday: Monday=0, Sunday=6
    if dt.weekday() == 4 and dt.time() >= dtime(22, 0):
        return False
    if dt.weekday() == 5:
        return False
    if dt.weekday() == 6 and dt.time() < dtime(22, 0):
        return False
    return any(inside_window(dt.time(), start, end) for start, end in SCOUTING_WINDOWS)


def fetch_twelve_data(interval: str, outputsize: int = 250) -> pd.DataFrame:
    if not TWELVE_DATA_API_KEY:
        raise RuntimeError("Missing TWELVE_DATA_API_KEY in environment")

    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": SYMBOL,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVE_DATA_API_KEY,
        "format": "JSON",
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    if "values" not in data:
        raise RuntimeError(f"Twelve Data error for {interval}: {data}")

    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df[["datetime", "open", "high", "low", "close"]]


def fetch_all_candles() -> Dict[str, pd.DataFrame]:
    return {
        "h1": fetch_twelve_data("1h", 300),
        "m15": fetch_twelve_data("15min", 300),
        "m5": fetch_twelve_data("5min", 300),
    }


def heartbeat_message() -> str:
    dt = now_local().strftime("%Y-%m-%d %H:%M:%S")
    return f"🟡 Gerko-XAU bot alive: {dt}\nMode: scanning only high-confidence XAUUSD setups."


def main() -> None:
    bot = TelegramBot()
    model = GerkoXAUModel(min_score=MIN_SCORE)

    sent_today = 0
    current_day = now_local().date()
    last_signal_key = None

    print("Gerko-XAU Quant Bot started")
    bot.send(heartbeat_message())

    while True:
        try:
            dt = now_local()
            if dt.date() != current_day:
                current_day = dt.date()
                sent_today = 0
                last_signal_key = None

            if not is_trading_time(dt):
                print(f"{dt} - Outside trading window. Sleeping.")
                time.sleep(CHECK_INTERVAL_SECONDS)
                continue

            if sent_today >= MAX_SIGNALS_PER_DAY:
                print(f"{dt} - Max daily signals reached.")
                time.sleep(CHECK_INTERVAL_SECONDS)
                continue

            candles = fetch_all_candles()
            signal = model.analyse(candles)

            if signal is None:
                print(f"{dt} - No high-quality setup.")
                time.sleep(CHECK_INTERVAL_SECONDS)
                continue

            signal_key = (signal.side, signal.entry_low, signal.entry_high, signal.stop_loss, signal.tp2)
            if signal_key == last_signal_key:
                print(f"{dt} - Duplicate signal ignored.")
                time.sleep(CHECK_INTERVAL_SECONDS)
                continue

            message = format_signal(signal)
            bot.send(message)
            print(message)
            sent_today += 1
            last_signal_key = signal_key

        except Exception as exc:
            print(f"ERROR: {exc}")

        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
