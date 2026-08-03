from __future__ import annotations

import os
import time
from datetime import datetime, time as dtime
from typing import Dict, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from dotenv import load_dotenv

from strategy import GerkoXAUModel, Signal, format_signal
from telegram_bot import TelegramBot

load_dotenv()

# ============================================================
# ENVIRONMENT CONFIGURATION
# ============================================================

TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "").strip()
SYMBOL = os.getenv("SYMBOL", "XAU/USD").strip()

CHECK_INTERVAL_SECONDS = int(
    os.getenv("CHECK_INTERVAL_SECONDS", "300")
)

MIN_SCORE = int(
    os.getenv("MIN_SCORE", "80")
)

MAX_SIGNALS_PER_DAY = int(
    os.getenv("MAX_SIGNALS_PER_DAY", "8")
)

SIGNAL_COOLDOWN_MINUTES = int(
    os.getenv("SIGNAL_COOLDOWN_MINUTES", "45")
)

BOT_TIMEZONE = os.getenv(
    "BOT_TIMEZONE",
    "Europe/London",
).strip()

SEND_STARTUP_MESSAGE = (
    os.getenv("SEND_STARTUP_MESSAGE", "true")
    .strip()
    .lower()
    in {"1", "true", "yes", "on"}
)

# ============================================================
# SESSION WINDOWS
# ============================================================

# London/UK local time.
SCOUTING_WINDOWS = [
    (dtime(23, 0), dtime(3, 0)),
    (dtime(7, 0), dtime(11, 0)),
    (dtime(12, 0), dtime(17, 0)),
]


def get_timezone() -> ZoneInfo:
    try:
        return ZoneInfo(BOT_TIMEZONE)
    except Exception:
        print(
            f"Invalid BOT_TIMEZONE '{BOT_TIMEZONE}'. "
            "Falling back to Europe/London."
        )
        return ZoneInfo("Europe/London")


LOCAL_TIMEZONE = get_timezone()


def now_local() -> datetime:
    return datetime.now(LOCAL_TIMEZONE)


def inside_window(
    current: dtime,
    start: dtime,
    end: dtime,
) -> bool:
    """
    Check whether current time is inside a session window.

    Supports windows crossing midnight, such as 23:00-03:00.
    """

    if start <= end:
        return start <= current <= end

    return current >= start or current <= end


def market_weekend_closed(dt: datetime) -> bool:
    """
    Basic XAUUSD weekend filter.

    Stops scanning:
    - Friday from 22:00 London time
    - All Saturday
    - Sunday before 22:00 London time
    """

    weekday = dt.weekday()

    if weekday == 4 and dt.time() >= dtime(22, 0):
        return True

    if weekday == 5:
        return True

    if weekday == 6 and dt.time() < dtime(22, 0):
        return True

    return False


def is_trading_time(dt: datetime) -> bool:
    if market_weekend_closed(dt):
        return False

    return any(
        inside_window(dt.time(), start, end)
        for start, end in SCOUTING_WINDOWS
    )


# ============================================================
# MARKET DATA
# ============================================================

def fetch_twelve_data(
    interval: str,
    outputsize: int = 300,
) -> pd.DataFrame:
    if not TWELVE_DATA_API_KEY:
        raise RuntimeError(
            "TWELVE_DATA_API_KEY is missing from environment variables."
        )

    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": SYMBOL,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVE_DATA_API_KEY,
        "format": "JSON",
        "timezone": BOT_TIMEZONE,
    }

    response = requests.get(
        url,
        params=params,
        timeout=30,
    )

    response.raise_for_status()
    data = response.json()

    if data.get("status") == "error":
        message = data.get(
            "message",
            "Unknown Twelve Data error",
        )

        raise RuntimeError(
            f"Twelve Data error for {interval}: {message}"
        )

    values = data.get("values")

    if not values:
        raise RuntimeError(
            f"No candle values returned for {interval}: {data}"
        )

    dataframe = pd.DataFrame(values)

    required_columns = {
        "datetime",
        "open",
        "high",
        "low",
        "close",
    }

    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise RuntimeError(
            f"Missing columns for {interval}: "
            f"{sorted(missing_columns)}"
        )

    dataframe["datetime"] = pd.to_datetime(
        dataframe["datetime"],
        errors="coerce",
    )

    dataframe = (
        dataframe
        .dropna(subset=["datetime"])
        .sort_values("datetime")
        .drop_duplicates(
            subset=["datetime"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return dataframe[
        [
            "datetime",
            "open",
            "high",
            "low",
            "close",
        ]
    ]


def fetch_all_candles() -> Dict[str, pd.DataFrame]:
    return {
        "h1": fetch_twelve_data("1h", 350),
        "m15": fetch_twelve_data("15min", 350),
        "m5": fetch_twelve_data("5min", 350),
    }


# ============================================================
# SIGNAL CONTROL
# ============================================================

def build_signal_key(
    signal: Signal,
) -> Tuple[str, float, float, float]:
    """
    Creates a duplicate-control key.

    Prices are rounded to reduce duplicate alerts caused by tiny
    data-provider price differences.
    """

    return (
        signal.side,
        round(signal.entry_mid, 1),
        round(signal.stop_loss, 1),
        round(signal.tp2, 1),
    )


def cooldown_finished(
    last_signal_time: Optional[datetime],
    current_time: datetime,
) -> bool:
    if last_signal_time is None:
        return True

    elapsed_seconds = (
        current_time - last_signal_time
    ).total_seconds()

    required_seconds = SIGNAL_COOLDOWN_MINUTES * 60

    return elapsed_seconds >= required_seconds


# ============================================================
# TELEGRAM MESSAGES
# ============================================================

def heartbeat_message() -> str:
    current_time = now_local().strftime(
        "%d %b %Y, %H:%M:%S"
    )

    return f"""
🧠⚡ <b>GERKO-XAU QUANT ONLINE</b>
━━━━━━━━━━━━━━━━━━━━

<b>Model:</b> GERKO-MIMIC XAU
<b>Symbol:</b> XAUUSD
<b>Status:</b> ACTIVE
<b>Minimum Score:</b> {MIN_SCORE}/100
<b>Daily Signal Limit:</b> {MAX_SIGNALS_PER_DAY}
<b>Scan Interval:</b> {CHECK_INTERVAL_SECONDS} seconds
<b>Time:</b> {current_time}

The Gerko engine is now scanning for high-confidence,
probability-based XAUUSD setups.

#GERKO_XAU #GX_SYSTEM
""".strip()


def daily_limit_message() -> str:
    return f"""
🧠🛑 <b>GERKO-XAU DAILY LIMIT</b>
━━━━━━━━━━━━━━━━━━━━

The Gerko engine has reached its daily limit of
<b>{MAX_SIGNALS_PER_DAY} signals</b>.

No additional Gerko signals will be sent today.

#GERKO_XAU #GX_LIMIT
""".strip()


# ============================================================
# MAIN APPLICATION
# ============================================================

def main() -> None:
    telegram = TelegramBot()

    model = GerkoXAUModel(
        min_score=MIN_SCORE,
    )

    current_day = now_local().date()
    sent_today = 0

    last_signal_key: Optional[
        Tuple[str, float, float, float]
    ] = None

    last_signal_time: Optional[datetime] = None
    daily_limit_notification_sent = False

    print("=" * 60)
    print("Gerko-XAU Quant Bot started")
    print(f"Symbol: {SYMBOL}")
    print(f"Timezone: {BOT_TIMEZONE}")
    print(f"Minimum score: {MIN_SCORE}")
    print(f"Scan interval: {CHECK_INTERVAL_SECONDS}s")
    print("=" * 60)

    if SEND_STARTUP_MESSAGE:
        telegram.send(
            heartbeat_message()
        )

    while True:
        cycle_started = now_local()

        try:
            # Reset daily counters after the local date changes.
            if cycle_started.date() != current_day:
                current_day = cycle_started.date()
                sent_today = 0
                last_signal_key = None
                last_signal_time = None
                daily_limit_notification_sent = False

                print(
                    f"{cycle_started.isoformat()} - "
                    "New trading day. Counters reset."
                )

            if not is_trading_time(cycle_started):
                print(
                    f"{cycle_started.isoformat()} - "
                    "Outside Gerko scanning window."
                )

                time.sleep(
                    CHECK_INTERVAL_SECONDS
                )

                continue

            if sent_today >= MAX_SIGNALS_PER_DAY:
                print(
                    f"{cycle_started.isoformat()} - "
                    "Maximum daily signals reached."
                )

                if not daily_limit_notification_sent:
                    telegram.send(
                        daily_limit_message()
                    )

                    daily_limit_notification_sent = True

                time.sleep(
                    CHECK_INTERVAL_SECONDS
                )

                continue

            print(
                f"{cycle_started.isoformat()} - "
                "Fetching XAUUSD market data..."
            )

            candles = fetch_all_candles()

            signal = model.analyse(
                candles=candles,
                signal_time=cycle_started,
            )

            if signal is None:
                print(
                    f"{cycle_started.isoformat()} - "
                    "No Gerko-grade setup."
                )

                time.sleep(
                    CHECK_INTERVAL_SECONDS
                )

                continue

            signal_key = build_signal_key(
                signal
            )

            if signal_key == last_signal_key:
                print(
                    f"{cycle_started.isoformat()} - "
                    "Duplicate Gerko signal ignored."
                )

                time.sleep(
                    CHECK_INTERVAL_SECONDS
                )

                continue

            if not cooldown_finished(
                last_signal_time,
                cycle_started,
            ):
                print(
                    f"{cycle_started.isoformat()} - "
                    "Gerko signal cooldown active."
                )

                time.sleep(
                    CHECK_INTERVAL_SECONDS
                )

                continue

            message = format_signal(
                signal
            )

            message_sent = telegram.send(
                message
            )

            if message_sent:
                sent_today += 1
                last_signal_key = signal_key
                last_signal_time = cycle_started

                print(
                    f"Gerko signal sent: "
                    f"{signal.signal_id}"
                )
            else:
                print(
                    "Gerko setup detected, but Telegram "
                    "delivery was unsuccessful."
                )

            print(message)

        except KeyboardInterrupt:
            print(
                "Gerko-XAU Quant Bot stopped manually."
            )

            break

        except requests.RequestException as exc:
            print(
                f"MARKET DATA NETWORK ERROR: {exc}"
            )

        except Exception as exc:
            print(
                f"GERKO BOT ERROR: "
                f"{type(exc).__name__}: {exc}"
            )

        time.sleep(
            CHECK_INTERVAL_SECONDS
        )


if __name__ == "__main__":
    main()