from __future__ import annotations

import os
import time
from datetime import datetime, time as dtime
from typing import Dict, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from dotenv import load_dotenv

from strategy import (
    GerkoXAUModel,
    Signal,
    format_signal,
)

from telegram_bot import TelegramBot


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


TWELVE_DATA_API_KEY = os.getenv(
    "TWELVE_DATA_API_KEY",
    "",
).strip()

SYMBOL = os.getenv(
    "SYMBOL",
    "XAU/USD",
).strip()

BOT_TIMEZONE = os.getenv(
    "BOT_TIMEZONE",
    "Europe/London",
).strip()

CHECK_INTERVAL_SECONDS = int(
    os.getenv(
        "CHECK_INTERVAL_SECONDS",
        "180",
    )
)

MAX_SIGNALS_PER_DAY = int(
    os.getenv(
        "MAX_SIGNALS_PER_DAY",
        "20",
    )
)

SIGNAL_COOLDOWN_MINUTES = int(
    os.getenv(
        "SIGNAL_COOLDOWN_MINUTES",
        "20",
    )
)

SEND_STARTUP_MESSAGE = os.getenv(
    "SEND_STARTUP_MESSAGE",
    "true",
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# API protection
API_CONNECT_TIMEOUT = int(
    os.getenv(
        "API_CONNECT_TIMEOUT",
        "10",
    )
)

API_READ_TIMEOUT = int(
    os.getenv(
        "API_READ_TIMEOUT",
        "30",
    )
)

API_RETRIES = int(
    os.getenv(
        "API_RETRIES",
        "3",
    )
)

API_RETRY_DELAY_SECONDS = int(
    os.getenv(
        "API_RETRY_DELAY_SECONDS",
        "5",
    )
)


# ============================================================
# TIMEZONE
# ============================================================

try:

    LOCAL_TIMEZONE = ZoneInfo(
        BOT_TIMEZONE
    )

except Exception:

    LOCAL_TIMEZONE = ZoneInfo(
        "Europe/London"
    )


def now_local() -> datetime:

    return datetime.now(
        LOCAL_TIMEZONE
    )


# ============================================================
# OPERATING WINDOWS
# ============================================================

TREND_WINDOWS = [

    (
        dtime(23, 0),
        dtime(3, 0),
    ),

    (
        dtime(7, 0),
        dtime(17, 30),
    ),
]


SR_WINDOWS = [

    (
        dtime(4, 0),
        dtime(22, 0),
    ),
]


def inside_window(
    current: dtime,
    start: dtime,
    end: dtime,
) -> bool:

    if start <= end:

        return (
            start
            <= current
            <= end
        )

    return (
        current >= start
        or current <= end
    )


# ============================================================
# MARKET HOURS
# ============================================================

def market_closed(
    dt: datetime,
) -> bool:

    weekday = dt.weekday()

    # Friday after 22:00
    if (
        weekday == 4
        and
        dt.time() >= dtime(22, 0)
    ):

        return True

    # Saturday
    if weekday == 5:

        return True

    # Sunday before 22:00
    if (
        weekday == 6
        and
        dt.time() < dtime(22, 0)
    ):

        return True

    return False


def trend_window_active(
    dt: datetime,
) -> bool:

    if market_closed(
        dt
    ):

        return False

    return any(

        inside_window(
            dt.time(),
            start,
            end,
        )

        for start, end
        in TREND_WINDOWS
    )


def sr_window_active(
    dt: datetime,
) -> bool:

    if market_closed(
        dt
    ):

        return False

    return any(

        inside_window(
            dt.time(),
            start,
            end,
        )

        for start, end
        in SR_WINDOWS
    )


# ============================================================
# LOGGING
# ============================================================

def log(
    message: str,
) -> None:

    timestamp = (
        now_local()
        .strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    print(
        f"{timestamp} | {message}",
        flush=True,
    )


# ============================================================
# TWELVE DATA
# ============================================================

SESSION = requests.Session()


def fetch_twelve_data(
    interval: str,
    outputsize: int,
) -> pd.DataFrame:

    if not TWELVE_DATA_API_KEY:

        raise RuntimeError(
            "TWELVE_DATA_API_KEY is missing."
        )

    url = (
        "https://api.twelvedata.com/"
        "time_series"
    )

    params = {

        "symbol":
            SYMBOL,

        "interval":
            interval,

        "outputsize":
            outputsize,

        "apikey":
            TWELVE_DATA_API_KEY,

        "format":
            "JSON",

        "timezone":
            BOT_TIMEZONE,
    }

    response = SESSION.get(
        url,
        params=params,
        timeout=(
            API_CONNECT_TIMEOUT,
            API_READ_TIMEOUT,
        ),
    )

    response.raise_for_status()

    payload = response.json()

    if (
        payload.get(
            "status"
        )
        == "error"
    ):

        raise RuntimeError(
            payload.get(
                "message",
                (
                    "Twelve Data error "
                    f"for {interval}"
                ),
            )
        )

    values = payload.get(
        "values"
    )

    if not values:

        raise RuntimeError(
            (
                "No candle data "
                f"returned for {interval}."
            )
        )

    df = pd.DataFrame(
        values
    )

    required = {

        "datetime",
        "open",
        "high",
        "low",
        "close",
    }

    missing = (
        required
        - set(
            df.columns
        )
    )

    if missing:

        raise RuntimeError(
            (
                f"{interval} missing "
                f"columns: {missing}"
            )
        )

    df["datetime"] = (
        pd.to_datetime(
            df["datetime"],
            errors="coerce",
        )
    )

    for column in [

        "open",
        "high",
        "low",
        "close",

    ]:

        df[column] = (
            pd.to_numeric(
                df[column],
                errors="coerce",
            )
        )

    df = (

        df
        .dropna()
        .sort_values(
            "datetime"
        )
        .drop_duplicates(
            subset=[
                "datetime"
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    if df.empty:

        raise RuntimeError(
            (
                f"{interval} dataframe "
                "is empty after cleaning."
            )
        )

    return df


# ============================================================
# RETRY SYSTEM
# ============================================================

def fetch_with_retry(
    label: str,
    interval: str,
    outputsize: int,
) -> pd.DataFrame:

    last_error = None

    for attempt in range(
        1,
        API_RETRIES + 1,
    ):

        try:

            log(
                (
                    f"FETCH {label} | "
                    f"{interval} | "
                    f"attempt "
                    f"{attempt}/{API_RETRIES}"
                )
            )

            started = (
                time.monotonic()
            )

            df = fetch_twelve_data(
                interval,
                outputsize,
            )

            elapsed = (
                time.monotonic()
                - started
            )

            latest_candle = (
                df.iloc[-1][
                    "datetime"
                ]
            )

            log(
                (
                    f"FETCH {label} OK | "
                    f"{len(df)} candles | "
                    f"latest={latest_candle} | "
                    f"{elapsed:.2f}s"
                )
            )

            return df

        except Exception as exc:

            last_error = exc

            log(
                (
                    f"FETCH {label} FAILED | "
                    f"attempt "
                    f"{attempt}/{API_RETRIES} | "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )
            )

            if (
                attempt
                < API_RETRIES
            ):

                time.sleep(
                    API_RETRY_DELAY_SECONDS
                )

    raise RuntimeError(
        (
            f"{label} failed after "
            f"{API_RETRIES} attempts | "
            f"last error: {last_error}"
        )
    )


# ============================================================
# FETCH ALL TIMEFRAMES
# ============================================================

def fetch_all_candles() -> Dict[
    str,
    pd.DataFrame,
]:

    started = (
        time.monotonic()
    )

    log(
        "MARKET DATA FETCH START"
    )

    candles = {}

    candles["h4"] = (
        fetch_with_retry(
            "H4",
            "4h",
            300,
        )
    )

    candles["h1"] = (
        fetch_with_retry(
            "H1",
            "1h",
            400,
        )
    )

    candles["m15"] = (
        fetch_with_retry(
            "M15",
            "15min",
            450,
        )
    )

    candles["m5"] = (
        fetch_with_retry(
            "M5",
            "5min",
            450,
        )
    )

    elapsed = (
        time.monotonic()
        - started
    )

    log(
        (
            "MARKET DATA FETCH COMPLETE | "
            f"{elapsed:.2f}s"
        )
    )

    return candles


# ============================================================
# SIGNAL DUPLICATE PROTECTION
# ============================================================

def signal_key(
    signal: Signal,
) -> Tuple:

    return (

        signal.setup_type,

        signal.side,

        round(
            signal.reference_level,
            1,
        ),

        round(
            signal.entry_mid,
            1,
        ),
    )


# ============================================================
# COOLDOWN
# ============================================================

def cooldown_finished(
    last_time: Optional[
        datetime
    ],
    current_time: datetime,
) -> bool:

    if last_time is None:

        return True

    elapsed = (
        current_time
        - last_time
    ).total_seconds()

    required = (
        SIGNAL_COOLDOWN_MINUTES
        * 60
    )

    return (
        elapsed >= required
    )


# ============================================================
# STARTUP MESSAGE
# ============================================================

def startup_message() -> str:

    return f"""
🧠⚡ <b>LumiAj-XAU TIMING ENGINE ONLINE</b>
━━━━━━━━━━━━━━━━━━━━━━

<b>Symbol:</b> XAUUSD

<b>Active Engines:</b>
📈 Trend Continuation
🟢 Support Bounce
🔴 Resistance Bounce
🔄 Break & Retest

<b>Timing Protection:</b>
✅ Current-impulse veto
✅ Anti-chase filter
✅ Exhaustion filter
✅ Breakout-vs-bounce veto
✅ Early RSI/MACD turn detection
✅ API retry protection
✅ Data-fetch diagnostics
✅ Worker heartbeat

<b>Max Signals:</b> {MAX_SIGNALS_PER_DAY}/day
<b>Signal Cooldown:</b> {SIGNAL_COOLDOWN_MINUTES} minutes
<b>Scan Interval:</b> {CHECK_INTERVAL_SECONDS}s
<b>Timezone:</b> {BOT_TIMEZONE}

#LUMIAJ_XAU
#LX_SYSTEM
""".strip()


# ============================================================
# MAIN ENGINE
# ============================================================

def main() -> None:

    log(
        "=" * 60
    )

    log(
        "LumiAj-XAU Timing Engine starting"
    )

    log(
        (
            f"Timezone={BOT_TIMEZONE} | "
            f"scan={CHECK_INTERVAL_SECONDS}s | "
            f"daily_cap={MAX_SIGNALS_PER_DAY} | "
            f"cooldown="
            f"{SIGNAL_COOLDOWN_MINUTES}m"
        )
    )

    log(
        "=" * 60
    )

    telegram = (
        TelegramBot()
    )

    model = (
        GerkoXAUModel()
    )

    current_day = (
        now_local().date()
    )

    signals_today = 0

    last_key = None

    last_signal_time = None

    scan_number = 0

    if SEND_STARTUP_MESSAGE:

        try:

            telegram.send(
                startup_message()
            )

        except Exception as exc:

            log(
                (
                    "STARTUP TELEGRAM ERROR | "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )
            )

    # ========================================================
    # CONTINUOUS WORKER LOOP
    # ========================================================

    while True:

        scan_number += 1

        scan_started = (
            time.monotonic()
        )

        now = (
            now_local()
        )

        try:

            log(
                (
                    f"HEARTBEAT | "
                    f"scan={scan_number} | "
                    f"signals="
                    f"{signals_today}/"
                    f"{MAX_SIGNALS_PER_DAY}"
                )
            )

            # --------------------------------------------
            # DAILY RESET
            # --------------------------------------------

            if (
                now.date()
                != current_day
            ):

                current_day = (
                    now.date()
                )

                signals_today = 0

                last_key = None

                last_signal_time = None

                log(
                    "NEW DAY | counters reset"
                )

            # --------------------------------------------
            # MARKET CLOSED
            # --------------------------------------------

            if market_closed(
                now
            ):

                log(
                    (
                        "MARKET CLOSED | "
                        f"weekday="
                        f"{now.strftime('%A')} | "
                        f"time="
                        f"{now.strftime('%H:%M:%S')}"
                    )
                )

                time.sleep(
                    CHECK_INTERVAL_SECONDS
                )

                continue

            # --------------------------------------------
            # DAILY SIGNAL CAP
            # --------------------------------------------

            if (
                signals_today
                >= MAX_SIGNALS_PER_DAY
            ):

                log(
                    (
                        "DAILY SIGNAL CAP | "
                        f"{signals_today}/"
                        f"{MAX_SIGNALS_PER_DAY}"
                    )
                )

                time.sleep(
                    CHECK_INTERVAL_SECONDS
                )

                continue

            # --------------------------------------------
            # ACTIVE WINDOWS
            # --------------------------------------------

            trend_active = (
                trend_window_active(
                    now
                )
            )

            sr_active = (
                sr_window_active(
                    now
                )
            )

            log(
                (
                    "WINDOW STATUS | "
                    f"trend={trend_active} | "
                    f"sr={sr_active}"
                )
            )

            if (
                not trend_active
                and
                not sr_active
            ):

                log(
                    "OUTSIDE ACTIVE SCAN WINDOWS"
                )

                time.sleep(
                    CHECK_INTERVAL_SECONDS
                )

                continue

            # --------------------------------------------
            # MARKET DATA
            # --------------------------------------------

            log(
                "SCAN START | fetching XAUUSD"
            )

            candles = (
                fetch_all_candles()
            )

            # --------------------------------------------
            # ANALYSIS
            # --------------------------------------------

            log(
                "ANALYSIS START"
            )

            analysis_started = (
                time.monotonic()
            )

            signal = model.analyse(

                candles=candles,

                signal_time=now,

                allow_trend=(
                    trend_active
                ),

                allow_sr=(
                    sr_active
                ),
            )

            analysis_elapsed = (
                time.monotonic()
                - analysis_started
            )

            log(
                (
                    "ANALYSIS COMPLETE | "
                    f"{analysis_elapsed:.2f}s"
                )
            )

            # --------------------------------------------
            # NO SIGNAL
            # --------------------------------------------

            if signal is None:

                log(
                    "NO QUALIFIED LUMIAJ SETUP"
                )

            else:

                log(
                    (
                        "CANDIDATE FOUND | "
                        f"{signal.setup_type} | "
                        f"{signal.side} | "
                        f"score={signal.score} | "
                        f"id={signal.signal_id}"
                    )
                )

                key = (
                    signal_key(
                        signal
                    )
                )

                # ----------------------------------------
                # DUPLICATE
                # ----------------------------------------

                if (
                    key
                    == last_key
                ):

                    log(
                        (
                            "SIGNAL REJECTED | "
                            "duplicate"
                        )
                    )

                # ----------------------------------------
                # COOLDOWN
                # ----------------------------------------

                elif not cooldown_finished(
                    last_signal_time,
                    now,
                ):

                    if (
                        last_signal_time
                        is not None
                    ):

                        elapsed_minutes = (

                            (
                                now
                                - last_signal_time
                            )
                            .total_seconds()
                            / 60
                        )

                    else:

                        elapsed_minutes = 0

                    log(
                        (
                            "SIGNAL REJECTED | "
                            "cooldown active | "
                            f"{elapsed_minutes:.1f}/"
                            f"{SIGNAL_COOLDOWN_MINUTES}m"
                        )
                    )

                # ----------------------------------------
                # SEND SIGNAL
                # ----------------------------------------

                else:

                    message = (
                        format_signal(
                            signal
                        )
                    )

                    log(
                        (
                            "TELEGRAM SEND START | "
                            f"{signal.signal_id}"
                        )
                    )

                    send_started = (
                        time.monotonic()
                    )

                    success = (
                        telegram.send(
                            message
                        )
                    )

                    send_elapsed = (
                        time.monotonic()
                        - send_started
                    )

                    if success:

                        signals_today += 1

                        last_key = key

                        last_signal_time = now

                        log(
                            (
                                "SIGNAL SENT | "
                                f"{signal.signal_id} | "
                                f"{signal.side} | "
                                f"score={signal.score} | "
                                f"signals_today="
                                f"{signals_today}/"
                                f"{MAX_SIGNALS_PER_DAY} | "
                                f"telegram="
                                f"{send_elapsed:.2f}s"
                            )
                        )

                    else:

                        log(
                            (
                                "TELEGRAM DELIVERY "
                                "FAILED | "
                                f"{signal.signal_id}"
                            )
                        )

        # ====================================================
        # MANUAL STOP
        # ====================================================

        except KeyboardInterrupt:

            log(
                "LumiAj-XAU manually stopped."
            )

            break

        # ====================================================
        # NETWORK FAILURE
        # ====================================================

        except requests.Timeout as exc:

            log(
                (
                    "REQUEST TIMEOUT | "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )
            )

        except requests.RequestException as exc:

            log(
                (
                    "NETWORK ERROR | "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )
            )

        # ====================================================
        # ANY OTHER FAILURE
        # ====================================================

        except Exception as exc:

            log(
                (
                    "SCAN ERROR | "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )
            )

        # ====================================================
        # SCAN FINISHED
        # ====================================================

        finally:

            elapsed = (
                time.monotonic()
                - scan_started
            )

            log(
                (
                    f"SCAN COMPLETE | "
                    f"scan={scan_number} | "
                    f"elapsed={elapsed:.2f}s"
                )
            )

        # ====================================================
        # NEXT SCAN
        # ====================================================

        log(
            (
                "NEXT SCAN | "
                f"{CHECK_INTERVAL_SECONDS}s"
            )
        )

        time.sleep(
            CHECK_INTERVAL_SECONDS
        )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    while True:

        try:

            main()

            break

        except KeyboardInterrupt:

            break

        except Exception as exc:

            print(
                (
                    "FATAL WORKER ERROR | "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
                flush=True,
            )

            print(
                (
                    "Restarting LumiAj engine "
                    "in 15 seconds..."
                ),
                flush=True,
            )

            time.sleep(
                15
            )