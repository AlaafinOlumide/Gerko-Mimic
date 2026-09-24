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
        "2",
    )
)

API_RETRY_DELAY_SECONDS = int(
    os.getenv(
        "API_RETRY_DELAY_SECONDS",
        "5",
    )
)

# Default backoff when Twelve Data returns 429 and
# does not supply Retry-After.
RATE_LIMIT_BACKOFF_SECONDS = int(
    os.getenv(
        "RATE_LIMIT_BACKOFF_SECONDS",
        "300",
    )
)


# ============================================================
# TIMEFRAME REFRESH SETTINGS
# ============================================================

# M5 remains responsive.
# Higher timeframes are cached because downloading them
# every 3 minutes wastes API credits.

TIMEFRAME_CONFIG = {
    "h4": {
        "label": "H4",
        "interval": "4h",
        "outputsize": 300,
        "refresh_seconds": 3600,
        "max_stale_seconds": 7200,
    },
    "h1": {
        "label": "H1",
        "interval": "1h",
        "outputsize": 400,
        "refresh_seconds": 1800,
        "max_stale_seconds": 3600,
    },
    "m15": {
        "label": "M15",
        "interval": "15min",
        "outputsize": 450,
        "refresh_seconds": 360,
        "max_stale_seconds": 1200,
    },
    "m5": {
        "label": "M5",
        "interval": "5min",
        "outputsize": 450,
        "refresh_seconds": CHECK_INTERVAL_SECONDS,
        "max_stale_seconds": 600,
    },
}


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
# LOGGING
# ============================================================

def log(message: str) -> None:

    timestamp = now_local().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    print(
        f"{timestamp} | {message}",
        flush=True,
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
        return start <= current <= end

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

    # Friday after 22:00 London
    if (
        weekday == 4
        and dt.time() >= dtime(22, 0)
    ):
        return True

    # Saturday
    if weekday == 5:
        return True

    # Sunday before 22:00 London
    if (
        weekday == 6
        and dt.time() < dtime(22, 0)
    ):
        return True

    return False


def trend_window_active(
    dt: datetime,
) -> bool:

    if market_closed(dt):
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

    if market_closed(dt):
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
# MARKET DATA CACHE
# ============================================================

DATA_CACHE: Dict[
    str,
    pd.DataFrame,
] = {}

LAST_FETCH_MONOTONIC: Dict[
    str,
    float,
] = {}

LAST_FETCH_LOCAL: Dict[
    str,
    datetime,
] = {}

RATE_LIMITED_UNTIL: float = 0.0


# ============================================================
# HTTP SESSION
# ============================================================

SESSION = requests.Session()


# ============================================================
# TWELVE DATA REQUEST
# ============================================================

class TwelveDataRateLimitError(
    RuntimeError
):
    def __init__(
        self,
        message: str,
        retry_after: int,
    ):
        super().__init__(
            message
        )

        self.retry_after = (
            retry_after
        )


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
        "symbol": SYMBOL,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVE_DATA_API_KEY,
        "format": "JSON",
        "timezone": BOT_TIMEZONE,
    }

    response = SESSION.get(
        url,
        params=params,
        timeout=(
            API_CONNECT_TIMEOUT,
            API_READ_TIMEOUT,
        ),
    )

    # --------------------------------------------
    # RATE LIMIT
    # --------------------------------------------

    if response.status_code == 429:

        retry_after_raw = (
            response.headers.get(
                "Retry-After"
            )
        )

        try:
            retry_after = int(
                retry_after_raw
            )
        except (
            TypeError,
            ValueError,
        ):
            retry_after = (
                RATE_LIMIT_BACKOFF_SECONDS
            )

        raise TwelveDataRateLimitError(
            (
                "Twelve Data returned "
                "HTTP 429 Too Many Requests"
            ),
            retry_after,
        )

    response.raise_for_status()

    payload = response.json()

    # Twelve Data can sometimes return an error
    # payload even where HTTP handling differs.
    if payload.get("status") == "error":

        message = payload.get(
            "message",
            "Unknown Twelve Data error",
        )

        code = payload.get(
            "code"
        )

        if code == 429:

            raise TwelveDataRateLimitError(
                message,
                RATE_LIMIT_BACKOFF_SECONDS,
            )

        raise RuntimeError(
            message
        )

    values = payload.get(
        "values"
    )

    if not values:
        raise RuntimeError(
            (
                "No candle data returned "
                f"for {interval}."
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
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            (
                f"{interval} missing "
                f"columns: {missing}"
            )
        )

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce",
    )

    for column in [
        "open",
        "high",
        "low",
        "close",
    ]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = (
        df
        .dropna()
        .sort_values(
            "datetime"
        )
        .drop_duplicates(
            subset=["datetime"],
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
                "empty after cleaning."
            )
        )

    return df


# ============================================================
# CACHE HELPERS
# ============================================================

def cache_exists(
    timeframe: str,
) -> bool:

    return (
        timeframe in DATA_CACHE
        and
        not DATA_CACHE[
            timeframe
        ].empty
    )


def cache_age_seconds(
    timeframe: str,
) -> Optional[float]:

    last_fetch = (
        LAST_FETCH_MONOTONIC.get(
            timeframe
        )
    )

    if last_fetch is None:
        return None

    return (
        time.monotonic()
        - last_fetch
    )


def cache_needs_refresh(
    timeframe: str,
) -> bool:

    if not cache_exists(
        timeframe
    ):
        return True

    age = cache_age_seconds(
        timeframe
    )

    if age is None:
        return True

    refresh_seconds = (
        TIMEFRAME_CONFIG[
            timeframe
        ][
            "refresh_seconds"
        ]
    )

    return (
        age >= refresh_seconds
    )


def cache_is_usable(
    timeframe: str,
) -> bool:

    if not cache_exists(
        timeframe
    ):
        return False

    age = cache_age_seconds(
        timeframe
    )

    if age is None:
        return False

    max_stale = (
        TIMEFRAME_CONFIG[
            timeframe
        ][
            "max_stale_seconds"
        ]
    )

    return (
        age <= max_stale
    )


# ============================================================
# RATE LIMIT HELPERS
# ============================================================

def rate_limit_active() -> bool:

    return (
        time.monotonic()
        < RATE_LIMITED_UNTIL
    )


def rate_limit_remaining() -> int:

    remaining = (
        RATE_LIMITED_UNTIL
        - time.monotonic()
    )

    return max(
        0,
        int(remaining),
    )


def activate_rate_limit(
    seconds: int,
) -> None:

    global RATE_LIMITED_UNTIL

    seconds = max(
        30,
        int(seconds),
    )

    new_until = (
        time.monotonic()
        + seconds
    )

    RATE_LIMITED_UNTIL = max(
        RATE_LIMITED_UNTIL,
        new_until,
    )

    log(
        (
            "API BACKOFF ACTIVATED | "
            f"{seconds}s"
        )
    )


# ============================================================
# SMART FETCH
# ============================================================

def refresh_timeframe(
    timeframe: str,
) -> bool:

    config = (
        TIMEFRAME_CONFIG[
            timeframe
        ]
    )

    label = config[
        "label"
    ]

    interval = config[
        "interval"
    ]

    outputsize = config[
        "outputsize"
    ]

    # --------------------------------------------
    # GLOBAL 429 BACKOFF
    # --------------------------------------------

    if rate_limit_active():

        remaining = (
            rate_limit_remaining()
        )

        log(
            (
                f"FETCH {label} SKIPPED | "
                "API backoff active | "
                f"{remaining}s remaining"
            )
        )

        return False

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
                    f"{attempt}/"
                    f"{API_RETRIES}"
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

            DATA_CACHE[
                timeframe
            ] = df

            LAST_FETCH_MONOTONIC[
                timeframe
            ] = time.monotonic()

            LAST_FETCH_LOCAL[
                timeframe
            ] = now_local()

            latest_candle = (
                df.iloc[-1][
                    "datetime"
                ]
            )

            log(
                (
                    f"FETCH {label} OK | "
                    f"{len(df)} candles | "
                    f"latest="
                    f"{latest_candle} | "
                    f"{elapsed:.2f}s"
                )
            )

            return True

        except TwelveDataRateLimitError as exc:

            last_error = exc

            log(
                (
                    f"FETCH {label} RATE LIMITED | "
                    f"HTTP 429 | "
                    f"backoff="
                    f"{exc.retry_after}s"
                )
            )

            activate_rate_limit(
                exc.retry_after
            )

            # DO NOT retry a 429 immediately.
            return False

        except requests.Timeout as exc:

            last_error = exc

            log(
                (
                    f"FETCH {label} TIMEOUT | "
                    f"attempt "
                    f"{attempt}/"
                    f"{API_RETRIES} | "
                    f"{exc}"
                )
            )

        except requests.RequestException as exc:

            last_error = exc

            log(
                (
                    f"FETCH {label} NETWORK ERROR | "
                    f"attempt "
                    f"{attempt}/"
                    f"{API_RETRIES} | "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )
            )

        except Exception as exc:

            last_error = exc

            log(
                (
                    f"FETCH {label} ERROR | "
                    f"attempt "
                    f"{attempt}/"
                    f"{API_RETRIES} | "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )
            )

        if attempt < API_RETRIES:

            time.sleep(
                API_RETRY_DELAY_SECONDS
            )

    log(
        (
            f"FETCH {label} FAILED | "
            f"using cache if safe | "
            f"last_error={last_error}"
        )
    )

    return False


# ============================================================
# SMART MARKET DATA MANAGER
# ============================================================

def get_market_data() -> Optional[
    Dict[str, pd.DataFrame]
]:

    log(
        "MARKET DATA CHECK START"
    )

    # Refresh slowest first during initialization.
    refresh_order = [
        "h4",
        "h1",
        "m15",
        "m5",
    ]

    for timeframe in refresh_order:

        config = (
            TIMEFRAME_CONFIG[
                timeframe
            ]
        )

        label = config[
            "label"
        ]

        if cache_needs_refresh(
            timeframe
        ):

            age = cache_age_seconds(
                timeframe
            )

            if age is None:

                log(
                    (
                        f"CACHE {label} | "
                        "empty | refresh required"
                    )
                )

            else:

                log(
                    (
                        f"CACHE {label} | "
                        f"age={age:.0f}s | "
                        "refresh required"
                    )
                )

            refresh_timeframe(
                timeframe
            )

        else:

            age = (
                cache_age_seconds(
                    timeframe
                )
                or 0
            )

            log(
                (
                    f"CACHE {label} HIT | "
                    f"age={age:.0f}s"
                )
            )

    # --------------------------------------------
    # VERIFY ALL REQUIRED DATA
    # --------------------------------------------

    unusable = []

    for timeframe in [
        "h4",
        "h1",
        "m15",
        "m5",
    ]:

        config = (
            TIMEFRAME_CONFIG[
                timeframe
            ]
        )

        label = config[
            "label"
        ]

        if not cache_is_usable(
            timeframe
        ):

            unusable.append(
                label
            )

    if unusable:

        log(
            (
                "MARKET DATA NOT SAFE | "
                "missing/stale="
                + ",".join(
                    unusable
                )
            )
        )

        return None

    log(
        "MARKET DATA READY"
    )

    return {
        "h4": DATA_CACHE["h4"],
        "h1": DATA_CACHE["h1"],
        "m15": DATA_CACHE["m15"],
        "m5": DATA_CACHE["m5"],
    }


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
🧠⚡ <b>LumiAj-XAU SMART DATA ENGINE ONLINE</b>
━━━━━━━━━━━━━━━━━━━━━━

<b>Symbol:</b> XAUUSD

<b>Active Engines:</b>
📈 Trend Continuation
🟢 Support Bounce
🔴 Resistance Bounce
🔄 Break & Retest

<b>Data Engine:</b>
M5: 3-minute refresh
M15: 6-minute cache
H1: 30-minute cache
H4: 60-minute cache

<b>Protection:</b>
✅ Current-impulse veto
✅ Anti-chase filter
✅ Exhaustion filter
✅ Breakout-vs-bounce veto
✅ Smart timeframe caching
✅ HTTP 429 backoff
✅ Stale-data protection
✅ Worker heartbeat

<b>Max Signals:</b> {MAX_SIGNALS_PER_DAY}/day
<b>Cooldown:</b> {SIGNAL_COOLDOWN_MINUTES} minutes
<b>Scan:</b> {CHECK_INTERVAL_SECONDS}s
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
        "LumiAj-XAU Smart Data Engine starting"
    )

    log(
        (
            f"Timezone={BOT_TIMEZONE} | "
            f"scan={CHECK_INTERVAL_SECONDS}s | "
            f"daily_cap="
            f"{MAX_SIGNALS_PER_DAY} | "
            f"cooldown="
            f"{SIGNAL_COOLDOWN_MINUTES}m"
        )
    )

    log(
        (
            "CACHE PLAN | "
            "M5=3m | "
            "M15=6m | "
            "H1=30m | "
            "H4=60m"
        )
    )

    log(
        "=" * 60
    )

    telegram = TelegramBot()

    model = GerkoXAUModel()

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
    # CONTINUOUS WORKER
    # ========================================================

    while True:

        scan_number += 1

        scan_started = (
            time.monotonic()
        )

        now = now_local()

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
            # MARKET HOURS
            # --------------------------------------------

            if market_closed(now):

                log(
                    (
                        "MARKET CLOSED | "
                        f"{now.strftime('%A')} | "
                        f"{now.strftime('%H:%M:%S')}"
                    )
                )

                time.sleep(
                    CHECK_INTERVAL_SECONDS
                )

                continue

            # --------------------------------------------
            # DAILY CAP
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
            # WINDOWS
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
            # API BACKOFF STATUS
            # --------------------------------------------

            if rate_limit_active():

                log(
                    (
                        "API RATE-LIMIT BACKOFF | "
                        f"{rate_limit_remaining()}s "
                        "remaining"
                    )
                )

            # --------------------------------------------
            # MARKET DATA
            # --------------------------------------------

            candles = (
                get_market_data()
            )

            if candles is None:

                log(
                    (
                        "SCAN SKIPPED | "
                        "market data unavailable "
                        "or stale"
                    )
                )

            else:

                # ----------------------------------------
                # ANALYSIS
                # ----------------------------------------

                log(
                    "ANALYSIS START"
                )

                analysis_started = (
                    time.monotonic()
                )

                signal = model.analyse(
                    candles=candles,
                    signal_time=now,
                    allow_trend=trend_active,
                    allow_sr=sr_active,
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

                # ----------------------------------------
                # NO SIGNAL
                # ----------------------------------------

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
                            f"score="
                            f"{signal.score} | "
                            f"id="
                            f"{signal.signal_id}"
                        )
                    )

                    key = (
                        signal_key(
                            signal
                        )
                    )

                    # ------------------------------------
                    # DUPLICATE
                    # ------------------------------------

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

                    # ------------------------------------
                    # COOLDOWN
                    # ------------------------------------

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
                                "cooldown | "
                                f"{elapsed_minutes:.1f}/"
                                f"{SIGNAL_COOLDOWN_MINUTES}m"
                            )
                        )

                    # ------------------------------------
                    # SEND
                    # ------------------------------------

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

                        success = (
                            telegram.send(
                                message
                            )
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
                                    f"score="
                                    f"{signal.score} | "
                                    f"signals_today="
                                    f"{signals_today}/"
                                    f"{MAX_SIGNALS_PER_DAY}"
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
        # UNEXPECTED SCAN ERROR
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
        # SCAN COMPLETE
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
# PROCESS WATCHDOG
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