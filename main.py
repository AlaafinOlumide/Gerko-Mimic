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


load_dotenv()


TWELVE_DATA_API_KEY = os.getenv(
    "TWELVE_DATA_API_KEY",
    "",
).strip()

SYMBOL = os.getenv(
    "SYMBOL",
    "XAU/USD",
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

BOT_TIMEZONE = os.getenv(
    "BOT_TIMEZONE",
    "Europe/London",
).strip()

SEND_STARTUP_MESSAGE = os.getenv(
    "SEND_STARTUP_MESSAGE",
    "true",
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


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


def market_closed(
    dt: datetime,
) -> bool:

    weekday = dt.weekday()

    if (
        weekday == 4
        and
        dt.time()
        >= dtime(22, 0)
    ):
        return True

    if weekday == 5:
        return True

    if (
        weekday == 6
        and
        dt.time()
        < dtime(22, 0)
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

    response = requests.get(
        url,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    payload = response.json()

    if (
        payload.get("status")
        == "error"
    ):

        raise RuntimeError(
            payload.get(
                "message",
                f"Twelve Data error on {interval}",
            )
        )

    values = payload.get(
        "values"
    )

    if not values:

        raise RuntimeError(
            f"No {interval} candles returned."
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
            f"Missing columns: {missing}"
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

    return (
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


def fetch_all_candles() -> Dict[
    str,
    pd.DataFrame,
]:

    return {

        "h4":
            fetch_twelve_data(
                "4h",
                300,
            ),

        "h1":
            fetch_twelve_data(
                "1h",
                400,
            ),

        "m15":
            fetch_twelve_data(
                "15min",
                450,
            ),

        "m5":
            fetch_twelve_data(
                "5min",
                450,
            ),
    }


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


def cooldown_finished(
    last_time: Optional[
        datetime
    ],
    current_time: datetime,
) -> bool:

    if last_time is None:
        return True

    return (
        current_time
        - last_time
    ).total_seconds() >= (
        SIGNAL_COOLDOWN_MINUTES
        * 60
    )


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

<b>Max Signals:</b> {MAX_SIGNALS_PER_DAY}/day
<b>Scan Interval:</b> {CHECK_INTERVAL_SECONDS}s
<b>Timezone:</b> {BOT_TIMEZONE}

#LUMIAJ_XAU
#LX_SYSTEM
""".strip()


def main() -> None:

    print(
        "=" * 60
    )

    print(
        "LumiAj-XAU Timing Engine started"
    )

    print(
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

    if SEND_STARTUP_MESSAGE:

        telegram.send(
            startup_message()
        )

    while True:

        now = now_local()

        try:

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

                print(
                    "Daily counters reset."
                )

            if market_closed(
                now
            ):

                print(
                    f"{now.isoformat()} "
                    "- market closed."
                )

                time.sleep(
                    CHECK_INTERVAL_SECONDS
                )

                continue

            if (
                signals_today
                >= MAX_SIGNALS_PER_DAY
            ):

                print(
                    "Daily signal cap reached."
                )

                time.sleep(
                    CHECK_INTERVAL_SECONDS
                )

                continue

            if (
                not trend_window_active(
                    now
                )
                and
                not sr_window_active(
                    now
                )
            ):

                print(
                    f"{now.isoformat()} "
                    "- outside active scan windows."
                )

                time.sleep(
                    CHECK_INTERVAL_SECONDS
                )

                continue

            print(
                f"{now.isoformat()} "
                "- fetching XAUUSD..."
            )

            candles = (
                fetch_all_candles()
            )

            signal = model.analyse(
                candles=candles,
                signal_time=now,

                allow_trend=(
                    trend_window_active(
                        now
                    )
                ),

                allow_sr=(
                    sr_window_active(
                        now
                    )
                ),
            )

            if signal is None:

                print(
                    "No qualified LumiAj setup."
                )

                time.sleep(
                    CHECK_INTERVAL_SECONDS
                )

                continue

            key = signal_key(
                signal
            )

            if key == last_key:

                print(
                    "Duplicate signal ignored."
                )

                time.sleep(
                    CHECK_INTERVAL_SECONDS
                )

                continue

            if not cooldown_finished(
                last_signal_time,
                now,
            ):

                print(
                    "Signal cooldown active."
                )

                time.sleep(
                    CHECK_INTERVAL_SECONDS
                )

                continue

            message = format_signal(
                signal
            )

            success = telegram.send(
                message
            )

            if success:

                signals_today += 1

                last_key = key

                last_signal_time = now

                print(
                    f"LumiAj signal sent: "
                    f"{signal.signal_id}"
                )

            else:

                print(
                    "Signal generated but Telegram "
                    "delivery failed."
                )

        except KeyboardInterrupt:

            print(
                "LumiAj-XAU stopped."
            )

            break

        except requests.RequestException as exc:

            print(
                "NETWORK ERROR:",
                exc,
            )

        except Exception as exc:

            print(
                "LUMIAJ ERROR:",
                type(exc).__name__,
                exc,
            )

        time.sleep(
            CHECK_INTERVAL_SECONDS
        )


if __name__ == "__main__":
    main()