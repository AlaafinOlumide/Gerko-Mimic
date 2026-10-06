from __future__ import annotations

import math
import os
import secrets
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from risk_engine import (
    calculate_rr,
    classify_signal,
)

from sr_engine import (
    build_levels,
    combine_levels,
    detect_break_retest,
    detect_resistance_bounce,
    detect_support_bounce,
)


# ============================================================
# THRESHOLDS
# ============================================================

TREND_MIN_SCORE = int(
    os.getenv(
        "TREND_MIN_SCORE",
        "76",
    )
)

SR_MIN_SCORE = int(
    os.getenv(
        "SR_MIN_SCORE",
        "74",
    )
)

COUNTERTREND_SR_MIN_SCORE = int(
    os.getenv(
        "COUNTERTREND_SR_MIN_SCORE",
        "84",
    )
)

BREAK_RETEST_MIN_SCORE = int(
    os.getenv(
        "BREAK_RETEST_MIN_SCORE",
        "76",
    )
)

MAX_TREND_CHASE_ATR = float(
    os.getenv(
        "MAX_TREND_CHASE_ATR",
        "0.65",
    )
)

MAX_SR_CHASE_ATR = float(
    os.getenv(
        "MAX_SR_CHASE_ATR",
        "0.45",
    )
)


# ============================================================
# RANGE ENGINE SETTINGS
# ============================================================

RANGE_MIN_SCORE = int(
    os.getenv(
        "RANGE_MIN_SCORE",
        "76",
    )
)

RANGE_ADX_MAX = float(
    os.getenv(
        "RANGE_ADX_MAX",
        "20",
    )
)

RANGE_EDGE_ATR = float(
    os.getenv(
        "RANGE_EDGE_ATR",
        "0.35",
    )
)

RANGE_MIN_WIDTH_ATR = float(
    os.getenv(
        "RANGE_MIN_WIDTH_ATR",
        "2.0",
    )
)

RANGE_MAX_WIDTH_ATR = float(
    os.getenv(
        "RANGE_MAX_WIDTH_ATR",
        "8.0",
    )
)

RANGE_BREAK_BUFFER_ATR = float(
    os.getenv(
        "RANGE_BREAK_BUFFER_ATR",
        "0.12",
    )
)


# ============================================================
# SIGNAL
# ============================================================

@dataclass
class Signal:

    signal_id: str

    created_at: datetime

    symbol: str

    setup_type: str

    setup_name: str

    side: str

    score: int

    grade: str

    quality: str

    trend_context: str

    impulse: str

    reference_level: float

    level_strength: int

    level_touches: int

    entry_low: float

    entry_high: float

    entry_mid: float

    stop_loss: float

    tp1: float

    tp2: float

    tp3: float

    rr: float

    lot_size: float

    reasons: List[str]

    invalidation: str


# ============================================================
# INDICATORS
# ============================================================

def ema(
    series: pd.Series,
    period: int,
) -> pd.Series:

    return series.ewm(
        span=period,
        adjust=False,
    ).mean()


def rsi(
    series: pd.Series,
    period: int = 14,
) -> pd.Series:

    delta = series.diff()

    gains = delta.clip(
        lower=0
    )

    losses = -delta.clip(
        upper=0
    )

    avg_gain = gains.ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()

    avg_loss = losses.ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        math.nan,
    )

    return 100 - 100 / (
        1 + rs
    )


def true_range(
    df: pd.DataFrame,
) -> pd.Series:

    return pd.concat(
        [
            df["high"] - df["low"],

            (
                df["high"]
                - df["close"].shift(1)
            ).abs(),

            (
                df["low"]
                - df["close"].shift(1)
            ).abs(),
        ],
        axis=1,
    ).max(
        axis=1
    )


def atr(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.Series:

    return true_range(
        df
    ).ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()


def macd_histogram(
    series: pd.Series,
) -> pd.Series:

    fast = ema(
        series,
        12,
    )

    slow = ema(
        series,
        26,
    )

    macd = fast - slow

    signal = macd.ewm(
        span=9,
        adjust=False,
    ).mean()

    return macd - signal


def adx(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.Series:

    up = df["high"].diff()

    down = -df["low"].diff()

    plus_dm = up.where(
        (up > down)
        & (up > 0),
        0.0,
    )

    minus_dm = down.where(
        (down > up)
        & (down > 0),
        0.0,
    )

    atr_series = (
        atr(
            df,
            period,
        )
        .replace(
            0,
            math.nan,
        )
    )

    plus_di = (
        100
        * plus_dm.ewm(
            alpha=1 / period,
            adjust=False,
        ).mean()
        / atr_series
    )

    minus_di = (
        100
        * minus_dm.ewm(
            alpha=1 / period,
            adjust=False,
        ).mean()
        / atr_series
    )

    denominator = (
        plus_di
        + minus_di
    ).replace(
        0,
        math.nan,
    )

    dx = (
        (
            plus_di
            - minus_di
        ).abs()
        / denominator
    ) * 100

    return dx.ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()


# ============================================================
# ENRICH DATA
# ============================================================

def enrich(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

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

    df["ema20"] = ema(
        df["close"],
        20,
    )

    df["ema50"] = ema(
        df["close"],
        50,
    )

    df["ema200"] = ema(
        df["close"],
        200,
    )

    df["rsi"] = rsi(
        df["close"]
    )

    df["atr"] = atr(
        df
    )

    df["adx"] = adx(
        df
    )

    df["macd_histogram"] = (
        macd_histogram(
            df["close"]
        )
    )

    df["ema50_slope"] = (
        df["ema50"].diff(5)
    )

    return (
        df
        .dropna()
        .reset_index(
            drop=True
        )
    )


# ============================================================
# CURRENT M5 IMPULSE
# ============================================================

def current_impulse(
    m5: pd.DataFrame,
) -> str:

    latest = m5.iloc[-1]

    previous = m5.iloc[-2]

    third = m5.iloc[-3]

    bullish = 0

    bearish = 0

    if (
        float(latest["close"])
        > float(previous["high"])
    ):

        bullish += 2

    if (
        float(latest["close"])
        < float(previous["low"])
    ):

        bearish += 2

    if (
        float(latest["close"])
        > float(latest["open"])
    ):

        bullish += 1

    elif (
        float(latest["close"])
        < float(latest["open"])
    ):

        bearish += 1

    if (
        float(latest["close"])
        > float(latest["ema20"])
    ):

        bullish += 1

    else:

        bearish += 1

    if (
        float(latest["rsi"])
        > 52
    ):

        bullish += 1

    elif (
        float(latest["rsi"])
        < 48
    ):

        bearish += 1

    if (
        float(latest["macd_histogram"])
        > float(previous["macd_histogram"])
    ):

        bullish += 1

    elif (
        float(latest["macd_histogram"])
        < float(previous["macd_histogram"])
    ):

        bearish += 1

    if (
        float(latest["close"])
        > float(previous["close"])
        > float(third["close"])
    ):

        bullish += 2

    elif (
        float(latest["close"])
        < float(previous["close"])
        < float(third["close"])
    ):

        bearish += 2

    if bullish >= bearish + 2:

        return "BUY"

    if bearish >= bullish + 2:

        return "SELL"

    return "NEUTRAL"


# ============================================================
# EXHAUSTION / CHASE PROTECTION
# ============================================================

def candle_exhausted(
    m5: pd.DataFrame,
    m15_atr: float,
) -> bool:

    latest = m5.iloc[-1]

    candle_range = float(
        latest["high"]
        - latest["low"]
    )

    return (
        candle_range
        > m15_atr * 0.90
    )


def overextended_from_value(
    price: float,
    ema20_value: float,
    atr_value: float,
) -> bool:

    return (
        abs(
            price
            - ema20_value
        )
        > atr_value * 0.55
    )


def move_exhausted(
    price: float,
    origin: float,
    atr_value: float,
    max_atr: float,
) -> bool:

    return (
        abs(
            price
            - origin
        )
        > atr_value * max_atr
    )


# ============================================================
# RANGE REGIME DETECTION
# ============================================================

def range_regime(
    m15: pd.DataFrame,
    h1: pd.DataFrame,
    atr_value: float,
) -> Optional[
    Dict[str, float]
]:

    if (
        len(m15) < 60
        or len(h1) < 60
        or atr_value <= 0
    ):

        return None

    # Last 48 M15 candles = approximately 12 hours.
    recent = m15.tail(
        48
    )

    range_high = float(
        recent["high"].max()
    )

    range_low = float(
        recent["low"].min()
    )

    width = (
        range_high
        - range_low
    )

    width_atr = (
        width
        / atr_value
    )

    h1_last = h1.iloc[-1]

    m15_last = m15.iloc[-1]

    h1_adx = float(
        h1_last["adx"]
    )

    m15_adx = float(
        m15_last["adx"]
    )

    h1_slope = abs(
        float(
            h1_last[
                "ema50_slope"
            ]
        )
    )

    m15_slope = abs(
        float(
            m15_last[
                "ema50_slope"
            ]
        )
    )

    # ----------------------------------------
    # RANGE WIDTH
    # ----------------------------------------

    if not (
        RANGE_MIN_WIDTH_ATR
        <= width_atr
        <= RANGE_MAX_WIDTH_ATR
    ):

        return None

    # ----------------------------------------
    # TREND STRENGTH
    # ----------------------------------------

    if (
        h1_adx
        > RANGE_ADX_MAX
        or
        m15_adx
        > RANGE_ADX_MAX + 3
    ):

        return None

    # ----------------------------------------
    # EMA FLATNESS
    # ----------------------------------------

    if (
        h1_slope
        > atr_value * 0.35
    ):

        return None

    if (
        m15_slope
        > atr_value * 0.45
    ):

        return None

    price = float(
        m15_last["close"]
    )

    buffer = (
        atr_value
        * RANGE_BREAK_BUFFER_ATR
    )

    # Already outside the range:
    # this is no longer a range-reversal environment.
    if (
        price
        > range_high + buffer
        or
        price
        < range_low - buffer
    ):

        return None

    return {
        "high": range_high,
        "low": range_low,
        "width": width,
        "width_atr": width_atr,
        "h1_adx": h1_adx,
        "m15_adx": m15_adx,
    }


# ============================================================
# RANGE BREAK DETECTION
# ============================================================

def range_breaking(
    m15: pd.DataFrame,
    range_info: Dict[str, float],
    atr_value: float,
) -> bool:

    if len(m15) < 3:

        return False

    latest = m15.iloc[-1]

    previous = m15.iloc[-2]

    buffer = (
        atr_value
        * RANGE_BREAK_BUFFER_ATR
    )

    high = range_info[
        "high"
    ]

    low = range_info[
        "low"
    ]

    bullish_break = (

        float(latest["close"])
        > high + buffer

        and

        float(previous["close"])
        > high

        and

        float(latest["macd_histogram"])
        > float(
            previous[
                "macd_histogram"
            ]
        )
    )

    bearish_break = (

        float(latest["close"])
        < low - buffer

        and

        float(previous["close"])
        < low

        and

        float(latest["macd_histogram"])
        < float(
            previous[
                "macd_histogram"
            ]
        )
    )

    return (
        bullish_break
        or bearish_break
    )


# ============================================================
# RANGE EDGE SIGNAL
# ============================================================

def range_signal(
    m15: pd.DataFrame,
    m5: pd.DataFrame,
    range_info: Dict[str, float],
    atr_value: float,
) -> Optional[dict]:

    price = float(
        m5.iloc[-1]["close"]
    )

    latest = m5.iloc[-1]

    previous = m5.iloc[-2]

    high = range_info[
        "high"
    ]

    low = range_info[
        "low"
    ]

    edge_distance = (
        atr_value
        * RANGE_EDGE_ATR
    )

    near_support = (
        abs(
            price
            - low
        )
        <= edge_distance
    )

    near_resistance = (
        abs(
            price
            - high
        )
        <= edge_distance
    )

    # ========================================================
    # MIDDLE OF RANGE = NO TRADE
    # ========================================================

    if (
        not near_support
        and
        not near_resistance
    ):

        return None

    if high <= low:

        return None

    candle_range = float(
        latest["high"]
        - latest["low"]
    )

    if candle_range <= 0:

        return None

    bullish_rejection = (

        float(latest["close"])
        > float(latest["open"])

        and

        (
            float(latest["close"])
            - float(latest["low"])
        )
        >= candle_range * 0.55
    )

    bearish_rejection = (

        float(latest["close"])
        < float(latest["open"])

        and

        (
            float(latest["high"])
            - float(latest["close"])
        )
        >= candle_range * 0.55
    )

    rsi_rising = (
        float(latest["rsi"])
        > float(previous["rsi"])
    )

    rsi_falling = (
        float(latest["rsi"])
        < float(previous["rsi"])
    )

    macd_rising = (
        float(
            latest[
                "macd_histogram"
            ]
        )
        >
        float(
            previous[
                "macd_histogram"
            ]
        )
    )

    macd_falling = (
        float(
            latest[
                "macd_histogram"
            ]
        )
        <
        float(
            previous[
                "macd_histogram"
            ]
        )
    )

    side = None

    level = None

    reasons: List[str] = []

    score = 0

    # ========================================================
    # SUPPORT → BUY
    # ========================================================

    if (
        near_support
        and
        bullish_rejection
    ):

        side = "BUY"

        level = low

        score += 15

        reasons.append(
            "M15 range support is holding."
        )

        score += 10

        reasons.append(
            "M5 bullish rejection at range support."
        )

        # Prior-candle reclaim
        if (
            float(latest["close"])
            > float(previous["high"])
        ):

            score += 15

            reasons.append(
                "M5 reclaimed the prior candle high."
            )

        # RSI turn
        if rsi_rising:

            score += 10

            reasons.append(
                "M5 RSI is turning upward."
            )

        # MACD turn
        if macd_rising:

            score += 10

            reasons.append(
                "M5 MACD histogram is improving."
            )

        # M15 momentum stabilisation
        if (
            float(
                m15.iloc[-1]["rsi"]
            )
            >=
            float(
                m15.iloc[-2]["rsi"]
            )
        ):

            score += 5

            reasons.append(
                "M15 momentum is no longer deteriorating."
            )

        # Sweep and reclaim
        if (
            float(latest["low"])
            < low
            and
            price > low
        ):

            score += 15

            reasons.append(
                "Support sweep and reclaim detected."
            )

    # ========================================================
    # RESISTANCE → SELL
    # ========================================================

    elif (
        near_resistance
        and
        bearish_rejection
    ):

        side = "SELL"

        level = high

        score += 15

        reasons.append(
            "M15 range resistance is holding."
        )

        score += 10

        reasons.append(
            "M5 bearish rejection at range resistance."
        )

        if (
            float(latest["close"])
            < float(previous["low"])
        ):

            score += 15

            reasons.append(
                "M5 rejected below the prior candle low."
            )

        if rsi_falling:

            score += 10

            reasons.append(
                "M5 RSI is turning downward."
            )

        if macd_falling:

            score += 10

            reasons.append(
                "M5 MACD histogram is weakening."
            )

        if (
            float(
                m15.iloc[-1]["rsi"]
            )
            <=
            float(
                m15.iloc[-2]["rsi"]
            )
        ):

            score += 5

            reasons.append(
                "M15 momentum is no longer strengthening."
            )

        # Sweep and rejection
        if (
            float(latest["high"])
            > high
            and
            price < high
        ):

            score += 15

            reasons.append(
                "Resistance sweep and rejection detected."
            )

    else:

        return None

    # ========================================================
    # ROOM TO OPPOSITE EDGE
    # ========================================================

    distance_to_opposite = (

        high - price

        if side == "BUY"

        else

        price - low
    )

    # Don't buy support when resistance is immediately
    # above us, and don't sell resistance when support
    # is immediately below us.
    if (
        distance_to_opposite
        < atr_value * 1.20
    ):

        return None

    return {
        "side": side,
        "level": float(level),
        "score": min(
            score,
            100,
        ),
        "reasons": reasons,
    }


# ============================================================
# MODEL
# ============================================================

class GerkoXAUModel:

    # ========================================================
    # SIGNAL ID
    # ========================================================

    def generate_id(
        self,
        setup_type: str,
        side: str,
        signal_time: datetime,
    ) -> str:

        setup_codes = {

            "TREND_CONTINUATION":
                "TC",

            "SUPPORT_BOUNCE":
                "SB",

            "RESISTANCE_BOUNCE":
                "RB",

            "BREAK_RETEST":
                "BR",

            "RANGE_REVERSION":
                "RG",
        }

        code = setup_codes.get(
            setup_type,
            "LX",
        )

        side_code = (
            "B"
            if side == "BUY"
            else "S"
        )

        time_code = (
            signal_time.strftime(
                "%Y%m%d-%H%M"
            )
        )

        random_code = (
            secrets.token_hex(
                2
            ).upper()
        )

        return (
            f"LX-{code}-{side_code}-"
            f"{time_code}-{random_code}"
        )

    # ========================================================
    # HIGHER-TIMEFRAME CONTEXT
    # ========================================================

    def trend_context(
        self,
        h4: pd.DataFrame,
        h1: pd.DataFrame,
    ) -> str:

        h4_last = h4.iloc[-1]

        h1_last = h1.iloc[-1]

        bullish = (

            float(h4_last["close"])
            > float(h4_last["ema50"])

            and

            float(h1_last["close"])
            > float(h1_last["ema50"])
        )

        bearish = (

            float(h4_last["close"])
            < float(h4_last["ema50"])

            and

            float(h1_last["close"])
            < float(h1_last["ema50"])
        )

        if bullish:

            return "BULLISH"

        if bearish:

            return "BEARISH"

        return "NEUTRAL"

    # ========================================================
    # TREND SIGNAL
    # ========================================================

    def trend_signal(
        self,
        h4: pd.DataFrame,
        h1: pd.DataFrame,
        m15: pd.DataFrame,
        m5: pd.DataFrame,
        signal_time: datetime,
    ) -> Optional[Signal]:

        h4_last = h4.iloc[-1]

        h1_last = h1.iloc[-1]

        m15_last = m15.iloc[-1]

        m5_last = m5.iloc[-1]

        context = self.trend_context(
            h4,
            h1,
        )

        if context == "NEUTRAL":

            return None

        side = (
            "BUY"
            if context == "BULLISH"
            else "SELL"
        )

        impulse = current_impulse(
            m5
        )

        # Never trade directly against
        # the live M5 impulse.
        if impulse not in {
            side,
            "NEUTRAL",
        }:

            return None

        atr_now = float(
            m15_last["atr"]
        )

        price = float(
            m5_last["close"]
        )

        ema20_value = float(
            m5_last["ema20"]
        )

        if candle_exhausted(
            m5,
            atr_now,
        ):

            return None

        if overextended_from_value(
            price,
            ema20_value,
            atr_now,
        ):

            return None

        score = 0

        reasons: List[str] = []

        if side == "BUY":

            if (
                float(h4_last["close"])
                > float(h4_last["ema50"])
            ):

                score += 12

                reasons.append(
                    "H4 price is above EMA50."
                )

            if (

                float(h1_last["close"])
                >
                float(h1_last["ema50"])
                >
                float(h1_last["ema200"])

            ):

                score += 18

                reasons.append(
                    "H1 bullish EMA structure."
                )

            if (
                float(
                    h1_last[
                        "ema50_slope"
                    ]
                )
                > 0
            ):

                score += 8

            if (
                float(m15_last["ema20"])
                >
                float(m15_last["ema50"])
            ):

                score += 10

                reasons.append(
                    "M15 trend structure is bullish."
                )

            if (
                float(m5_last["close"])
                >
                float(m5_last["ema20"])
            ):

                score += 8

            if (
                float(m5_last["rsi"])
                >
                float(
                    m5.iloc[-2]["rsi"]
                )
            ):

                score += 8

                reasons.append(
                    "M5 RSI is rising."
                )

            if (
                float(
                    m5_last[
                        "macd_histogram"
                    ]
                )
                >
                float(
                    m5.iloc[-2][
                        "macd_histogram"
                    ]
                )
            ):

                score += 8

                reasons.append(
                    "M5 MACD momentum is improving."
                )

        else:

            if (
                float(h4_last["close"])
                <
                float(h4_last["ema50"])
            ):

                score += 12

                reasons.append(
                    "H4 price is below EMA50."
                )

            if (

                float(h1_last["close"])
                <
                float(h1_last["ema50"])
                <
                float(h1_last["ema200"])

            ):

                score += 18

                reasons.append(
                    "H1 bearish EMA structure."
                )

            if (
                float(
                    h1_last[
                        "ema50_slope"
                    ]
                )
                < 0
            ):

                score += 8

            if (
                float(m15_last["ema20"])
                <
                float(m15_last["ema50"])
            ):

                score += 10

                reasons.append(
                    "M15 trend structure is bearish."
                )

            if (
                float(m5_last["close"])
                <
                float(m5_last["ema20"])
            ):

                score += 8

            if (
                float(m5_last["rsi"])
                <
                float(
                    m5.iloc[-2]["rsi"]
                )
            ):

                score += 8

                reasons.append(
                    "M5 RSI is falling."
                )

            if (
                float(
                    m5_last[
                        "macd_histogram"
                    ]
                )
                <
                float(
                    m5.iloc[-2][
                        "macd_histogram"
                    ]
                )
            ):

                score += 8

                reasons.append(
                    "M5 MACD momentum is weakening."
                )

        if (
            float(h1_last["adx"])
            >= 22
        ):

            score += 12

            reasons.append(
                (
                    f"H1 ADX "
                    f"{float(h1_last['adx']):.1f} "
                    "confirms trend strength."
                )
            )

        atr_median = float(
            m15["atr"]
            .tail(60)
            .median()
        )

        if (
            atr_now
            >= atr_median * 0.80
        ):

            score += 8

            reasons.append(
                "Volatility is sufficient."
            )

        score = min(
            score,
            100,
        )

        decision = classify_signal(
            score,
            TREND_MIN_SCORE,
        )

        if not decision.allowed:

            return None

        if move_exhausted(
            price,
            ema20_value,
            atr_now,
            MAX_TREND_CHASE_ATR,
        ):

            return None

        return self.create_signal(
            signal_time=signal_time,
            setup_type="TREND_CONTINUATION",
            setup_name=(
                f"{context.title()} "
                "Trend Continuation"
            ),
            side=side,
            score=score,
            context=context,
            impulse=impulse,
            reference_level=ema20_value,
            level_strength=0,
            level_touches=0,
            price=price,
            atr_value=atr_now,
            reasons=reasons,
            decision=decision,
        )

    # ========================================================
    # STANDARD SIGNAL CREATOR
    # ========================================================

    def create_signal(
        self,
        signal_time: datetime,
        setup_type: str,
        setup_name: str,
        side: str,
        score: int,
        context: str,
        impulse: str,
        reference_level: float,
        level_strength: int,
        level_touches: int,
        price: float,
        atr_value: float,
        reasons: List[str],
        decision,
    ) -> Signal:

        entry_buffer = (
            atr_value * 0.10
        )

        entry_low = (
            price
            - entry_buffer
        )

        entry_high = (
            price
            + entry_buffer
        )

        entry_mid = (
            entry_low
            + entry_high
        ) / 2

        if side == "BUY":

            if (
                setup_type
                == "SUPPORT_BOUNCE"
            ):

                stop_loss = min(
                    reference_level
                    - atr_value * 0.40,

                    entry_mid
                    - atr_value * 0.75,
                )

            else:

                stop_loss = (
                    entry_mid
                    - atr_value * 0.80
                )

            risk = (
                entry_mid
                - stop_loss
            )

            tp1 = (
                entry_mid
                + risk
            )

            tp2 = (
                entry_mid
                + risk * 2
            )

            tp3 = (
                entry_mid
                + risk * 3
            )

            invalidation = (
                f"M15 close below "
                f"{stop_loss:.2f} "
                "invalidates the BUY setup."
            )

        else:

            if (
                setup_type
                == "RESISTANCE_BOUNCE"
            ):

                stop_loss = max(
                    reference_level
                    + atr_value * 0.40,

                    entry_mid
                    + atr_value * 0.75,
                )

            else:

                stop_loss = (
                    entry_mid
                    + atr_value * 0.80
                )

            risk = (
                stop_loss
                - entry_mid
            )

            tp1 = (
                entry_mid
                - risk
            )

            tp2 = (
                entry_mid
                - risk * 2
            )

            tp3 = (
                entry_mid
                - risk * 3
            )

            invalidation = (
                f"M15 close above "
                f"{stop_loss:.2f} "
                "invalidates the SELL setup."
            )

        rr = calculate_rr(
            entry_mid,
            stop_loss,
            tp2,
            side,
        )

        return Signal(

            signal_id=self.generate_id(
                setup_type,
                side,
                signal_time,
            ),

            created_at=signal_time,

            symbol="XAUUSD",

            setup_type=setup_type,

            setup_name=setup_name,

            side=side,

            score=score,

            grade=decision.grade,

            quality=decision.quality,

            trend_context=context,

            impulse=impulse,

            reference_level=round(
                reference_level,
                2,
            ),

            level_strength=level_strength,

            level_touches=level_touches,

            entry_low=round(
                entry_low,
                2,
            ),

            entry_high=round(
                entry_high,
                2,
            ),

            entry_mid=round(
                entry_mid,
                2,
            ),

            stop_loss=round(
                stop_loss,
                2,
            ),

            tp1=round(
                tp1,
                2,
            ),

            tp2=round(
                tp2,
                2,
            ),

            tp3=round(
                tp3,
                2,
            ),

            rr=rr,

            lot_size=decision.lot_size,

            reasons=reasons[:10],

            invalidation=invalidation,
        )

    # ========================================================
    # RANGE SIGNAL CREATOR
    # ========================================================

    def create_range_signal(
        self,
        signal_time: datetime,
        side: str,
        score: int,
        context: str,
        impulse: str,
        reference_level: float,
        range_high: float,
        range_low: float,
        price: float,
        atr_value: float,
        reasons: List[str],
        decision,
    ) -> Signal:

        entry_buffer = (
            atr_value * 0.08
        )

        entry_low = (
            price
            - entry_buffer
        )

        entry_high = (
            price
            + entry_buffer
        )

        entry_mid = (
            entry_low
            + entry_high
        ) / 2

        # ----------------------------------------------------
        # RANGE BUY
        # ----------------------------------------------------

        if side == "BUY":

            stop_loss = min(

                range_low
                - atr_value * 0.30,

                entry_mid
                - atr_value * 0.65,
            )

            risk = (
                entry_mid
                - stop_loss
            )

            room = max(

                range_high
                - entry_mid,

                risk * 1.25,
            )

            tp1 = (
                entry_mid
                + min(
                    risk,
                    room * 0.50,
                )
            )

            tp2 = (
                entry_mid
                + min(
                    risk * 1.50,
                    room * 0.85,
                )
            )

            tp3 = (
                entry_mid
                + min(
                    risk * 2.00,
                    room * 0.95,
                )
            )

            invalidation = (
                f"M15 close below "
                f"{stop_loss:.2f} "
                "or clean range failure "
                "invalidates the BUY setup."
            )

        # ----------------------------------------------------
        # RANGE SELL
        # ----------------------------------------------------

        else:

            stop_loss = max(

                range_high
                + atr_value * 0.30,

                entry_mid
                + atr_value * 0.65,
            )

            risk = (
                stop_loss
                - entry_mid
            )

            room = max(

                entry_mid
                - range_low,

                risk * 1.25,
            )

            tp1 = (
                entry_mid
                - min(
                    risk,
                    room * 0.50,
                )
            )

            tp2 = (
                entry_mid
                - min(
                    risk * 1.50,
                    room * 0.85,
                )
            )

            tp3 = (
                entry_mid
                - min(
                    risk * 2.00,
                    room * 0.95,
                )
            )

            invalidation = (
                f"M15 close above "
                f"{stop_loss:.2f} "
                "or clean range failure "
                "invalidates the SELL setup."
            )

        rr = calculate_rr(
            entry_mid,
            stop_loss,
            tp2,
            side,
        )

        return Signal(

            signal_id=self.generate_id(
                "RANGE_REVERSION",
                side,
                signal_time,
            ),

            created_at=signal_time,

            symbol="XAUUSD",

            setup_type="RANGE_REVERSION",

            setup_name=(

                "Range Support Reversion"

                if side == "BUY"

                else

                "Range Resistance Reversion"
            ),

            side=side,

            score=score,

            grade=decision.grade,

            quality=decision.quality,

            trend_context=context,

            impulse=impulse,

            reference_level=round(
                reference_level,
                2,
            ),

            level_strength=0,

            level_touches=0,

            entry_low=round(
                entry_low,
                2,
            ),

            entry_high=round(
                entry_high,
                2,
            ),

            entry_mid=round(
                entry_mid,
                2,
            ),

            stop_loss=round(
                stop_loss,
                2,
            ),

            tp1=round(
                tp1,
                2,
            ),

            tp2=round(
                tp2,
                2,
            ),

            tp3=round(
                tp3,
                2,
            ),

            rr=rr,

            lot_size=decision.lot_size,

            reasons=reasons[:10],

            invalidation=invalidation,
        )

    # ========================================================
    # MASTER ANALYSIS
    # ========================================================

    def analyse(
        self,
        candles: Dict[str, pd.DataFrame],
        signal_time: datetime,
        allow_trend: bool = True,
        allow_sr: bool = True,
    ) -> Optional[Signal]:

        h4 = enrich(
            candles["h4"]
        )

        h1 = enrich(
            candles["h1"]
        )

        m15 = enrich(
            candles["m15"]
        )

        m5 = enrich(
            candles["m5"]
        )

        if min(
            len(h4),
            len(h1),
            len(m15),
            len(m5),
        ) < 100:

            return None

        atr_value = float(
            m15.iloc[-1]["atr"]
        )

        current_price = float(
            m5.iloc[-1]["close"]
        )

        impulse = current_impulse(
            m5
        )

        context = self.trend_context(
            h4,
            h1,
        )

        candidates: List[
            Signal
        ] = []

        # ====================================================
        # 1. TREND ENGINE
        # ====================================================

        if allow_trend:

            trend_candidate = (
                self.trend_signal(
                    h4,
                    h1,
                    m15,
                    m5,
                    signal_time,
                )
            )

            if trend_candidate:

                candidates.append(
                    trend_candidate
                )

        # ====================================================
        # 2. S/R + RANGE + BREAK/RETEST
        # ====================================================

        if allow_sr:

            h1_levels = build_levels(
                h1.tail(200),
                atr_value,
                "H1",
            )

            m15_levels = build_levels(
                m15.tail(250),
                atr_value,
                "M15",
            )

            levels = combine_levels(
                h1_levels,
                m15_levels,
                atr_value,
            )

            # =================================================
            # SUPPORT BOUNCE
            # =================================================

            support = (
                detect_support_bounce(
                    m5,
                    m15,
                    levels,
                    atr_value,
                )
            )

            if (
                support.detected
                and support.level
            ):

                if impulse == "SELL":

                    support.reasons.append(
                        (
                            "Current M5 impulse "
                            "is bearish, so BUY "
                            "bounce is vetoed."
                        )
                    )

                elif not move_exhausted(

                    current_price,

                    support.level.price,

                    atr_value,

                    MAX_SR_CHASE_ATR,
                ):

                    minimum = SR_MIN_SCORE

                    if context == "BEARISH":

                        minimum = (
                            COUNTERTREND_SR_MIN_SCORE
                        )

                        support.reasons.append(
                            (
                                "Countertrend support "
                                "bounce — stronger "
                                "confirmation required."
                            )
                        )

                    decision = classify_signal(
                        support.score,
                        minimum,
                    )

                    if decision.allowed:

                        candidates.append(

                            self.create_signal(

                                signal_time=signal_time,

                                setup_type=(
                                    "SUPPORT_BOUNCE"
                                ),

                                setup_name=(
                                    "Support "
                                    "Rejection Bounce"
                                ),

                                side="BUY",

                                score=support.score,

                                context=context,

                                impulse=impulse,

                                reference_level=(
                                    support.level.price
                                ),

                                level_strength=(
                                    support.level.strength
                                ),

                                level_touches=(
                                    support.level.touches
                                ),

                                price=current_price,

                                atr_value=atr_value,

                                reasons=(
                                    support.reasons
                                ),

                                decision=decision,
                            )
                        )

            # =================================================
            # RESISTANCE BOUNCE
            # =================================================

            resistance = (
                detect_resistance_bounce(
                    m5,
                    m15,
                    levels,
                    atr_value,
                )
            )

            if (
                resistance.detected
                and resistance.level
            ):

                if impulse == "BUY":

                    resistance.reasons.append(
                        (
                            "Current M5 impulse "
                            "is bullish, so SELL "
                            "bounce is vetoed."
                        )
                    )

                elif not move_exhausted(

                    current_price,

                    resistance.level.price,

                    atr_value,

                    MAX_SR_CHASE_ATR,
                ):

                    minimum = SR_MIN_SCORE

                    if context == "BULLISH":

                        minimum = (
                            COUNTERTREND_SR_MIN_SCORE
                        )

                        resistance.reasons.append(
                            (
                                "Countertrend resistance "
                                "bounce — stronger "
                                "confirmation required."
                            )
                        )

                    decision = classify_signal(
                        resistance.score,
                        minimum,
                    )

                    if decision.allowed:

                        candidates.append(

                            self.create_signal(

                                signal_time=signal_time,

                                setup_type=(
                                    "RESISTANCE_BOUNCE"
                                ),

                                setup_name=(
                                    "Resistance "
                                    "Rejection Bounce"
                                ),

                                side="SELL",

                                score=(
                                    resistance.score
                                ),

                                context=context,

                                impulse=impulse,

                                reference_level=(
                                    resistance.level.price
                                ),

                                level_strength=(
                                    resistance.level.strength
                                ),

                                level_touches=(
                                    resistance.level.touches
                                ),

                                price=current_price,

                                atr_value=atr_value,

                                reasons=(
                                    resistance.reasons
                                ),

                                decision=decision,
                            )
                        )

            # =================================================
            # 3. RANGE ENGINE
            # =================================================

            range_info = range_regime(
                m15,
                h1,
                atr_value,
            )

            if range_info:

                # ---------------------------------------------
                # A range breakout takes priority over
                # range reversal.
                # ---------------------------------------------

                if not range_breaking(
                    m15,
                    range_info,
                    atr_value,
                ):

                    ranged = range_signal(

                        m15=m15,

                        m5=m5,

                        range_info=range_info,

                        atr_value=atr_value,
                    )

                    if ranged:

                        decision = (
                            classify_signal(
                                ranged["score"],
                                RANGE_MIN_SCORE,
                            )
                        )

                        if decision.allowed:

                            range_reasons = list(
                                ranged[
                                    "reasons"
                                ]
                            )

                            range_width = (
                                range_info[
                                    "width"
                                ]
                            )

                            range_width_atr = (
                                range_info[
                                    "width_atr"
                                ]
                            )

                            range_reasons.insert(
                                0,
                                (
                                    "Ranging regime "
                                    "detected: "
                                    f"{range_width_atr:.1f} "
                                    "ATR wide."
                                ),
                            )

                            range_reasons.append(
                                (
                                    "Range-edge entry; "
                                    "middle-of-range "
                                    "trades are blocked."
                                )
                            )

                            range_reasons.append(
                                (
                                    f"H1 ADX "
                                    f"{range_info['h1_adx']:.1f} / "
                                    f"M15 ADX "
                                    f"{range_info['m15_adx']:.1f} "
                                    "show limited "
                                    "trend strength."
                                )
                            )

                            candidates.append(

                                self.create_range_signal(

                                    signal_time=signal_time,

                                    side=ranged[
                                        "side"
                                    ],

                                    score=ranged[
                                        "score"
                                    ],

                                    context=context,

                                    impulse=impulse,

                                    reference_level=(
                                        ranged[
                                            "level"
                                        ]
                                    ),

                                    range_high=(
                                        range_info[
                                            "high"
                                        ]
                                    ),

                                    range_low=(
                                        range_info[
                                            "low"
                                        ]
                                    ),

                                    price=current_price,

                                    atr_value=atr_value,

                                    reasons=(
                                        range_reasons
                                    ),

                                    decision=decision,
                                )
                            )

            # =================================================
            # 4. BREAK & RETEST
            # =================================================

            break_retest = (
                detect_break_retest(
                    m5,
                    levels,
                    atr_value,
                )
            )

            if (
                break_retest.detected
                and break_retest.level
            ):

                if impulse not in {
                    break_retest.side,
                    "NEUTRAL",
                }:

                    pass

                elif not move_exhausted(

                    current_price,

                    break_retest.level.price,

                    atr_value,

                    MAX_SR_CHASE_ATR,
                ):

                    decision = classify_signal(

                        break_retest.score,

                        BREAK_RETEST_MIN_SCORE,
                    )

                    if decision.allowed:

                        candidates.append(

                            self.create_signal(

                                signal_time=signal_time,

                                setup_type=(
                                    "BREAK_RETEST"
                                ),

                                setup_name=(
                                    "Support/Resistance "
                                    "Break & Retest"
                                ),

                                side=(
                                    break_retest.side
                                ),

                                score=(
                                    break_retest.score
                                ),

                                context=context,

                                impulse=impulse,

                                reference_level=(
                                    break_retest
                                    .level
                                    .price
                                ),

                                level_strength=(
                                    break_retest
                                    .level
                                    .strength
                                ),

                                level_touches=(
                                    break_retest
                                    .level
                                    .touches
                                ),

                                price=current_price,

                                atr_value=atr_value,

                                reasons=(
                                    break_retest
                                    .reasons
                                ),

                                decision=decision,
                            )
                        )

        # ====================================================
        # NO CANDIDATE
        # ====================================================

        if not candidates:

            return None

        # ====================================================
        # CANDIDATE SELECTION
        # ====================================================

        candidates.sort(

            key=lambda signal: (

                signal.score,

                # Prefer actual level-based setups
                # when scores are tied.
                1
                if signal.setup_type
                != "TREND_CONTINUATION"
                else 0,

                # Prefer range edge over trend when
                # score is identical in a ranging market.
                1
                if signal.setup_type
                == "RANGE_REVERSION"
                else 0,
            ),

            reverse=True,
        )

        return candidates[0]


# ============================================================
# TELEGRAM FORMATTER
# ============================================================

def format_signal(
    signal: Signal,
) -> str:

    headers = {

        "TREND_CONTINUATION":
            "📈🧠 <b>LumiAj-XAU TREND SIGNAL</b>",

        "SUPPORT_BOUNCE":
            "🟢🧠 <b>LumiAj-XAU SUPPORT BOUNCE</b>",

        "RESISTANCE_BOUNCE":
            "🔴🧠 <b>LumiAj-XAU RESISTANCE BOUNCE</b>",

        "BREAK_RETEST":
            "🔄🧠 <b>LumiAj-XAU BREAK & RETEST</b>",

        "RANGE_REVERSION":
            "↔️🧠 <b>LumiAj-XAU RANGE REVERSAL</b>",
    }

    header = headers.get(

        signal.setup_type,

        "🧠 <b>LumiAj-XAU SIGNAL</b>",
    )

    direction_emoji = (
        "🟢"
        if signal.side == "BUY"
        else "🔴"
    )

    reasons = "\n".join(

        f"• {reason}"

        for reason
        in signal.reasons
    )

    setup_tag = {

        "TREND_CONTINUATION":
            "#LX_TREND",

        "SUPPORT_BOUNCE":
            "#LX_SUPPORT_BOUNCE",

        "RESISTANCE_BOUNCE":
            "#LX_RESISTANCE_BOUNCE",

        "BREAK_RETEST":
            "#LX_BREAK_RETEST",

        "RANGE_REVERSION":
            "#LX_RANGE",

    }.get(
        signal.setup_type,
        "#LX_SIGNAL",
    )

    level_section = ""

    if signal.level_touches > 0:

        level_section = f"""
<b>KEY LEVEL</b>
Level: <b>{signal.reference_level:.2f}</b>
Touches: {signal.level_touches}
Strength: {signal.level_strength}/20
"""

    return f"""
{header}
━━━━━━━━━━━━━━━━━━━━━━

<b>LumiAj IDENTITY</b>
Signal ID: <code>{signal.signal_id}</code>
Symbol: XAUUSD
Setup: <b>{signal.setup_name}</b>

<b>DECISION</b>
{direction_emoji} Direction: <b>{signal.side}</b>
Confidence: <b>{signal.score}/100</b>
Grade: <b>{signal.grade}</b>
Quality: {signal.quality}
HTF Context: <b>{signal.trend_context}</b>
Current M5 Impulse: <b>{signal.impulse}</b>

{level_section}
<b>ENTRY</b>
Entry Zone: <b>{signal.entry_low:.2f} – {signal.entry_high:.2f}</b>
Entry Mid: {signal.entry_mid:.2f}
Stop Loss: <b>{signal.stop_loss:.2f}</b>

<b>TARGETS</b>
TP1: {signal.tp1:.2f}
TP2: {signal.tp2:.2f}
TP3: {signal.tp3:.2f}

<b>RISK PROFILE</b>
TP2 R:R: <b>1:{signal.rr:.2f}</b>
Model Lot Grade: <b>{signal.lot_size:.2f}</b>

<b>WHY LumiAj TRIGGERED</b>
{reasons}

<b>INVALIDATION</b>
{signal.invalidation}

⚠️ Anti-chase protection active.
Do not enter materially outside the stated entry zone.

#LUMIAJ_XAU
{setup_tag}
#LX_{signal.side}
""".strip()