from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from risk_engine import classify_signal

from sr_engine import (
    build_levels,
    combine_levels,
    detect_support_bounce,
    detect_resistance_bounce,
    detect_break_retest,
)


# ============================================================
# CONFIGURATION
# ============================================================

TREND_MIN_SCORE = int(os.getenv("TREND_MIN_SCORE", "76"))
SR_MIN_SCORE = int(os.getenv("SR_MIN_SCORE", "74"))
COUNTERTREND_SR_MIN_SCORE = int(
    os.getenv("COUNTERTREND_SR_MIN_SCORE", "84")
)
BREAK_RETEST_MIN_SCORE = int(
    os.getenv("BREAK_RETEST_MIN_SCORE", "76")
)

MAX_TREND_CHASE_ATR = float(
    os.getenv("MAX_TREND_CHASE_ATR", "0.65")
)

MAX_SR_CHASE_ATR = float(
    os.getenv("MAX_SR_CHASE_ATR", "0.45")
)

RANGE_MIN_SCORE = int(
    os.getenv("RANGE_MIN_SCORE", "76")
)

RANGE_ADX_MAX = float(
    os.getenv("RANGE_ADX_MAX", "20")
)

RANGE_EDGE_ATR = float(
    os.getenv("RANGE_EDGE_ATR", "0.35")
)

RANGE_MIN_WIDTH_ATR = float(
    os.getenv("RANGE_MIN_WIDTH_ATR", "2.0")
)

RANGE_MAX_WIDTH_ATR = float(
    os.getenv("RANGE_MAX_WIDTH_ATR", "8.0")
)

RANGE_BREAK_BUFFER_ATR = float(
    os.getenv("RANGE_BREAK_BUFFER_ATR", "0.12")
)


# ============================================================
# SIGNAL OBJECT
# ============================================================

@dataclass
class Signal:
    signal_id: str
    created_at: str
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

    reasons: list[str]

    invalidation: str


# ============================================================
# BASIC HELPERS
# ============================================================

def calculate_rr(
    entry: float,
    stop: float,
    target: float,
) -> float:

    risk = abs(entry - stop)

    if risk <= 0:
        return 0.0

    reward = abs(target - entry)

    return round(reward / risk, 2)


def safe_float(value, default=0.0) -> float:

    try:
        result = float(value)

        if not np.isfinite(result):
            return default

        return result

    except Exception:
        return default


def safe_int(value, default=0) -> int:

    try:
        return int(value)

    except Exception:
        return default


# ============================================================
# INDICATORS
# ============================================================

def ema(series: pd.Series, period: int) -> pd.Series:

    return series.ewm(
        span=period,
        adjust=False,
        min_periods=period,
    ).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    result = 100 - (100 / (1 + rs))

    return result.fillna(50)


def true_range(df: pd.DataFrame) -> pd.Series:

    previous_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]

    tr2 = (df["high"] - previous_close).abs()

    tr3 = (df["low"] - previous_close).abs()

    return pd.concat(
        [tr1, tr2, tr3],
        axis=1,
    ).max(axis=1)


def atr(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.Series:

    return true_range(df).ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()


def macd_histogram(
    series: pd.Series,
) -> pd.Series:

    fast = ema(series, 12)
    slow = ema(series, 26)

    macd = fast - slow

    signal = ema(macd, 9)

    return macd - signal


def adx(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.Series:

    high = df["high"]
    low = df["low"]

    up_move = high.diff()

    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) & (up_move > 0),
            up_move,
            0.0,
        ),
        index=df.index,
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) & (down_move > 0),
            down_move,
            0.0,
        ),
        index=df.index,
    )

    tr = true_range(df)

    atr_value = tr.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    plus_di = (
        100
        * plus_dm.ewm(
            alpha=1 / period,
            adjust=False,
        ).mean()
        / atr_value.replace(0, np.nan)
    )

    minus_di = (
        100
        * minus_dm.ewm(
            alpha=1 / period,
            adjust=False,
        ).mean()
        / atr_value.replace(0, np.nan)
    )

    denominator = (
        plus_di + minus_di
    ).replace(0, np.nan)

    dx = (
        100
        * (plus_di - minus_di).abs()
        / denominator
    )

    return dx.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean().fillna(0)


def enrich(df: pd.DataFrame) -> pd.DataFrame:

    data = df.copy()

    required = [
        "open",
        "high",
        "low",
        "close",
    ]

    for column in required:

        if column not in data.columns:

            raise ValueError(
                f"Missing required candle column: {column}"
            )

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data = data.dropna(
        subset=required
    ).copy()

    data["ema20"] = ema(
        data["close"],
        20,
    )

    data["ema50"] = ema(
        data["close"],
        50,
    )

    data["ema200"] = ema(
        data["close"],
        200,
    )

    data["rsi"] = rsi(
        data["close"],
        14,
    )

    data["atr"] = atr(
        data,
        14,
    )

    data["macd_histogram"] = macd_histogram(
        data["close"]
    )

    data["adx"] = adx(
        data,
        14,
    )

    data["body"] = (
        data["close"] - data["open"]
    ).abs()

    data["range"] = (
        data["high"] - data["low"]
    )

    data["upper_wick"] = (
        data["high"]
        - data[["open", "close"]].max(axis=1)
    )

    data["lower_wick"] = (
        data[["open", "close"]].min(axis=1)
        - data["low"]
    )

    return data


# ============================================================
# CURRENT M5 IMPULSE
# ============================================================

def current_impulse(
    m5: pd.DataFrame,
) -> str:

    if len(m5) < 5:
        return "NEUTRAL"

    latest = m5.iloc[-1]
    previous = m5.iloc[-2]

    close = safe_float(latest["close"])
    previous_close = safe_float(
        previous["close"]
    )

    atr_value = safe_float(
        latest["atr"]
    )

    if atr_value <= 0:
        return "NEUTRAL"

    move = close - previous_close

    rsi_now = safe_float(
        latest["rsi"],
        50,
    )

    rsi_previous = safe_float(
        previous["rsi"],
        50,
    )

    macd_now = safe_float(
        latest["macd_histogram"]
    )

    macd_previous = safe_float(
        previous["macd_histogram"]
    )

    bullish = (
        move > atr_value * 0.08
        and rsi_now > rsi_previous
        and macd_now >= macd_previous
    )

    bearish = (
        move < -atr_value * 0.08
        and rsi_now < rsi_previous
        and macd_now <= macd_previous
    )

    if bullish:
        return "BULLISH"

    if bearish:
        return "BEARISH"

    return "NEUTRAL"


# ============================================================
# EXHAUSTION / CHASE PROTECTION
# ============================================================

def trigger_exhausted(
    m5: pd.DataFrame,
    side: str,
    atr_value: float,
) -> bool:

    if len(m5) < 3:
        return False

    latest = m5.iloc[-1]

    candle_range = safe_float(
        latest["high"]
    ) - safe_float(
        latest["low"]
    )

    if candle_range <= 0:
        return False

    body = abs(
        safe_float(latest["close"])
        - safe_float(latest["open"])
    )

    # Very large trigger candle.
    if candle_range > atr_value * 1.35:
        return True

    # Huge body relative to ATR.
    if body > atr_value * 1.05:
        return True

    return False


def chase_distance(
    close: float,
    entry_reference: float,
    atr_value: float,
) -> float:

    if atr_value <= 0:
        return 999.0

    return abs(
        close - entry_reference
    ) / atr_value


# ============================================================
# TREND CONTEXT
# ============================================================

class GerkoXAUModel:

    def __init__(
        self,
        symbol: str = "XAU/USD",
    ):

        self.symbol = symbol

    # --------------------------------------------------------
    # TREND CONTEXT
    # --------------------------------------------------------

    def trend_context(
        self,
        h4: pd.DataFrame,
        h1: pd.DataFrame,
        m15: pd.DataFrame,
    ) -> str:

        h4_last = h4.iloc[-1]
        h1_last = h1.iloc[-1]
        m15_last = m15.iloc[-1]

        h4_close = safe_float(
            h4_last["close"]
        )

        h1_close = safe_float(
            h1_last["close"]
        )

        m15_close = safe_float(
            m15_last["close"]
        )

        h4_ema50 = safe_float(
            h4_last["ema50"]
        )

        h4_ema200 = safe_float(
            h4_last["ema200"]
        )

        h1_ema50 = safe_float(
            h1_last["ema50"]
        )

        h1_ema200 = safe_float(
            h1_last["ema200"]
        )

        m15_ema20 = safe_float(
            m15_last["ema20"]
        )

        bullish_score = 0
        bearish_score = 0

        if h4_close > h4_ema50:
            bullish_score += 1

        if h4_ema50 > h4_ema200:
            bullish_score += 1

        if h1_close > h1_ema50:
            bullish_score += 1

        if h1_ema50 > h1_ema200:
            bullish_score += 1

        if m15_close > m15_ema20:
            bullish_score += 1

        if h4_close < h4_ema50:
            bearish_score += 1

        if h4_ema50 < h4_ema200:
            bearish_score += 1

        if h1_close < h1_ema50:
            bearish_score += 1

        if h1_ema50 < h1_ema200:
            bearish_score += 1

        if m15_close < m15_ema20:
            bearish_score += 1

        if bullish_score >= 4:
            return "BULLISH"

        if bearish_score >= 4:
            return "BEARISH"

        return "NEUTRAL"

    # --------------------------------------------------------
    # TREND SIGNAL
    # --------------------------------------------------------

    def trend_signal(
        self,
        h4: pd.DataFrame,
        h1: pd.DataFrame,
        m15: pd.DataFrame,
        m5: pd.DataFrame,
        atr_value: float,
        context: str,
    ):

        latest = m5.iloc[-1]
        previous = m5.iloc[-2]

        close = safe_float(
            latest["close"]
        )

        ema20 = safe_float(
            latest["ema20"]
        )

        ema50 = safe_float(
            latest["ema50"]
        )

        rsi_now = safe_float(
            latest["rsi"],
            50,
        )

        rsi_previous = safe_float(
            previous["rsi"],
            50,
        )

        macd_now = safe_float(
            latest["macd_histogram"]
        )

        macd_previous = safe_float(
            previous["macd_histogram"]
        )

        bullish = (
            context == "BULLISH"
            and close > ema20
            and ema20 > ema50
            and rsi_now >= rsi_previous
            and macd_now >= macd_previous
        )

        bearish = (
            context == "BEARISH"
            and close < ema20
            and ema20 < ema50
            and rsi_now <= rsi_previous
            and macd_now <= macd_previous
        )

        if not bullish and not bearish:
            return None

        side = "BUY" if bullish else "SELL"

        reasons = []

        score = 0

        if context == "BULLISH" and side == "BUY":
            score += 25
            reasons.append(
                "Higher-timeframe structure remains bullish."
            )

        if context == "BEARISH" and side == "SELL":
            score += 25
            reasons.append(
                "Higher-timeframe structure remains bearish."
            )

        if side == "BUY":

            if close > ema20:
                score += 10
                reasons.append(
                    "Price is holding above M5 EMA20."
                )

            if ema20 > ema50:
                score += 10
                reasons.append(
                    "M5 EMA20 remains above EMA50."
                )

            if rsi_now > rsi_previous:
                score += 10
                reasons.append(
                    "M5 RSI is turning higher."
                )

            if macd_now > macd_previous:
                score += 10
                reasons.append(
                    "M5 MACD momentum is improving."
                )

        else:

            if close < ema20:
                score += 10
                reasons.append(
                    "Price is holding below M5 EMA20."
                )

            if ema20 < ema50:
                score += 10
                reasons.append(
                    "M5 EMA20 remains below EMA50."
                )

            if rsi_now < rsi_previous:
                score += 10
                reasons.append(
                    "M5 RSI is turning lower."
                )

            if macd_now < macd_previous:
                score += 10
                reasons.append(
                    "M5 MACD momentum is weakening."
                )

        # Anti-late-entry protection.
        distance = abs(
            close - ema20
        ) / max(atr_value, 1e-9)

        if distance > MAX_TREND_CHASE_ATR:
            return None

        if trigger_exhausted(
            m5,
            side,
            atr_value,
        ):
            return None

        score += 10

        reasons.append(
            "Entry remains within the anti-chase zone."
        )

        if score < TREND_MIN_SCORE:
            return None

        return {
            "detected": True,
            "side": side,
            "score": min(score, 100),
            "reference_level": ema20,
            "level_strength": 0,
            "level_touches": 0,
            "reasons": reasons,
        }

    # --------------------------------------------------------
    # RANGE REGIME
    # --------------------------------------------------------

    def range_regime(
        self,
        m15: pd.DataFrame,
        h1: pd.DataFrame,
        atr_value: float,
    ):

        if len(m15) < 60 or len(h1) < 60:
            return None

        recent = m15.tail(48)

        range_high = safe_float(
            recent["high"].max()
        )

        range_low = safe_float(
            recent["low"].min()
        )

        width = range_high - range_low

        if width <= 0:
            return None

        width_atr = width / max(
            atr_value,
            1e-9,
        )

        if (
            width_atr < RANGE_MIN_WIDTH_ATR
            or width_atr > RANGE_MAX_WIDTH_ATR
        ):
            return None

        h1_last = h1.iloc[-1]
        m15_last = m15.iloc[-1]

        h1_adx = safe_float(
            h1_last["adx"]
        )

        m15_adx = safe_float(
            m15_last["adx"]
        )

        if h1_adx > RANGE_ADX_MAX:
            return None

        if m15_adx > RANGE_ADX_MAX + 3:
            return None

        h1_ema50_slope = abs(
            safe_float(
                h1["ema50"].iloc[-1]
            )
            - safe_float(
                h1["ema50"].iloc[-6]
            )
        )

        m15_ema50_slope = abs(
            safe_float(
                m15["ema50"].iloc[-1]
            )
            - safe_float(
                m15["ema50"].iloc[-6]
            )
        )

        if h1_ema50_slope > atr_value * 0.35:
            return None

        if m15_ema50_slope > atr_value * 0.45:
            return None

        close = safe_float(
            m15_last["close"]
        )

        buffer = (
            atr_value
            * RANGE_BREAK_BUFFER_ATR
        )

        if (
            close < range_low - buffer
            or close > range_high + buffer
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

    # --------------------------------------------------------
    # RANGE BREAK
    # --------------------------------------------------------

    def range_breaking(
        self,
        m15: pd.DataFrame,
        atr_value: float,
        range_info: dict,
    ) -> bool:

        if len(m15) < 3:
            return False

        latest = m15.iloc[-1]
        previous = m15.iloc[-2]

        high = safe_float(
            range_info["high"]
        )

        low = safe_float(
            range_info["low"]
        )

        buffer = (
            atr_value
            * RANGE_BREAK_BUFFER_ATR
        )

        latest_close = safe_float(
            latest["close"]
        )

        previous_close = safe_float(
            previous["close"]
        )

        macd_latest = safe_float(
            latest["macd_histogram"]
        )

        macd_previous = safe_float(
            previous["macd_histogram"]
        )

        bullish_break = (
            latest_close > high + buffer
            and previous_close > high
            and macd_latest > macd_previous
        )

        bearish_break = (
            latest_close < low - buffer
            and previous_close < low
            and macd_latest < macd_previous
        )

        return (
            bullish_break
            or bearish_break
        )

    # --------------------------------------------------------
    # RANGE SIGNAL
    # --------------------------------------------------------

    def range_signal(
        self,
        m5: pd.DataFrame,
        m15: pd.DataFrame,
        h1: pd.DataFrame,
        atr_value: float,
        range_info: dict,
    ):

        latest = m5.iloc[-1]

        close = safe_float(
            latest["close"]
        )

        range_high = safe_float(
            range_info["high"]
        )

        range_low = safe_float(
            range_info["low"]
        )

        distance_high = abs(
            range_high - close
        )

        distance_low = abs(
            close - range_low
        )

        edge_distance = min(
            distance_high,
            distance_low,
        )

        if edge_distance > (
            atr_value * RANGE_EDGE_ATR
        ):
            return None

        reasons = []

        score = 0

        if distance_low <= distance_high:

            side = "BUY"
            reference = range_low

            score += 15
            reasons.append(
                "Price is testing the lower range boundary."
            )

            if (
                safe_float(latest["close"])
                > safe_float(latest["open"])
            ):
                score += 10
                reasons.append(
                    "M5 candle is showing bullish rejection."
                )

            if (
                safe_float(latest["lower_wick"])
                >= safe_float(latest["range"]) * 0.30
            ):
                score += 10
                reasons.append(
                    "Lower wick shows rejection of the range low."
                )

            if (
                safe_float(latest["rsi"])
                > safe_float(
                    m5["rsi"].iloc[-2]
                )
            ):
                score += 10
                reasons.append(
                    "M5 RSI is turning higher."
                )

            if (
                safe_float(latest["macd_histogram"])
                > safe_float(
                    m5["macd_histogram"].iloc[-2]
                )
            ):
                score += 10
                reasons.append(
                    "M5 MACD momentum is improving."
                )

            if (
                safe_float(
                    m15["rsi"].iloc[-1]
                )
                >= safe_float(
                    m15["rsi"].iloc[-2]
                )
            ):
                score += 5

            score += 15

            reasons.append(
                "Range support is holding."
            )

        else:

            side = "SELL"
            reference = range_high

            score += 15
            reasons.append(
                "Price is testing the upper range boundary."
            )

            if (
                safe_float(latest["close"])
                < safe_float(latest["open"])
            ):
                score += 10
                reasons.append(
                    "M5 candle is showing bearish rejection."
                )

            if (
                safe_float(latest["upper_wick"])
                >= safe_float(latest["range"]) * 0.30
            ):
                score += 10
                reasons.append(
                    "Upper wick shows rejection of the range high."
                )

            if (
                safe_float(latest["rsi"])
                < safe_float(
                    m5["rsi"].iloc[-2]
                )
            ):
                score += 10
                reasons.append(
                    "M5 RSI is turning lower."
                )

            if (
                safe_float(latest["macd_histogram"])
                < safe_float(
                    m5["macd_histogram"].iloc[-2]
                )
            ):
                score += 10
                reasons.append(
                    "M5 MACD momentum is weakening."
                )

            if (
                safe_float(
                    m15["rsi"].iloc[-1]
                )
                <= safe_float(
                    m15["rsi"].iloc[-2]
                )
            ):
                score += 5

            score += 15

            reasons.append(
                "Range resistance is holding."
            )

        # Need room to the opposite side.
        if side == "BUY":
            room = range_high - close
        else:
            room = close - range_low

        if room < atr_value * 1.20:
            return None

        if score < RANGE_MIN_SCORE:
            return None

        return {
            "detected": True,
            "side": side,
            "score": min(score, 100),
            "reference_level": reference,
            "level_strength": 20,
            "level_touches": 0,
            "reasons": reasons,
        }

    # --------------------------------------------------------
    # CREATE STANDARD SIGNAL
    # --------------------------------------------------------

    def create_signal(
        self,
        result,
        setup_type: str,
        m5: pd.DataFrame,
        m15: pd.DataFrame,
        h1: pd.DataFrame,
        atr_value: float,
        context: str,
        impulse: str,
    ):

        side = result["side"]

        score = safe_int(
            result.get("score", 0)
        )

        # ----------------------------------------------------
        # COUNTERTREND PROTECTION
        # ----------------------------------------------------

        opposite_context = (
            (
                side == "BUY"
                and context == "BEARISH"
            )
            or
            (
                side == "SELL"
                and context == "BULLISH"
            )
        )

        opposite_impulse = (
            (
                side == "BUY"
                and impulse == "BEARISH"
            )
            or
            (
                side == "SELL"
                and impulse == "BULLISH"
            )
        )

        if (
            setup_type
            in {
                "SUPPORT_BOUNCE",
                "RESISTANCE_BOUNCE",
            }
            and (
                opposite_context
                or opposite_impulse
            )
        ):

            if score < COUNTERTREND_SR_MIN_SCORE:
                return None

        # ----------------------------------------------------
        # CURRENT M5 IMPULSE VETO
        # ----------------------------------------------------

        if setup_type == "TREND":

            if side == "BUY" and impulse == "BEARISH":
                return None

            if side == "SELL" and impulse == "BULLISH":
                return None

        if setup_type == "BREAK_RETEST":

            if side == "BUY" and impulse == "BEARISH":
                return None

            if side == "SELL" and impulse == "BULLISH":
                return None

        # ----------------------------------------------------
        # REFERENCE
        # ----------------------------------------------------

        close = safe_float(
            m5["close"].iloc[-1]
        )

        level = result.get(
            "reference_level"
        )

        if level is None:

            reference_price = close
            level_strength = 0
            level_touches = 0

        else:

            reference_price = safe_float(
                level,
                close,
            )

            level_strength = safe_int(
                result.get(
                    "level_strength",
                    0,
                )
            )

            level_touches = safe_int(
                result.get(
                    "level_touches",
                    0,
                )
            )

        # ----------------------------------------------------
        # ENTRY
        # ----------------------------------------------------

        entry_buffer = atr_value * 0.08

        if side == "BUY":

            entry_low = min(
                close,
                reference_price,
            )

            entry_high = max(
                close,
                reference_price + entry_buffer,
            )

            entry_mid = (
                entry_low
                + entry_high
            ) / 2

        else:

            entry_low = min(
                close,
                reference_price - entry_buffer,
            )

            entry_high = max(
                close,
                reference_price,
            )

            entry_mid = (
                entry_low
                + entry_high
            ) / 2

        # ----------------------------------------------------
        # STOP
        # ----------------------------------------------------

        if side == "BUY":

            stop_loss = min(
                reference_price
                - atr_value * 0.45,
                entry_mid
                - atr_value * 0.65,
            )

        else:

            stop_loss = max(
                reference_price
                + atr_value * 0.45,
                entry_mid
                + atr_value * 0.65,
            )

        # ----------------------------------------------------
        # TARGETS
        # ----------------------------------------------------

        risk = abs(
            entry_mid - stop_loss
        )

        if risk <= 0:
            return None

        if side == "BUY":

            tp1 = entry_mid + risk * 1.0
            tp2 = entry_mid + risk * 2.0
            tp3 = entry_mid + risk * 3.0

        else:

            tp1 = entry_mid - risk * 1.0
            tp2 = entry_mid - risk * 2.0
            tp3 = entry_mid - risk * 3.0

        # ----------------------------------------------------
        # ANTI-CHASE
        # ----------------------------------------------------

        chase = chase_distance(
            close,
            reference_price,
            atr_value,
        )

        if setup_type == "TREND":

            if chase > MAX_TREND_CHASE_ATR:
                return None

        else:

            if chase > MAX_SR_CHASE_ATR:
                return None

        # ----------------------------------------------------
        # RISK ENGINE
        # ----------------------------------------------------

        minimum = TREND_MIN_SCORE

        if setup_type in {
            "SUPPORT_BOUNCE",
            "RESISTANCE_BOUNCE",
        }:
            minimum = SR_MIN_SCORE

        if setup_type == "BREAK_RETEST":
            minimum = BREAK_RETEST_MIN_SCORE

        risk_decision = classify_signal(
            score,
            minimum,
        )

        if not risk_decision.allowed:
            return None

        # ----------------------------------------------------
        # SETUP NAME
        # ----------------------------------------------------

        setup_names = {
            "TREND": "Trend Continuation",
            "SUPPORT_BOUNCE": "Support Bounce",
            "RESISTANCE_BOUNCE": "Resistance Bounce",
            "BREAK_RETEST": "Break & Retest",
        }

        setup_name = setup_names.get(
            setup_type,
            setup_type,
        )

        # ----------------------------------------------------
        # SIGNAL ID
        # ----------------------------------------------------

        now = datetime.utcnow()

        codes = {
            "TREND": "TC",
            "SUPPORT_BOUNCE": "SB",
            "RESISTANCE_BOUNCE": "RB",
            "BREAK_RETEST": "BR",
        }

        code = codes.get(
            setup_type,
            "LX",
        )

        signal_id = (
            f"LX-{code}-"
            f"{'B' if side == 'BUY' else 'S'}-"
            f"{now:%Y%m%d-%H%M}-"
            f"{uuid.uuid4().hex[:5].upper()}"
        )

        reasons = list(
            result.get(
                "reasons",
                [],
            )
        )

        reasons.append(
            f"Current M5 impulse: {impulse}."
        )

        reasons.append(
            f"Market context: {context}."
        )

        reasons.append(
            f"Entry chase distance: {chase:.2f} ATR."
        )

        return Signal(
            signal_id=signal_id,
            created_at=now.isoformat(),
            symbol=self.symbol,
            setup_type=setup_type,
            setup_name=setup_name,
            side=side,
            score=score,
            grade=risk_decision.grade,
            quality=risk_decision.quality,
            trend_context=context,
            impulse=impulse,
            reference_level=round(
                reference_price,
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
            rr=calculate_rr(
                entry_mid,
                stop_loss,
                tp2,
            ),
            lot_size=risk_decision.lot_size,
            reasons=reasons,
            invalidation=(
                "Setup invalidates if M5 momentum "
                "flips materially against the signal "
                "or price breaks the reference structure."
            ),
        )

    # --------------------------------------------------------
    # CREATE RANGE SIGNAL
    # --------------------------------------------------------

    def create_range_signal(
        self,
        result,
        m5: pd.DataFrame,
        m15: pd.DataFrame,
        h1: pd.DataFrame,
        atr_value: float,
        context: str,
        impulse: str,
    ):

        side = result["side"]

        score = safe_int(
            result.get(
                "score",
                0,
            )
        )

        if score < RANGE_MIN_SCORE:
            return None

        close = safe_float(
            m5["close"].iloc[-1]
        )

        range_high = safe_float(
            result.get(
                "range_high",
                close + atr_value * 2,
            )
        )

        range_low = safe_float(
            result.get(
                "range_low",
                close - atr_value * 2,
            )
        )

        reference = safe_float(
            result.get(
                "reference_level",
                range_low
                if side == "BUY"
                else range_high,
            )
        )

        entry_buffer = atr_value * 0.08

        if side == "BUY":

            entry_low = min(
                close,
                reference,
            )

            entry_high = max(
                close,
                reference + entry_buffer,
            )

            entry_mid = (
                entry_low
                + entry_high
            ) / 2

            stop_loss = min(
                range_low
                - atr_value * 0.30,
                entry_mid
                - atr_value * 0.65,
            )

            room = range_high - entry_mid

            tp1 = min(
                entry_mid
                + abs(entry_mid - stop_loss),
                range_high - atr_value * 0.20,
            )

            tp2 = min(
                entry_mid
                + abs(entry_mid - stop_loss) * 2,
                range_high - atr_value * 0.10,
            )

            tp3 = min(
                entry_mid
                + abs(entry_mid - stop_loss) * 3,
                range_high,
            )

        else:

            entry_low = min(
                close,
                reference - entry_buffer,
            )

            entry_high = max(
                close,
                reference,
            )

            entry_mid = (
                entry_low
                + entry_high
            ) / 2

            stop_loss = max(
                range_high
                + atr_value * 0.30,
                entry_mid
                + atr_value * 0.65,
            )

            room = entry_mid - range_low

            tp1 = max(
                entry_mid
                - abs(entry_mid - stop_loss),
                range_low + atr_value * 0.20,
            )

            tp2 = max(
                entry_mid
                - abs(entry_mid - stop_loss) * 2,
                range_low + atr_value * 0.10,
            )

            tp3 = max(
                entry_mid
                - abs(entry_mid - stop_loss) * 3,
                range_low,
            )

        if room < atr_value * 1.20:
            return None

        risk_decision = classify_signal(
            score,
            RANGE_MIN_SCORE,
        )

        if not risk_decision.allowed:
            return None

        now = datetime.utcnow()

        signal_id = (
            f"LX-RG-"
            f"{'B' if side == 'BUY' else 'S'}-"
            f"{now:%Y%m%d-%H%M}-"
            f"{uuid.uuid4().hex[:5].upper()}"
        )

        reasons = list(
            result.get(
                "reasons",
                [],
            )
        )

        reasons.append(
            f"Current M5 impulse: {impulse}."
        )

        reasons.append(
            "Range engine confirms mean-reversion conditions."
        )

        reasons.append(
            "Trade is being taken at the range edge rather than the middle."
        )

        return Signal(
            signal_id=signal_id,
            created_at=now.isoformat(),
            symbol=self.symbol,
            setup_type="RANGE_REVERSION",
            setup_name="Range Reversal",
            side=side,
            score=score,
            grade=risk_decision.grade,
            quality=risk_decision.quality,
            trend_context=context,
            impulse=impulse,
            reference_level=round(
                reference,
                2,
            ),
            level_strength=safe_int(
                result.get(
                    "level_strength",
                    0,
                )
            ),
            level_touches=safe_int(
                result.get(
                    "level_touches",
                    0,
                )
            ),
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
            rr=calculate_rr(
                entry_mid,
                stop_loss,
                tp2,
            ),
            lot_size=risk_decision.lot_size,
            reasons=reasons,
            invalidation=(
                "M15 close outside the range invalidates "
                "the reversal thesis. A clean breakout "
                "cancels the range-reversion setup."
            ),
        )

    # --------------------------------------------------------
    # MAIN ANALYSIS ENGINE
    # --------------------------------------------------------

    def analyse(
        self,
        candles=None,
        h4=None,
        h1=None,
        m15=None,
        m5=None,
    ):
        """
        Main analysis interface.

        Supports:

            model.analyse(candles={
                "H4": h4,
                "H1": h1,
                "M15": m15,
                "M5": m5,
            })

        and:

            model.analyse(
                h4=h4,
                h1=h1,
                m15=m15,
                m5=m5,
            )

        This compatibility is intentional so main.py cannot
        fail because of a candles keyword mismatch.
        """

        # ====================================================
        # NORMALISE CANDLE INPUT
        # ====================================================

        if candles is not None:

            if not isinstance(
                candles,
                dict,
            ):
                raise TypeError(
                    "candles must be a dictionary."
                )

            h4 = (
                candles.get("H4")
                or candles.get("h4")
                or candles.get("4h")
            )

            h1 = (
                candles.get("H1")
                or candles.get("h1")
                or candles.get("1h")
            )

            m15 = (
                candles.get("M15")
                or candles.get("m15")
                or candles.get("15min")
            )

            m5 = (
                candles.get("M5")
                or candles.get("m5")
                or candles.get("5min")
            )

        # ====================================================
        # VALIDATE
        # ====================================================

        if h4 is None:
            raise ValueError(
                "H4 candles missing."
            )

        if h1 is None:
            raise ValueError(
                "H1 candles missing."
            )

        if m15 is None:
            raise ValueError(
                "M15 candles missing."
            )

        if m5 is None:
            raise ValueError(
                "M5 candles missing."
            )

        # ====================================================
        # ENRICH
        # ====================================================

        h4 = enrich(
            h4.copy()
        )

        h1 = enrich(
            h1.copy()
        )

        m15 = enrich(
            m15.copy()
        )

        m5 = enrich(
            m5.copy()
        )

        if min(
            len(h4),
            len(h1),
            len(m15),
            len(m5),
        ) < 100:
            return None

        atr_value = safe_float(
            m5["atr"].iloc[-1]
        )

        if atr_value <= 0:
            return None

        # ====================================================
        # MARKET STATE
        # ====================================================

        impulse = current_impulse(
            m5
        )

        context = self.trend_context(
            h4,
            h1,
            m15,
        )

        candidates = []

        # ====================================================
        # TREND
        # ====================================================

        trend_candidate = self.trend_signal(
            h4=h4,
            h1=h1,
            m15=m15,
            m5=m5,
            atr_value=atr_value,
            context=context,
        )

        if trend_candidate is not None:

            candidates.append(
                (
                    "TREND",
                    trend_candidate,
                )
            )

        # ====================================================
        # SUPPORT / RESISTANCE
        # ====================================================

        try:

            h1_levels = build_levels(
                h1,
                atr_value,
                "H1",
            )

            m15_levels = build_levels(
                m15,
                atr_value,
                "M15",
            )

            levels = combine_levels(
                h1_levels,
                m15_levels,
                atr_value,
            )

        except Exception:

            levels = []

        # ====================================================
        # SUPPORT BOUNCE
        # ====================================================

        try:

            support = detect_support_bounce(
                m5=m5,
                m15=m15,
                h1=h1,
                levels=levels,
                atr_value=atr_value,
            )

            if (
                support is not None
                and support.detected
                and support.score >= SR_MIN_SCORE
            ):

                candidates.append(
                    (
                        "SUPPORT_BOUNCE",
                        {
                            "side": support.side,
                            "score": support.score,
                            "reference_level": (
                                support.level.price
                                if support.level
                                else None
                            ),
                            "level_strength": (
                                support.level.strength
                                if support.level
                                else 0
                            ),
                            "level_touches": (
                                support.level.touches
                                if support.level
                                else 0
                            ),
                            "reasons": support.reasons,
                        },
                    )
                )

        except Exception:
            pass

        # ====================================================
        # RESISTANCE BOUNCE
        # ====================================================

        try:

            resistance = detect_resistance_bounce(
                m5=m5,
                m15=m15,
                h1=h1,
                levels=levels,
                atr_value=atr_value,
            )

            if (
                resistance is not None
                and resistance.detected
                and resistance.score >= SR_MIN_SCORE
            ):

                candidates.append(
                    (
                        "RESISTANCE_BOUNCE",
                        {
                            "side": resistance.side,
                            "score": resistance.score,
                            "reference_level": (
                                resistance.level.price
                                if resistance.level
                                else None
                            ),
                            "level_strength": (
                                resistance.level.strength
                                if resistance.level
                                else 0
                            ),
                            "level_touches": (
                                resistance.level.touches
                                if resistance.level
                                else 0
                            ),
                            "reasons": resistance.reasons,
                        },
                    )
                )

        except Exception:
            pass

        # ====================================================
        # RANGE
        # ====================================================

        range_info = self.range_regime(
            m15=m15,
            h1=h1,
            atr_value=atr_value,
        )

        if range_info is not None:

            if not self.range_breaking(
                m15=m15,
                atr_value=atr_value,
                range_info=range_info,
            ):

                range_result = self.range_signal(
                    m5=m5,
                    m15=m15,
                    h1=h1,
                    atr_value=atr_value,
                    range_info=range_info,
                )

                if range_result is not None:

                    range_result[
                        "range_high"
                    ] = range_info["high"]

                    range_result[
                        "range_low"
                    ] = range_info["low"]

                    candidates.append(
                        (
                            "RANGE",
                            range_result,
                        )
                    )

        # ====================================================
        # BREAK & RETEST
        # ====================================================

        try:

            break_retest = detect_break_retest(
                m5=m5,
                m15=m15,
                h1=h1,
                levels=levels,
                atr_value=atr_value,
            )

            if (
                break_retest is not None
                and break_retest.detected
                and break_retest.score >= BREAK_RETEST_MIN_SCORE
            ):

                candidates.append(
                    (
                        "BREAK_RETEST",
                        {
                            "side": break_retest.side,
                            "score": break_retest.score,
                            "reference_level": (
                                break_retest.level.price
                                if break_retest.level
                                else None
                            ),
                            "level_strength": (
                                break_retest.level.strength
                                if break_retest.level
                                else 0
                            ),
                            "level_touches": (
                                break_retest.level.touches
                                if break_retest.level
                                else 0
                            ),
                            "reasons": break_retest.reasons,
                        },
                    )
                )

        except Exception:
            pass

        # ====================================================
        # NOTHING QUALIFIED
        # ====================================================

        if not candidates:
            return None

        # ====================================================
        # PRIORITY
        # ====================================================

        priority = {
            "RANGE": 5,
            "SUPPORT_BOUNCE": 4,
            "RESISTANCE_BOUNCE": 4,
            "BREAK_RETEST": 3,
            "TREND": 2,
        }

        candidates.sort(
            key=lambda item: (
                safe_int(
                    item[1].get(
                        "score",
                        0,
                    )
                ),
                priority.get(
                    item[0],
                    0,
                ),
            ),
            reverse=True,
        )

        best_type, best_result = candidates[0]

        # ====================================================
        # FINAL SIGNAL
        # ====================================================

        if best_type == "RANGE":

            return self.create_range_signal(
                result=best_result,
                m5=m5,
                m15=m15,
                h1=h1,
                atr_value=atr_value,
                context=context,
                impulse=impulse,
            )

        return self.create_signal(
            result=best_result,
            setup_type=best_type,
            m5=m5,
            m15=m15,
            h1=h1,
            atr_value=atr_value,
            context=context,
            impulse=impulse,
        )

    # --------------------------------------------------------
    # TELEGRAM FORMAT
    # --------------------------------------------------------

    def format_signal(
        self,
        signal: Signal,
    ) -> str:

        if signal.side == "BUY":

            side_icon = "🟢"

            direction = (
                "BUY"
            )

            side_tag = (
                "#LX_BUY"
            )

        else:

            side_icon = "🔴"

            direction = (
                "SELL"
            )

            side_tag = (
                "#LX_SELL"
            )

        headers = {
            "TREND": (
                "📈🧠 "
                "<b>LumiAj-XAU TREND SIGNAL</b>"
            ),

            "SUPPORT_BOUNCE": (
                "🟢🧠 "
                "<b>LumiAj-XAU SUPPORT BOUNCE</b>"
            ),

            "RESISTANCE_BOUNCE": (
                "🔴🧠 "
                "<b>LumiAj-XAU RESISTANCE BOUNCE</b>"
            ),

            "BREAK_RETEST": (
                "🔄🧠 "
                "<b>LumiAj-XAU BREAK &amp; RETEST</b>"
            ),

            "RANGE_REVERSION": (
                "↔️🧠 "
                "<b>LumiAj-XAU RANGE REVERSAL</b>"
            ),
        }

        header = headers.get(
            signal.setup_type,
            "🧠 <b>LumiAj-XAU SIGNAL</b>",
        )

        reason_text = "\n".join(
            f"• {reason}"
            for reason in signal.reasons
        )

        setup_tag = {
            "TREND": "#LX_TREND",
            "SUPPORT_BOUNCE": "#LX_SUPPORT_BOUNCE",
            "RESISTANCE_BOUNCE": "#LX_RESISTANCE_BOUNCE",
            "BREAK_RETEST": "#LX_BREAK_RETEST",
            "RANGE_REVERSION": "#LX_RANGE",
        }.get(
            signal.setup_type,
            "#LUMIAJ_XAU",
        )

        return (
            f"{header}\n\n"

            f"{side_icon} "
            f"<b>{direction} {signal.symbol}</b>\n\n"

            f"<b>Signal ID:</b> "
            f"<code>{signal.signal_id}</code>\n\n"

            f"<b>Score:</b> "
            f"{signal.score}/100\n"

            f"<b>Grade:</b> "
            f"{signal.grade}\n"

            f"<b>Quality:</b> "
            f"{signal.quality}\n\n"

            f"<b>Trend Context:</b> "
            f"{signal.trend_context}\n"

            f"<b>Current M5 Impulse:</b> "
            f"{signal.impulse}\n\n"

            f"<b>Reference:</b> "
            f"{signal.reference_level:.2f}\n"

            f"<b>Entry:</b> "
            f"{signal.entry_low:.2f} – "
            f"{signal.entry_high:.2f}\n"

            f"<b>Entry Mid:</b> "
            f"{signal.entry_mid:.2f}\n\n"

            f"<b>Stop Loss:</b> "
            f"{signal.stop_loss:.2f}\n"

            f"<b>TP1:</b> "
            f"{signal.tp1:.2f}\n"

            f"<b>TP2:</b> "
            f"{signal.tp2:.2f}\n"

            f"<b>TP3:</b> "
            f"{signal.tp3:.2f}\n\n"

            f"<b>RR to TP2:</b> "
            f"{signal.rr:.2f}R\n"

            f"<b>Model Lot:</b> "
            f"{signal.lot_size:.2f}\n\n"

            f"🧠 <b>WHY LumiAj TRIGGERED</b>\n"
            f"{reason_text}\n\n"

            f"⚠️ <b>INVALIDATION</b>\n"
            f"{signal.invalidation}\n\n"

            f"🛡 <b>ANTI-CHASE:</b> "
            f"Protected\n\n"

            f"{setup_tag} "
            f"{side_tag} "
            f"#LUMIAJ_XAU"
        )


# ============================================================
# MODULE-LEVEL FORMAT FUNCTION
# ============================================================

def format_signal(
    signal: Signal,
) -> str:

    model = GerkoXAUModel(
        symbol=signal.symbol
    )

    return model.format_signal(
        signal
    )