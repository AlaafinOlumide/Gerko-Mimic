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
# ENVIRONMENT / THRESHOLDS
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
# SIGNAL MODEL
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
    level_strength: float
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
# BASIC INDICATORS
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

    value = 100 - (100 / (1 + rs))

    return value.fillna(50)


def true_range(df: pd.DataFrame) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    previous_close = close.shift(1)

    ranges = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    )

    return ranges.max(axis=1)


def atr(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.Series:
    return true_range(df).rolling(
        period,
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
    close = df["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) & (up_move > 0),
            up_move,
            0,
        ),
        index=df.index,
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) & (down_move > 0),
            down_move,
            0,
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
            min_periods=period,
        ).mean()
        / atr_value.replace(0, np.nan)
    )

    minus_di = (
        100
        * minus_dm.ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
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

    df = df.copy()

    df.columns = [
        str(column).lower().strip()
        for column in df.columns
    ]

    required = [
        "open",
        "high",
        "low",
        "close",
    ]

    for column in required:
        if column not in df.columns:
            raise ValueError(
                f"Missing required candle column: {column}"
            )

    for column in required:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=required
    ).reset_index(drop=True)

    df["ema20"] = ema(
        df["close"],
        20,
    )

    df["ema50"] = ema(
        df["close"],
        50,
    )

    df["ema100"] = ema(
        df["close"],
        100,
    )

    df["rsi"] = rsi(
        df["close"],
        14,
    )

    df["atr"] = atr(
        df,
        14,
    )

    df["macd_hist"] = macd_histogram(
        df["close"]
    )

    df["adx"] = adx(
        df,
        14,
    )

    return df


# ============================================================
# RISK / RR
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

    return reward / risk


# ============================================================
# MARKET STRUCTURE HELPERS
# ============================================================

def _last(df: pd.DataFrame, column: str, default=0.0):

    if df is None:
        return default

    if column not in df.columns:
        return default

    value = df[column].iloc[-1]

    if pd.isna(value):
        return default

    return float(value)


def current_impulse(
    m5: pd.DataFrame,
) -> str:

    if m5 is None or len(m5) < 10:
        return "NEUTRAL"

    close = _last(m5, "close")
    ema20_value = _last(m5, "ema20")
    ema50_value = _last(m5, "ema50")

    recent_close = float(
        m5["close"].iloc[-3]
    )

    momentum = close - recent_close

    atr_value = _last(
        m5,
        "atr",
        0.0,
    )

    if atr_value <= 0:
        return "NEUTRAL"

    if (
        close > ema20_value
        and ema20_value > ema50_value
        and momentum > atr_value * 0.10
    ):
        return "BUY"

    if (
        close < ema20_value
        and ema20_value < ema50_value
        and momentum < -atr_value * 0.10
    ):
        return "SELL"

    return "NEUTRAL"


def trigger_exhausted(
    m5: pd.DataFrame,
    side: str,
) -> bool:

    if m5 is None or len(m5) < 20:
        return False

    candle = m5.iloc[-1]

    candle_range = (
        float(candle["high"])
        - float(candle["low"])
    )

    atr_value = _last(
        m5,
        "atr",
        0.0,
    )

    if atr_value <= 0:
        return False

    if candle_range < atr_value * 1.35:
        return False

    body = abs(
        float(candle["close"])
        - float(candle["open"])
    )

    if side == "BUY":
        upper_wick = (
            float(candle["high"])
            - max(
                float(candle["open"]),
                float(candle["close"]),
            )
        )

        if (
            upper_wick > body * 0.75
            and float(candle["close"])
            < float(candle["high"])
        ):
            return True

    if side == "SELL":
        lower_wick = (
            min(
                float(candle["open"]),
                float(candle["close"]),
            )
            - float(candle["low"])
        )

        if (
            lower_wick > body * 0.75
            and float(candle["close"])
            > float(candle["low"])
        ):
            return True

    return False


def chase_distance(
    price: float,
    reference: float,
    atr_value: float,
) -> float:

    if atr_value <= 0:
        return 999.0

    return abs(price - reference) / atr_value


# ============================================================
# MAIN MODEL
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
        h1: pd.DataFrame,
        m15: pd.DataFrame,
        m5: pd.DataFrame,
    ) -> str:

        h1_close = _last(h1, "close")
        h1_ema50 = _last(h1, "ema50")
        h1_ema100 = _last(h1, "ema100")

        m15_close = _last(m15, "close")
        m15_ema50 = _last(m15, "ema50")

        m5_close = _last(m5, "close")
        m5_ema20 = _last(m5, "ema20")
        m5_ema50 = _last(m5, "ema50")

        bullish = (
            h1_close > h1_ema50
            and h1_ema50 > h1_ema100
            and m15_close > m15_ema50
            and m5_close > m5_ema20
            and m5_ema20 >= m5_ema50
        )

        bearish = (
            h1_close < h1_ema50
            and h1_ema50 < h1_ema100
            and m15_close < m15_ema50
            and m5_close < m5_ema20
            and m5_ema20 <= m5_ema50
        )

        if bullish:
            return "BULLISH"

        if bearish:
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
        context: str,
        impulse: str,
    ):

        if context == "NEUTRAL":
            return None

        if impulse not in ("BUY", "SELL"):
            return None

        side = (
            "BUY"
            if context == "BULLISH"
            else "SELL"
        )

        # Current M5 impulse must agree.
        if side != impulse:
            return None

        price = _last(m5, "close")
        atr_value = _last(m5, "atr")

        if atr_value <= 0:
            return None

        ema20_value = _last(
            m5,
            "ema20",
        )

        distance = abs(
            price - ema20_value
        ) / atr_value

        # Do not chase an already-expanded move.
        if distance > MAX_TREND_CHASE_ATR:
            return None

        if trigger_exhausted(
            m5,
            side,
        ):
            return None

        score = 0
        reasons = []

        h1_close = _last(h1, "close")
        h1_ema50 = _last(h1, "ema50")
        h1_ema100 = _last(h1, "ema100")

        m15_close = _last(m15, "close")
        m15_ema20 = _last(m15, "ema20")
        m15_ema50 = _last(m15, "ema50")

        m5_rsi = _last(m5, "rsi")
        m5_macd = _last(m5, "macd_hist")

        if side == "BUY":

            if h1_close > h1_ema50:
                score += 15
                reasons.append(
                    "H1 price above EMA50"
                )

            if h1_ema50 > h1_ema100:
                score += 12
                reasons.append(
                    "H1 EMA structure bullish"
                )

            if m15_close > m15_ema50:
                score += 12
                reasons.append(
                    "M15 price above EMA50"
                )

            if m15_ema20 > m15_ema50:
                score += 8
                reasons.append(
                    "M15 short-term structure bullish"
                )

            if 50 <= m5_rsi <= 68:
                score += 12
                reasons.append(
                    "M5 RSI supports continuation"
                )

            if m5_macd > 0:
                score += 10
                reasons.append(
                    "M5 MACD momentum positive"
                )

            score += 10
            reasons.append(
                "Current M5 impulse confirms BUY"
            )

        else:

            if h1_close < h1_ema50:
                score += 15
                reasons.append(
                    "H1 price below EMA50"
                )

            if h1_ema50 < h1_ema100:
                score += 12
                reasons.append(
                    "H1 EMA structure bearish"
                )

            if m15_close < m15_ema50:
                score += 12
                reasons.append(
                    "M15 price below EMA50"
                )

            if m15_ema20 < m15_ema50:
                score += 8
                reasons.append(
                    "M15 short-term structure bearish"
                )

            if 32 <= m5_rsi <= 50:
                score += 12
                reasons.append(
                    "M5 RSI supports continuation"
                )

            if m5_macd < 0:
                score += 10
                reasons.append(
                    "M5 MACD momentum negative"
                )

            score += 10
            reasons.append(
                "Current M5 impulse confirms SELL"
            )

        score = min(score, 100)

        if score < TREND_MIN_SCORE:
            return None

        if side == "BUY":

            entry = price

            stop = min(
                float(m5["low"].tail(8).min()),
                entry - atr_value * 1.15,
            )

            risk = entry - stop

            if risk <= 0:
                return None

            tp1 = entry + risk * 1.2
            tp2 = entry + risk * 2.0
            tp3 = entry + risk * 3.0

        else:

            entry = price

            stop = max(
                float(m5["high"].tail(8).max()),
                entry + atr_value * 1.15,
            )

            risk = stop - entry

            if risk <= 0:
                return None

            tp1 = entry - risk * 1.2
            tp2 = entry - risk * 2.0
            tp3 = entry - risk * 3.0

        return {
            "setup_type": "TC",
            "setup_name": "TREND CONTINUATION",
            "side": side,
            "score": int(score),
            "reference_level": entry,
            "level_strength": 0.0,
            "level_touches": 0,
            "entry_low": entry,
            "entry_high": entry,
            "entry_mid": entry,
            "stop_loss": stop,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp3,
            "reasons": reasons,
            "invalidation": (
                "M5 structure must remain aligned with "
                "the signal direction."
            ),
        }

    # --------------------------------------------------------
    # RANGE REGIME
    # --------------------------------------------------------

    def range_regime(
        self,
        h1: pd.DataFrame,
        m15: pd.DataFrame,
        m5: pd.DataFrame,
    ):

        if (
            h1 is None
            or m15 is None
            or m5 is None
        ):
            return None

        if len(m15) < 48:
            return None

        atr_value = _last(
            m15,
            "atr",
            0.0,
        )

        h1_atr = _last(
            h1,
            "atr",
            atr_value,
        )

        if atr_value <= 0:
            return None

        recent = m15.tail(48)

        range_high = float(
            recent["high"].max()
        )

        range_low = float(
            recent["low"].min()
        )

        width = range_high - range_low

        width_atr = width / atr_value

        if not (
            RANGE_MIN_WIDTH_ATR
            <= width_atr
            <= RANGE_MAX_WIDTH_ATR
        ):
            return None

        h1_adx = _last(
            h1,
            "adx",
            99.0,
        )

        m15_adx = _last(
            m15,
            "adx",
            99.0,
        )

        if h1_adx > RANGE_ADX_MAX:
            return None

        if m15_adx > 23:
            return None

        h1_ema50_slope = abs(
            _last(h1, "ema50")
            - float(h1["ema50"].iloc[-6])
        )

        m15_ema50_slope = abs(
            _last(m15, "ema50")
            - float(m15["ema50"].iloc[-6])
        )

        if h1_ema50_slope > h1_atr * 0.35:
            return None

        if m15_ema50_slope > atr_value * 0.45:
            return None

        price = _last(
            m5,
            "close",
        )

        buffer = (
            atr_value
            * RANGE_BREAK_BUFFER_ATR
        )

        if price > range_high + buffer:
            return None

        if price < range_low - buffer:
            return None

        return {
            "high": range_high,
            "low": range_low,
            "width": width,
            "width_atr": width_atr,
            "atr": atr_value,
        }

    # --------------------------------------------------------
    # RANGE SIGNAL
    # --------------------------------------------------------

    def range_signal(
        self,
        regime,
        h1,
        m15,
        m5,
        signal_time=None,
    ):

        if regime is None:
            return None

        price = _last(
            m5,
            "close",
        )

        atr_value = regime["atr"]

        range_high = regime["high"]
        range_low = regime["low"]

        distance_low = abs(
            price - range_low
        )

        distance_high = abs(
            range_high - price
        )

        low_edge = (
            distance_low / atr_value
            <= RANGE_EDGE_ATR
        )

        high_edge = (
            distance_high / atr_value
            <= RANGE_EDGE_ATR
        )

        if not low_edge and not high_edge:
            return None

        # Suppress range reversals when the market is
        # actually breaking out.
        buffer = (
            atr_value
            * RANGE_BREAK_BUFFER_ATR
        )

        if (
            price > range_high + buffer
            or price < range_low - buffer
        ):
            return None

        impulse = current_impulse(m5)

        if low_edge:

            side = "BUY"

            if impulse == "SELL":
                return None

            reference = range_low

            room = (
                range_high - price
            )

        else:

            side = "SELL"

            if impulse == "BUY":
                return None

            reference = range_high

            room = (
                price - range_low
            )

        if room < atr_value * 1.2:
            return None

        score = 76
        reasons = [
            "Low-volatility range regime",
            "Price at range edge",
            "Adequate room toward opposite range edge",
        ]

        if impulse == side:
            score += 8
            reasons.append(
                "Current M5 impulse agrees with range reversal"
            )

        rsi_value = _last(
            m5,
            "rsi",
            50,
        )

        if side == "BUY" and rsi_value < 45:
            score += 5
            reasons.append(
                "M5 RSI supports lower-range reversal"
            )

        if side == "SELL" and rsi_value > 55:
            score += 5
            reasons.append(
                "M5 RSI supports upper-range reversal"
            )

        score = min(
            score,
            100,
        )

        if score < RANGE_MIN_SCORE:
            return None

        entry = price

        if side == "BUY":

            stop = range_low - (
                atr_value * 0.35
            )

            risk = entry - stop

            tp1 = entry + risk * 1.2
            tp2 = entry + risk * 2.0
            tp3 = min(
                range_high,
                entry + risk * 3.0,
            )

        else:

            stop = range_high + (
                atr_value * 0.35
            )

            risk = stop - entry

            tp1 = entry - risk * 1.2
            tp2 = entry - risk * 2.0
            tp3 = max(
                range_low,
                entry - risk * 3.0,
            )

        if risk <= 0:
            return None

        return {
            "setup_type": "RG",
            "setup_name": "RANGE EDGE REVERSAL",
            "side": side,
            "score": int(score),
            "reference_level": reference,
            "level_strength": 0.0,
            "level_touches": 0,
            "entry_low": entry,
            "entry_high": entry,
            "entry_mid": entry,
            "stop_loss": stop,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp3,
            "reasons": reasons,
            "invalidation": (
                "Range break beyond the protected edge "
                "invalidates the reversal."
            ),
        }

    # --------------------------------------------------------
    # SUPPORT / RESISTANCE
    # --------------------------------------------------------

    def sr_signal(
        self,
        result,
        m5,
        m15,
        h1,
        context,
        impulse,
    ):

        if result is None:
            return None

        detected = getattr(
            result,
            "detected",
            False,
        )

        if not detected:
            return None

        side = getattr(
            result,
            "side",
            None,
        )

        if side not in (
            "BUY",
            "SELL",
        ):
            return None

        score = int(
            getattr(
                result,
                "score",
                0,
            )
        )

        reasons = list(
            getattr(
                result,
                "reasons",
                [],
            )
            or []
        )

        level = float(
            getattr(
                result,
                "level",
                _last(m5, "close"),
            )
        )

        price = _last(
            m5,
            "close",
        )

        atr_value = _last(
            m5,
            "atr",
            0.0,
        )

        if atr_value <= 0:
            return None

        # --------------------------------------------
        # Countertrend S/R needs much stronger evidence.
        # --------------------------------------------

        countertrend = False

        if context == "BULLISH" and side == "SELL":
            countertrend = True

        if context == "BEARISH" and side == "BUY":
            countertrend = True

        if impulse in ("BUY", "SELL") and side != impulse:
            countertrend = True

        minimum = (
            COUNTERTREND_SR_MIN_SCORE
            if countertrend
            else SR_MIN_SCORE
        )

        if score < minimum:
            return None

        # --------------------------------------------
        # Anti-chase
        # --------------------------------------------

        distance = chase_distance(
            price,
            level,
            atr_value,
        )

        if distance > MAX_SR_CHASE_ATR:
            return None

        # --------------------------------------------
        # Current impulse veto
        # --------------------------------------------

        if (
            impulse in ("BUY", "SELL")
            and impulse != side
            and not countertrend
        ):
            return None

        if trigger_exhausted(
            m5,
            side,
        ):
            return None

        entry = price

        if side == "BUY":

            recent_low = float(
                m5["low"].tail(10).min()
            )

            stop = min(
                recent_low,
                level - atr_value * 0.25,
            )

            risk = entry - stop

            if risk <= 0:
                return None

            tp1 = entry + risk * 1.2
            tp2 = entry + risk * 2.0
            tp3 = entry + risk * 3.0

        else:

            recent_high = float(
                m5["high"].tail(10).max()
            )

            stop = max(
                recent_high,
                level + atr_value * 0.25,
            )

            risk = stop - entry

            if risk <= 0:
                return None

            tp1 = entry - risk * 1.2
            tp2 = entry - risk * 2.0
            tp3 = entry - risk * 3.0

        if countertrend:
            reasons.insert(
                0,
                "Countertrend filter passed with elevated confirmation",
            )

        return {
            "setup_type": (
                "SB"
                if side == "BUY"
                else "RB"
            ),
            "setup_name": (
                "SUPPORT BOUNCE"
                if side == "BUY"
                else "RESISTANCE BOUNCE"
            ),
            "side": side,
            "score": score,
            "reference_level": level,
            "level_strength": float(
                getattr(
                    result,
                    "strength",
                    0.0,
                )
                or 0.0
            ),
            "level_touches": int(
                getattr(
                    result,
                    "touches",
                    0,
                )
                or 0
            ),
            "entry_low": entry,
            "entry_high": entry,
            "entry_mid": entry,
            "stop_loss": stop,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp3,
            "reasons": reasons,
            "invalidation": (
                "The protected support/resistance level "
                "must hold."
            ),
        }

    # --------------------------------------------------------
    # BREAK / RETEST
    # --------------------------------------------------------

    def break_retest_signal(
        self,
        result,
        m5,
        context,
        impulse,
    ):

        if result is None:
            return None

        detected = getattr(
            result,
            "detected",
            False,
        )

        if not detected:
            return None

        side = getattr(
            result,
            "side",
            None,
        )

        if side not in (
            "BUY",
            "SELL",
        ):
            return None

        # Break/retest must agree with current impulse.
        if (
            impulse in ("BUY", "SELL")
            and side != impulse
        ):
            return None

        # Break/retest must also agree with broad trend
        # unless the signal has unusually strong confirmation.
        if context == "BULLISH" and side == "SELL":
            return None

        if context == "BEARISH" and side == "BUY":
            return None

        score = int(
            getattr(
                result,
                "score",
                0,
            )
        )

        if score < BREAK_RETEST_MIN_SCORE:
            return None

        price = _last(
            m5,
            "close",
        )

        atr_value = _last(
            m5,
            "atr",
            0.0,
        )

        level = float(
            getattr(
                result,
                "level",
                price,
            )
        )

        if atr_value <= 0:
            return None

        if chase_distance(
            price,
            level,
            atr_value,
        ) > MAX_TREND_CHASE_ATR:
            return None

        if trigger_exhausted(
            m5,
            side,
        ):
            return None

        reasons = list(
            getattr(
                result,
                "reasons",
                [],
            )
            or []
        )

        reasons.insert(
            0,
            "Break/retest agrees with current M5 impulse",
        )

        entry = price

        if side == "BUY":

            stop = level - (
                atr_value * 0.45
            )

            risk = entry - stop

            if risk <= 0:
                return None

            tp1 = entry + risk * 1.2
            tp2 = entry + risk * 2.0
            tp3 = entry + risk * 3.0

        else:

            stop = level + (
                atr_value * 0.45
            )

            risk = stop - entry

            if risk <= 0:
                return None

            tp1 = entry - risk * 1.2
            tp2 = entry - risk * 2.0
            tp3 = entry - risk * 3.0

        return {
            "setup_type": "BR",
            "setup_name": "BREAK & RETEST",
            "side": side,
            "score": score,
            "reference_level": level,
            "level_strength": 0.0,
            "level_touches": 0,
            "entry_low": entry,
            "entry_high": entry,
            "entry_mid": entry,
            "stop_loss": stop,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp3,
            "reasons": reasons,
            "invalidation": (
                "The broken level must hold on the retest."
            ),
        }

    # --------------------------------------------------------
    # CREATE SIGNAL OBJECT
    # --------------------------------------------------------

    def create_signal(
        self,
        candidate,
        signal_time=None,
    ) -> Optional[Signal]:

        if candidate is None:
            return None

        score = int(
            candidate["score"]
        )

        setup_type = candidate[
            "setup_type"
        ]

        if setup_type == "TC":
            minimum = TREND_MIN_SCORE

        elif setup_type in (
            "SB",
            "RB",
        ):
            minimum = SR_MIN_SCORE

            if score >= COUNTERTREND_SR_MIN_SCORE:
                minimum = SR_MIN_SCORE

        elif setup_type == "BR":
            minimum = BREAK_RETEST_MIN_SCORE

        elif setup_type == "RG":
            minimum = RANGE_MIN_SCORE

        else:
            minimum = 76

        decision = classify_signal(
            score,
            minimum,
        )

        if not decision.allowed:
            return None

        if signal_time is None:
            created = datetime.now()
        elif isinstance(
            signal_time,
            datetime,
        ):
            created = signal_time
        else:
            try:
                created = pd.Timestamp(
                    signal_time
                ).to_pydatetime()
            except Exception:
                created = datetime.now()

        timestamp = created.strftime(
            "%Y%m%d-%H%M"
        )

        random_part = uuid.uuid4().hex[:4].upper()

        signal_id = (
            f"LX-{setup_type}-"
            f"{candidate['side'][0]}-"
            f"{timestamp}-"
            f"{random_part}"
        )

        entry_mid = float(
            candidate["entry_mid"]
        )

        rr = calculate_rr(
            entry_mid,
            float(candidate["stop_loss"]),
            float(candidate["tp2"]),
        )

        return Signal(
            signal_id=signal_id,
            created_at=created.isoformat(
                timespec="seconds"
            ),
            symbol=self.symbol,
            setup_type=setup_type,
            setup_name=candidate[
                "setup_name"
            ],
            side=candidate["side"],
            score=score,
            grade=decision.grade,
            quality=decision.quality,
            trend_context=candidate.get(
                "trend_context",
                "NEUTRAL",
            ),
            impulse=candidate.get(
                "impulse",
                "NEUTRAL",
            ),
            reference_level=float(
                candidate[
                    "reference_level"
                ]
            ),
            level_strength=float(
                candidate.get(
                    "level_strength",
                    0.0,
                )
            ),
            level_touches=int(
                candidate.get(
                    "level_touches",
                    0,
                )
            ),
            entry_low=float(
                candidate["entry_low"]
            ),
            entry_high=float(
                candidate["entry_high"]
            ),
            entry_mid=entry_mid,
            stop_loss=float(
                candidate["stop_loss"]
            ),
            tp1=float(
                candidate["tp1"]
            ),
            tp2=float(
                candidate["tp2"]
            ),
            tp3=float(
                candidate["tp3"]
            ),
            rr=rr,
            lot_size=decision.lot_size,
            reasons=list(
                candidate.get(
                    "reasons",
                    [],
                )
            ),
            invalidation=candidate.get(
                "invalidation",
                "Setup invalidated if market structure changes.",
            ),
        )

    # --------------------------------------------------------
    # RANGE SIGNAL CREATOR
    # --------------------------------------------------------

    def create_range_signal(
        self,
        candidate,
        signal_time=None,
    ):

        return self.create_signal(
            candidate,
            signal_time=signal_time,
        )

    # --------------------------------------------------------
    # ANALYSIS
    #
    # IMPORTANT:
    # signal_time is intentionally accepted here because
    # main.py passes it.
    #
    # **kwargs also prevents harmless future interface
    # mismatches.
    # --------------------------------------------------------

    def analyse(
        self,
        candles=None,
        h4=None,
        h1=None,
        m15=None,
        m5=None,
        signal_time=None,
        **kwargs,
    ):

        # ----------------------------------------------------
        # Compatibility with main.py
        # ----------------------------------------------------

        if candles is not None:

            if isinstance(candles, dict):

                h4 = (
                    candles.get("h4")
                    or candles.get("H4")
                    or h4
                )

                h1 = (
                    candles.get("h1")
                    or candles.get("H1")
                    or h1
                )

                m15 = (
                    candles.get("m15")
                    or candles.get("M15")
                    or m15
                )

                m5 = (
                    candles.get("m5")
                    or candles.get("M5")
                    or m5
                )

            elif isinstance(
                candles,
                pd.DataFrame,
            ):

                # If main passes a single dataframe,
                # treat it as M5.
                if m5 is None:
                    m5 = candles

        if any(
            frame is None
            for frame in (
                h1,
                m15,
                m5,
            )
        ):
            return None

        # ----------------------------------------------------
        # Enrich data
        # ----------------------------------------------------

        h1 = enrich(h1)
        m15 = enrich(m15)
        m5 = enrich(m5)

        if h4 is not None:
            h4 = enrich(h4)

        if len(m5) < 100:
            return None

        if len(m15) < 100:
            return None

        if len(h1) < 100:
            return None

        # ----------------------------------------------------
        # Current state
        # ----------------------------------------------------

        context = self.trend_context(
            h1,
            m15,
            m5,
        )

        impulse = current_impulse(
            m5
        )

        candidates = []

        # ----------------------------------------------------
        # 1. TREND CONTINUATION
        # ----------------------------------------------------

        trend = self.trend_signal(
            h4,
            h1,
            m15,
            m5,
            context,
            impulse,
        )

        if trend is not None:

            trend["trend_context"] = context
            trend["impulse"] = impulse

            candidates.append(
                trend
            )

        # ----------------------------------------------------
        # 2. SUPPORT / RESISTANCE
        # ----------------------------------------------------

        try:

            atr_value = _last(
                m5,
                "atr",
                0.0,
            )

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

            support = detect_support_bounce(
                m5,
                m15,
                h1,
                levels,
                atr_value,
            )

            support_signal = self.sr_signal(
                support,
                m5,
                m15,
                h1,
                context,
                impulse,
            )

            if support_signal is not None:

                support_signal[
                    "trend_context"
                ] = context

                support_signal[
                    "impulse"
                ] = impulse

                candidates.append(
                    support_signal
                )

            resistance = detect_resistance_bounce(
                m5,
                m15,
                h1,
                levels,
                atr_value,
            )

            resistance_signal = self.sr_signal(
                resistance,
                m5,
                m15,
                h1,
                context,
                impulse,
            )

            if resistance_signal is not None:

                resistance_signal[
                    "trend_context"
                ] = context

                resistance_signal[
                    "impulse"
                ] = impulse

                candidates.append(
                    resistance_signal
                )

        except Exception:
            # Do not allow an S/R engine issue to kill
            # the complete analysis cycle.
            pass

        # ----------------------------------------------------
        # 3. RANGE ENGINE
        # ----------------------------------------------------

        regime = self.range_regime(
            h1,
            m15,
            m5,
        )

        range_candidate = self.range_signal(
            regime,
            h1,
            m15,
            m5,
            signal_time=signal_time,
        )

        if range_candidate is not None:

            range_candidate[
                "trend_context"
            ] = context

            range_candidate[
                "impulse"
            ] = impulse

            candidates.append(
                range_candidate
            )

        # ----------------------------------------------------
        # 4. BREAK / RETEST
        # ----------------------------------------------------

        try:

            if "levels" in locals():

                break_result = detect_break_retest(
                    m5,
                    m15,
                    h1,
                    levels,
                    _last(
                        m5,
                        "atr",
                        0.0,
                    ),
                )

                break_candidate = (
                    self.break_retest_signal(
                        break_result,
                        m5,
                        context,
                        impulse,
                    )
                )

                if break_candidate is not None:

                    break_candidate[
                        "trend_context"
                    ] = context

                    break_candidate[
                        "impulse"
                    ] = impulse

                    candidates.append(
                        break_candidate
                    )

        except Exception:
            pass

        # ----------------------------------------------------
        # No candidates
        # ----------------------------------------------------

        if not candidates:
            return None

        # ----------------------------------------------------
        # Candidate ranking
        #
        # Range/SR gets priority only when score is comparable.
        # A high-quality continuation should not be beaten by
        # a mediocre reversal simply because of setup type.
        # ----------------------------------------------------

        priority = {
            "RG": 5,
            "SB": 4,
            "RB": 4,
            "BR": 3,
            "TC": 2,
        }

        candidates.sort(
            key=lambda candidate: (
                int(
                    candidate.get(
                        "score",
                        0,
                    )
                ),
                priority.get(
                    candidate.get(
                        "setup_type"
                    ),
                    0,
                ),
            ),
            reverse=True,
        )

        best = candidates[0]

        # ----------------------------------------------------
        # Final safety check
        # ----------------------------------------------------

        side = best["side"]

        if (
            side in ("BUY", "SELL")
            and impulse in ("BUY", "SELL")
        ):

            setup_type = best[
                "setup_type"
            ]

            # A continuation/breakout signal cannot fight
            # a strong immediate M5 impulse.
            if (
                setup_type in (
                    "TC",
                    "BR",
                )
                and side != impulse
            ):
                return None

        return self.create_signal(
            best,
            signal_time=signal_time,
        )

    # --------------------------------------------------------
    # TELEGRAM FORMATTER
    # --------------------------------------------------------

    def format_signal(
        self,
        signal: Signal,
    ) -> str:

        direction_icon = (
            "🟢"
            if signal.side == "BUY"
            else "🔴"
        )

        setup_hashtag = {
            "TC": "#LX_TREND",
            "SB": "#LX_SUPPORT_BOUNCE",
            "RB": "#LX_RESISTANCE_BOUNCE",
            "BR": "#LX_BREAK_RETEST",
            "RG": "#LX_RANGE",
        }.get(
            signal.setup_type,
            "#LUMIAJ_XAU",
        )

        reasons = "\n".join(
            f"• {reason}"
            for reason in signal.reasons[:7]
        )

        return (
            "━━━━━━━━━━━━━━━━━━━━\n"
            "🧠⚡ <b>LumiAj-XAU</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n\n"

            f"{direction_icon} "
            f"<b>{signal.side} XAU/USD</b>\n\n"

            f"🎯 <b>{signal.setup_name}</b>\n"
            f"📊 Score: <b>{signal.score}/100</b>\n"
            f"🏆 Grade: <b>{signal.grade}</b>\n"
            f"💎 {signal.quality}\n\n"

            f"💰 Entry: "
            f"<b>{signal.entry_low:.2f}"
            f" – "
            f"{signal.entry_high:.2f}</b>\n"

            f"🛑 Stop Loss: "
            f"<b>{signal.stop_loss:.2f}</b>\n\n"

            f"🎯 TP1: <b>{signal.tp1:.2f}</b>\n"
            f"🎯 TP2: <b>{signal.tp2:.2f}</b>\n"
            f"🎯 TP3: <b>{signal.tp3:.2f}</b>\n\n"

            f"📐 RR to TP2: "
            f"<b>1:{signal.rr:.2f}</b>\n"
            f"📦 Model Lot: <b>{signal.lot_size:.2f}</b>\n\n"

            f"🧭 Trend: <b>{signal.trend_context}</b>\n"
            f"⚡ M5 Impulse: <b>{signal.impulse}</b>\n\n"

            "<b>WHY LumiAj TRIGGERED</b>\n"
            f"{reasons}\n\n"

            f"⚠️ <b>Invalidation:</b>\n"
            f"{signal.invalidation}\n\n"

            f"🆔 {signal.signal_id}\n"
            f"{setup_hashtag} "
            f"#LX_{signal.side}\n"
            "#LUMIAJ_XAU"
        )


# ============================================================
# MODULE-LEVEL FORMATTER
#
# main.py can safely continue importing:
#
# from strategy import format_signal
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