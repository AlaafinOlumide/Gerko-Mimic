from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from risk_engine import classify_signal

from sr_engine import (
    build_levels,
    combine_levels,
    detect_break_retest,
    detect_resistance_bounce,
    detect_support_bounce,
)


# ============================================================
# CONFIG
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

# Range engine
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
# SIGNAL DATACLASS
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
# RISK / RR
# ============================================================

def calculate_rr(
    entry: float,
    stop: float,
    target: float,
    side: str,
) -> float:
    """
    Calculate reward-to-risk ratio.
    """

    side = side.upper().strip()

    if side == "BUY":
        risk = entry - stop
        reward = target - entry

    elif side == "SELL":
        risk = stop - entry
        reward = entry - target

    else:
        raise ValueError("side must be BUY or SELL")

    if risk <= 0 or reward <= 0:
        return 0.0

    return round(reward / risk, 2)


# ============================================================
# INDICATORS
# ============================================================

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(
        span=period,
        adjust=False,
        min_periods=period,
    ).mean()


def rsi(
    series: pd.Series,
    period: int = 14,
) -> pd.Series:

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
    close: pd.Series,
) -> pd.Series:

    fast = close.ewm(
        span=12,
        adjust=False,
    ).mean()

    slow = close.ewm(
        span=26,
        adjust=False,
    ).mean()

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

    data.columns = [
        str(column).lower()
        for column in data.columns
    ]

    required = {
        "open",
        "high",
        "low",
        "close",
    }

    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing OHLC columns: {sorted(missing)}"
        )

    for column in [
        "open",
        "high",
        "low",
        "close",
    ]:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

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

    return data.dropna(
        subset=[
            "ema20",
            "ema50",
            "rsi",
            "atr",
            "macd_histogram",
        ]
    )


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

    close = float(latest["close"])
    ema20_value = float(latest["ema20"])

    macd_now = float(
        latest["macd_histogram"]
    )

    macd_previous = float(
        previous["macd_histogram"]
    )

    rsi_now = float(latest["rsi"])
    rsi_previous = float(previous["rsi"])

    atr_value = float(latest["atr"])

    move = close - float(
        m5.iloc[-4]["close"]
    )

    bullish = (
        close > ema20_value
        and macd_now > macd_previous
        and rsi_now > rsi_previous
        and move > atr_value * 0.20
    )

    bearish = (
        close < ema20_value
        and macd_now < macd_previous
        and rsi_now < rsi_previous
        and move < -atr_value * 0.20
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
) -> bool:

    if len(m5) < 3:
        return False

    latest = m5.iloc[-1]

    candle_range = (
        float(latest["high"])
        - float(latest["low"])
    )

    atr_value = float(latest["atr"])

    if atr_value <= 0:
        return False

    body = abs(
        float(latest["close"])
        - float(latest["open"])
    )

    # Very large trigger candle.
    if candle_range >= atr_value * 1.8:
        return True

    # Large body with little remaining room.
    if body >= atr_value * 1.35:
        return True

    return False


def chase_distance(
    m5: pd.DataFrame,
) -> float:

    latest = m5.iloc[-1]

    return abs(
        float(latest["close"])
        - float(latest["ema20"])
    )


# ============================================================
# RANGE DETECTION
# ============================================================

def range_regime(
    m15: pd.DataFrame,
    h1: pd.DataFrame,
    atr_value: float,
) -> Optional[dict]:

    if len(m15) < 60 or len(h1) < 60:
        return None

    if atr_value <= 0:
        return None

    recent_m15 = m15.iloc[-48:]

    range_high = float(
        recent_m15["high"].max()
    )

    range_low = float(
        recent_m15["low"].min()
    )

    width = range_high - range_low

    if width <= 0:
        return None

    width_atr = width / atr_value

    if not (
        RANGE_MIN_WIDTH_ATR
        <= width_atr
        <= RANGE_MAX_WIDTH_ATR
    ):
        return None

    h1_adx = float(
        h1.iloc[-1]["adx"]
    )

    m15_adx = float(
        m15.iloc[-1]["adx"]
    )

    if h1_adx > RANGE_ADX_MAX:
        return None

    if m15_adx > RANGE_ADX_MAX + 3:
        return None

    h1_slope = abs(
        float(h1.iloc[-1]["ema50"])
        - float(h1.iloc[-6]["ema50"])
    )

    m15_slope = abs(
        float(m15.iloc[-1]["ema50"])
        - float(m15.iloc[-6]["ema50"])
    )

    if h1_slope > atr_value * 0.35:
        return None

    if m15_slope > atr_value * 0.45:
        return None

    price = float(
        m15.iloc[-1]["close"]
    )

    buffer = atr_value * RANGE_BREAK_BUFFER_ATR

    if price < range_low - buffer:
        return None

    if price > range_high + buffer:
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
    range_info: dict,
    atr_value: float,
) -> bool:

    if len(m15) < 3:
        return False

    buffer = (
        atr_value
        * RANGE_BREAK_BUFFER_ATR
    )

    latest = m15.iloc[-1]
    previous = m15.iloc[-2]

    high = range_info["high"]
    low = range_info["low"]

    latest_close = float(
        latest["close"]
    )

    previous_close = float(
        previous["close"]
    )

    latest_macd = float(
        latest["macd_histogram"]
    )

    previous_macd = float(
        previous["macd_histogram"]
    )

    bullish_break = (
        latest_close > high + buffer
        and previous_close > high
        and latest_macd > previous_macd
    )

    bearish_break = (
        latest_close < low - buffer
        and previous_close < low
        and latest_macd < previous_macd
    )

    return bool(
        bullish_break
        or bearish_break
    )


# ============================================================
# RANGE EDGE SIGNAL
# ============================================================

def range_signal(
    m15: pd.DataFrame,
    m5: pd.DataFrame,
    range_info: dict,
    atr_value: float,
) -> Optional[dict]:

    if len(m15) < 3 or len(m5) < 3:
        return None

    latest = m5.iloc[-1]
    previous = m5.iloc[-2]

    price = float(
        latest["close"]
    )

    range_high = range_info["high"]
    range_low = range_info["low"]

    edge_distance_low = (
        abs(price - range_low)
    )

    edge_distance_high = (
        abs(range_high - price)
    )

    near_support = (
        edge_distance_low
        <= atr_value * RANGE_EDGE_ATR
    )

    near_resistance = (
        edge_distance_high
        <= atr_value * RANGE_EDGE_ATR
    )

    if not near_support and not near_resistance:
        return None

    latest_open = float(
        latest["open"]
    )

    latest_high = float(
        latest["high"]
    )

    latest_low = float(
        latest["low"]
    )

    latest_close = float(
        latest["close"]
    )

    previous_high = float(
        previous["high"]
    )

    previous_low = float(
        previous["low"]
    )

    candle_range = (
        latest_high - latest_low
    )

    if candle_range <= 0:
        return None

    lower_wick = min(
        latest_open,
        latest_close
    ) - latest_low

    upper_wick = (
        latest_high
        - max(
            latest_open,
            latest_close,
        )
    )

    rsi_now = float(
        latest["rsi"]
    )

    rsi_previous = float(
        previous["rsi"]
    )

    macd_now = float(
        latest["macd_histogram"]
    )

    macd_previous = float(
        previous["macd_histogram"]
    )

    m15_latest = m15.iloc[-1]
    m15_previous = m15.iloc[-2]

    m15_rsi_now = float(
        m15_latest["rsi"]
    )

    m15_rsi_previous = float(
        m15_previous["rsi"]
    )

    # --------------------------------------------------------
    # SUPPORT REVERSION
    # --------------------------------------------------------

    if near_support:

        score = 0
        reasons = []

        # Price is actually holding the range low.
        score += 15
        reasons.append(
            "Price holding range support"
        )

        # Bullish rejection candle.
        bullish_rejection = (
            latest_close > latest_open
            and lower_wick >= candle_range * 0.30
        )

        if bullish_rejection:
            score += 10
            reasons.append(
                "Bullish M5 rejection"
            )

        # Reclaim previous candle high.
        if latest_close > previous_high:
            score += 15
            reasons.append(
                "M5 reclaimed prior high"
            )

        # RSI turning upward.
        if rsi_now > rsi_previous:
            score += 10
            reasons.append(
                "M5 RSI rising"
            )

        # MACD improving.
        if macd_now > macd_previous:
            score += 10
            reasons.append(
                "M5 MACD improving"
            )

        # M15 stabilising.
        if m15_rsi_now >= m15_rsi_previous:
            score += 5
            reasons.append(
                "M15 momentum stabilising"
            )

        # Sweep/reclaim.
        if (
            latest_low < range_low
            and latest_close > range_low
        ):
            score += 15
            reasons.append(
                "Support sweep and reclaim"
            )

        # Enough room to opposite edge.
        room = range_high - price

        if room < atr_value * 1.20:
            return None

        if score < RANGE_MIN_SCORE:
            return None

        return {
            "side": "BUY",
            "score": score,
            "level": range_low,
            "reasons": reasons,
        }

    # --------------------------------------------------------
    # RESISTANCE REVERSION
    # --------------------------------------------------------

    if near_resistance:

        score = 0
        reasons = []

        score += 15
        reasons.append(
            "Price holding range resistance"
        )

        bearish_rejection = (
            latest_close < latest_open
            and upper_wick >= candle_range * 0.30
        )

        if bearish_rejection:
            score += 10
            reasons.append(
                "Bearish M5 rejection"
            )

        if latest_close < previous_low:
            score += 15
            reasons.append(
                "M5 reclaimed prior low"
            )

        if rsi_now < rsi_previous:
            score += 10
            reasons.append(
                "M5 RSI falling"
            )

        if macd_now < macd_previous:
            score += 10
            reasons.append(
                "M5 MACD weakening"
            )

        if m15_rsi_now <= m15_rsi_previous:
            score += 5
            reasons.append(
                "M15 momentum stabilising"
            )

        if (
            latest_high > range_high
            and latest_close < range_high
        ):
            score += 15
            reasons.append(
                "Resistance sweep and rejection"
            )

        room = price - range_low

        if room < atr_value * 1.20:
            return None

        if score < RANGE_MIN_SCORE:
            return None

        return {
            "side": "SELL",
            "score": score,
            "level": range_high,
            "reasons": reasons,
        }

    return None


# ============================================================
# MODEL
# ============================================================

class GerkoXAUModel:

    def __init__(
        self,
        symbol: str = "XAU/USD",
    ):
        self.symbol = symbol

    # ========================================================
    # TREND CONTEXT
    # ========================================================

    def trend_context(
        self,
        h4: pd.DataFrame,
        h1: pd.DataFrame,
        m15: pd.DataFrame,
    ) -> str:

        h4_last = h4.iloc[-1]
        h1_last = h1.iloc[-1]
        m15_last = m15.iloc[-1]

        bullish_votes = 0
        bearish_votes = 0

        if (
            float(h4_last["close"])
            > float(h4_last["ema50"])
        ):
            bullish_votes += 1
        else:
            bearish_votes += 1

        if (
            float(h1_last["close"])
            > float(h1_last["ema50"])
        ):
            bullish_votes += 1
        else:
            bearish_votes += 1

        if (
            float(m15_last["close"])
            > float(m15_last["ema50"])
        ):
            bullish_votes += 1
        else:
            bearish_votes += 1

        if bullish_votes >= 2:
            return "BULLISH"

        if bearish_votes >= 2:
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
        context: str,
        impulse: str,
        atr_value: float,
    ) -> Optional[dict]:

        if context not in {
            "BULLISH",
            "BEARISH",
        }:
            return None

        if impulse != context:
            return None

        latest = m5.iloc[-1]

        close = float(
            latest["close"]
        )

        ema20_value = float(
            latest["ema20"]
        )

        rsi_now = float(
            latest["rsi"]
        )

        rsi_previous = float(
            m5.iloc[-2]["rsi"]
        )

        macd_now = float(
            latest["macd_histogram"]
        )

        macd_previous = float(
            m5.iloc[-2]["macd_histogram"]
        )

        distance = abs(
            close - ema20_value
        )

        # Anti-chase.
        if distance > (
            atr_value
            * MAX_TREND_CHASE_ATR
        ):
            return None

        score = 0
        reasons = []

        if context == "BULLISH":

            score += 20
            reasons.append(
                "H4/H1/M15 bullish structure"
            )

            if close > ema20_value:
                score += 10
                reasons.append(
                    "Price above M5 EMA20"
                )

            if rsi_now > rsi_previous:
                score += 10
                reasons.append(
                    "M5 RSI turning higher"
                )

            if macd_now > macd_previous:
                score += 10
                reasons.append(
                    "M5 MACD improving"
                )

            if (
                float(m15.iloc[-1]["close"])
                > float(m15.iloc[-1]["ema20"])
            ):
                score += 10
                reasons.append(
                    "M15 momentum bullish"
                )

            if (
                float(h1.iloc[-1]["close"])
                > float(h1.iloc[-1]["ema20"])
            ):
                score += 10
                reasons.append(
                    "H1 momentum bullish"
                )

            score += 10
            reasons.append(
                "Current M5 impulse confirms BUY"
            )

            return {
                "side": "BUY",
                "score": score,
                "reasons": reasons,
            }

        score += 20
        reasons.append(
            "H4/H1/M15 bearish structure"
        )

        if close < ema20_value:
            score += 10
            reasons.append(
                "Price below M5 EMA20"
            )

        if rsi_now < rsi_previous:
            score += 10
            reasons.append(
                "M5 RSI turning lower"
            )

        if macd_now < macd_previous:
            score += 10
            reasons.append(
                "M5 MACD weakening"
            )

        if (
            float(m15.iloc[-1]["close"])
            < float(m15.iloc[-1]["ema20"])
        ):
            score += 10
            reasons.append(
                "M15 momentum bearish"
            )

        if (
            float(h1.iloc[-1]["close"])
            < float(h1.iloc[-1]["ema20"])
        ):
            score += 10
            reasons.append(
                "H1 momentum bearish"
            )

        score += 10
        reasons.append(
            "Current M5 impulse confirms SELL"
        )

        return {
            "side": "SELL",
            "score": score,
            "reasons": reasons,
        }

    # ========================================================
    # CREATE NORMAL SIGNAL
    # ========================================================

    def create_signal(
        self,
        setup_type: str,
        setup_name: str,
        side: str,
        score: int,
        level,
        reasons: list[str],
        m5: pd.DataFrame,
        context: str,
        impulse: str,
        minimum_score: int,
    ) -> Optional[Signal]:

        decision = classify_signal(
            score,
            minimum_score,
        )

        if not decision.allowed:
            return None

        latest = m5.iloc[-1]

        atr_value = float(
            latest["atr"]
        )

        close = float(
            latest["close"]
        )

        # Entry zone.
        entry_buffer = atr_value * 0.08

        if side == "BUY":
            entry_low = close
            entry_high = close + entry_buffer
            entry_mid = (
                entry_low + entry_high
            ) / 2

            stop_loss = min(
                float(level.price)
                - atr_value * 0.30,
                entry_mid
                - atr_value * 0.75,
            )

            risk = (
                entry_mid
                - stop_loss
            )

            tp1 = entry_mid + risk
            tp2 = entry_mid + risk * 1.5
            tp3 = entry_mid + risk * 2.0

        else:
            entry_high = close
            entry_low = close - entry_buffer
            entry_mid = (
                entry_low + entry_high
            ) / 2

            stop_loss = max(
                float(level.price)
                + atr_value * 0.30,
                entry_mid
                + atr_value * 0.75,
            )

            risk = (
                stop_loss
                - entry_mid
            )

            tp1 = entry_mid - risk
            tp2 = entry_mid - risk * 1.5
            tp3 = entry_mid - risk * 2.0

        rr = calculate_rr(
            entry_mid,
            stop_loss,
            tp2,
            side,
        )

        now = pd.Timestamp.now(
            tz="Europe/London"
        )

        code = {
            "TREND": "TC",
            "SUPPORT_BOUNCE": "SB",
            "RESISTANCE_BOUNCE": "RB",
            "BREAK_RETEST": "BR",
        }.get(
            setup_type,
            "LX",
        )

        signal_id = (
            f"LX-{code}-"
            f"{side[0]}-"
            f"{now.strftime('%Y%m%d-%H%M')}-"
            f"{uuid.uuid4().hex[:4].upper()}"
        )

        return Signal(
            signal_id=signal_id,
            created_at=now.isoformat(),
            symbol=self.symbol,
            setup_type=setup_type,
            setup_name=setup_name,
            side=side,
            score=int(score),
            grade=decision.grade,
            quality=decision.quality,
            trend_context=context,
            impulse=impulse,
            reference_level=float(
                level.price
            ),
            level_strength=int(
                level.strength
            ),
            level_touches=int(
                level.touches
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
            rr=rr,
            lot_size=decision.lot_size,
            reasons=reasons,
            invalidation=(
                "Setup invalid if price closes "
                "through the reference level "
                "with confirmed momentum."
            ),
        )

    # ========================================================
    # CREATE RANGE SIGNAL
    # ========================================================

    def create_range_signal(
        self,
        range_candidate: dict,
        range_info: dict,
        m5: pd.DataFrame,
        context: str,
        impulse: str,
    ) -> Optional[Signal]:

        score = int(
            range_candidate["score"]
        )

        decision = classify_signal(
            score,
            RANGE_MIN_SCORE,
        )

        if not decision.allowed:
            return None

        latest = m5.iloc[-1]

        atr_value = float(
            latest["atr"]
        )

        close = float(
            latest["close"]
        )

        side = range_candidate["side"]

        level = float(
            range_candidate["level"]
        )

        if side == "BUY":

            entry_low = close
            entry_high = (
                close
                + atr_value * 0.08
            )

            entry_mid = (
                entry_low
                + entry_high
            ) / 2

            stop_loss = min(
                range_info["low"]
                - atr_value * 0.30,
                entry_mid
                - atr_value * 0.65,
            )

            risk = (
                entry_mid
                - stop_loss
            )

            room = (
                range_info["high"]
                - entry_mid
            )

            tp1 = entry_mid + min(
                risk,
                room * 0.50,
            )

            tp2 = entry_mid + min(
                risk * 1.50,
                room * 0.85,
            )

            tp3 = entry_mid + min(
                risk * 2.00,
                room * 0.95,
            )

            setup_name = (
                "Range Support Reversion"
            )

        else:

            entry_high = close
            entry_low = (
                close
                - atr_value * 0.08
            )

            entry_mid = (
                entry_low
                + entry_high
            ) / 2

            stop_loss = max(
                range_info["high"]
                + atr_value * 0.30,
                entry_mid
                + atr_value * 0.65,
            )

            risk = (
                stop_loss
                - entry_mid
            )

            room = (
                entry_mid
                - range_info["low"]
            )

            tp1 = entry_mid - min(
                risk,
                room * 0.50,
            )

            tp2 = entry_mid - min(
                risk * 1.50,
                room * 0.85,
            )

            tp3 = entry_mid - min(
                risk * 2.00,
                room * 0.95,
            )

            setup_name = (
                "Range Resistance Reversion"
            )

        rr = calculate_rr(
            entry_mid,
            stop_loss,
            tp2,
            side,
        )

        now = pd.Timestamp.now(
            tz="Europe/London"
        )

        signal_id = (
            f"LX-RG-"
            f"{side[0]}-"
            f"{now.strftime('%Y%m%d-%H%M')}-"
            f"{uuid.uuid4().hex[:4].upper()}"
        )

        return Signal(
            signal_id=signal_id,
            created_at=now.isoformat(),
            symbol=self.symbol,
            setup_type="RANGE_REVERSION",
            setup_name=setup_name,
            side=side,
            score=score,
            grade=decision.grade,
            quality=decision.quality,
            trend_context=context,
            impulse=impulse,
            reference_level=round(
                level,
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
            reasons=range_candidate[
                "reasons"
            ],
            invalidation=(
                "M15 close beyond the range "
                "edge invalidates the reversal. "
                "A clean range breakout cancels "
                "the range-reversion setup."
            ),
        )

    # ========================================================
    # ANALYSE
    # ========================================================

    def analyse(
        self,
        h4: pd.DataFrame,
        h1: pd.DataFrame,
        m15: pd.DataFrame,
        m5: pd.DataFrame,
    ) -> Optional[Signal]:

        # ----------------------------------------------------
        # ENRICH
        # ----------------------------------------------------

        h4 = enrich(h4)
        h1 = enrich(h1)
        m15 = enrich(m15)
        m5 = enrich(m5)

        if min(
            len(h4),
            len(h1),
            len(m15),
            len(m5),
        ) < 100:

            return None

        latest = m5.iloc[-1]

        atr_value = float(
            latest["atr"]
        )

        if atr_value <= 0:
            return None

        context = self.trend_context(
            h4,
            h1,
            m15,
        )

        impulse = current_impulse(
            m5
        )

        # ----------------------------------------------------
        # CURRENT IMPULSE PROTECTION
        # ----------------------------------------------------

        # If the latest M5 move is clearly bullish,
        # don't issue a stale SELL based on old resistance.
        #
        # If clearly bearish, don't issue stale BUY.
        #
        # Range reversals are handled separately because
        # they intentionally anticipate turns at range edges.

        # ----------------------------------------------------
        # EXHAUSTION PROTECTION
        # ----------------------------------------------------

        exhausted = trigger_exhausted(
            m5,
            impulse,
        )

        # ----------------------------------------------------
        # CANDIDATES
        # ----------------------------------------------------

        candidates = []

        # ====================================================
        # TREND
        # ====================================================

        trend_candidate = self.trend_signal(
            h4,
            h1,
            m15,
            m5,
            context,
            impulse,
            atr_value,
        )

        if trend_candidate:

            if not exhausted:

                candidate = {
                    "type": "TREND",
                    "name": "Trend Continuation",
                    "side": trend_candidate["side"],
                    "score": trend_candidate["score"],
                    "reasons": trend_candidate[
                        "reasons"
                    ],
                    "level": None,
                    "minimum": TREND_MIN_SCORE,
                }

                candidates.append(
                    candidate
                )

        # ====================================================
        # SUPPORT / RESISTANCE
        # ====================================================

        h1_levels = build_levels(
            h1,
            atr_value,
            timeframe="H1",
        )

        m15_levels = build_levels(
            m15,
            atr_value,
            timeframe="M15",
        )

        levels = combine_levels(
            h1_levels,
            m15_levels,
            atr_value,
        )

        # ----------------------------------------------------
        # SUPPORT BOUNCE
        # ----------------------------------------------------

        support = detect_support_bounce(
            m5,
            m15,
            h1,
            levels,
            atr_value,
        )

        if support.detected:

            side = "BUY"

            # Do not take a stale counter-direction setup
            # unless it is exceptionally strong.
            minimum = SR_MIN_SCORE

            if context == "BEARISH":
                minimum = COUNTERTREND_SR_MIN_SCORE

            if impulse == "BEARISH":
                minimum = max(
                    minimum,
                    COUNTERTREND_SR_MIN_SCORE,
                )

            if (
                not exhausted
                and support.score >= minimum
                and chase_distance(m5)
                <= atr_value
                * MAX_SR_CHASE_ATR
            ):

                candidates.append(
                    {
                        "type": "SUPPORT_BOUNCE",
                        "name": "Support Bounce",
                        "side": side,
                        "score": support.score,
                        "reasons": support.reasons,
                        "level": support.level,
                        "minimum": minimum,
                    }
                )

        # ----------------------------------------------------
        # RESISTANCE BOUNCE
        # ----------------------------------------------------

        resistance = detect_resistance_bounce(
            m5,
            m15,
            h1,
            levels,
            atr_value,
        )

        if resistance.detected:

            side = "SELL"

            minimum = SR_MIN_SCORE

            if context == "BULLISH":
                minimum = COUNTERTREND_SR_MIN_SCORE

            if impulse == "BULLISH":
                minimum = max(
                    minimum,
                    COUNTERTREND_SR_MIN_SCORE,
                )

            if (
                not exhausted
                and resistance.score >= minimum
                and chase_distance(m5)
                <= atr_value
                * MAX_SR_CHASE_ATR
            ):

                candidates.append(
                    {
                        "type": "RESISTANCE_BOUNCE",
                        "name": "Resistance Bounce",
                        "side": side,
                        "score": resistance.score,
                        "reasons": resistance.reasons,
                        "level": resistance.level,
                        "minimum": minimum,
                    }
                )

        # ====================================================
        # RANGE ENGINE
        # ====================================================

        range_info = range_regime(
            m15,
            h1,
            atr_value,
        )

        if range_info:

            # A clean breakout overrides range reversion.
            range_is_breaking = range_breaking(
                m15,
                range_info,
                atr_value,
            )

            if not range_is_breaking:

                range_candidate = range_signal(
                    m15,
                    m5,
                    range_info,
                    atr_value,
                )

                if range_candidate:

                    range_signal_obj = (
                        self.create_range_signal(
                            range_candidate,
                            range_info,
                            m5,
                            context,
                            impulse,
                        )
                    )

                    if range_signal_obj:

                        return range_signal_obj

        # ====================================================
        # BREAK & RETEST
        # ====================================================

        break_retest = detect_break_retest(
            m5,
            m15,
            h1,
            levels,
            atr_value,
        )

        if break_retest.detected:

            if (
                not exhausted
                and break_retest.score
                >= BREAK_RETEST_MIN_SCORE
            ):

                # Current impulse must agree with breakout.
                if (
                    break_retest.side
                    == "BUY"
                    and impulse != "BEARISH"
                ) or (
                    break_retest.side
                    == "SELL"
                    and impulse != "BULLISH"
                ):

                    candidates.append(
                        {
                            "type": "BREAK_RETEST",
                            "name": "Break & Retest",
                            "side": break_retest.side,
                            "score": break_retest.score,
                            "reasons": break_retest.reasons,
                            "level": break_retest.level,
                            "minimum": BREAK_RETEST_MIN_SCORE,
                        }
                    )

        # ====================================================
        # NO CANDIDATES
        # ====================================================

        if not candidates:
            return None

        # ====================================================
        # SORT
        # ====================================================

        # Higher score first.
        #
        # If scores are tied:
        #   1. Non-trend setup preferred.
        #   2. Range reversal preferred.
        #   3. Break/retest next.
        #
        # This helps prevent the model from choosing a
        # late trend continuation when a cleaner structural
        # opportunity exists.

        priority = {
            "RANGE_REVERSION": 5,
            "SUPPORT_BOUNCE": 4,
            "RESISTANCE_BOUNCE": 4,
            "BREAK_RETEST": 3,
            "TREND": 2,
        }

        candidates.sort(
            key=lambda item: (
                item["score"],
                priority.get(
                    item["type"],
                    0,
                ),
            ),
            reverse=True,
        )

        best = candidates[0]

        # ====================================================
        # FINAL SIGNAL
        # ====================================================

        return self.create_signal(
            setup_type=best["type"],
            setup_name=best["name"],
            side=best["side"],
            score=best["score"],
            level=best["level"],
            reasons=best["reasons"],
            m5=m5,
            context=context,
            impulse=impulse,
            minimum_score=best["minimum"],
        )

    # ========================================================
    # FORMAT TELEGRAM SIGNAL
    # ========================================================

    def format_signal(
        self,
        signal: Signal,
    ) -> str:

        if signal.setup_type == "TREND":
            header = (
                "📈🧠 "
                "<b>LumiAj-XAU TREND SIGNAL</b>"
            )
            setup_tag = "#LX_TREND"

        elif signal.setup_type == "SUPPORT_BOUNCE":
            header = (
                "🟢🧠 "
                "<b>LumiAj-XAU SUPPORT BOUNCE</b>"
            )
            setup_tag = "#LX_SUPPORT_BOUNCE"

        elif signal.setup_type == "RESISTANCE_BOUNCE":
            header = (
                "🔴🧠 "
                "<b>LumiAj-XAU RESISTANCE BOUNCE</b>"
            )
            setup_tag = "#LX_RESISTANCE_BOUNCE"

        elif signal.setup_type == "BREAK_RETEST":
            header = (
                "🔄🧠 "
                "<b>LumiAj-XAU BREAK & RETEST</b>"
            )
            setup_tag = "#LX_BREAK_RETEST"

        else:
            header = (
                "↔️🧠 "
                "<b>LumiAj-XAU RANGE REVERSAL</b>"
            )
            setup_tag = "#LX_RANGE"

        reasons_text = "\n".join(
            f"• {reason}"
            for reason in signal.reasons
        )

        return (
            f"{header}\n\n"

            f"💎 <b>{signal.symbol}</b>\n"
            f"🎯 <b>{signal.setup_name}</b>\n"
            f"📊 Direction: "
            f"<b>{signal.side}</b>\n\n"

            f"⭐ Score: "
            f"<b>{signal.score}/100</b>\n"
            f"🏆 Grade: "
            f"<b>{signal.grade}</b>\n"
            f"🔥 Quality: "
            f"<b>{signal.quality}</b>\n\n"

            f"🧭 Trend Context: "
            f"<b>{signal.trend_context}</b>\n"
            f"⚡ Current M5 Impulse: "
            f"<b>{signal.impulse}</b>\n\n"

            f"📍 Reference Level: "
            f"<b>{signal.reference_level:.2f}</b>\n\n"

            f"💰 Entry Zone: "
            f"<b>{signal.entry_low:.2f}"
            f" – "
            f"{signal.entry_high:.2f}</b>\n"

            f"🛑 Stop Loss: "
            f"<b>{signal.stop_loss:.2f}</b>\n"

            f"🎯 TP1: "
            f"<b>{signal.tp1:.2f}</b>\n"

            f"🎯 TP2: "
            f"<b>{signal.tp2:.2f}</b>\n"

            f"🎯 TP3: "
            f"<b>{signal.tp3:.2f}</b>\n\n"

            f"📐 RR to TP2: "
            f"<b>1:{signal.rr:.2f}</b>\n"

            f"📦 Model Lot: "
            f"<b>{signal.lot_size:.2f}</b>\n\n"

            f"🧠 <b>WHY LumiAj TRIGGERED</b>\n"
            f"{reasons_text}\n\n"

            f"⚠️ <b>Invalidation:</b>\n"
            f"{signal.invalidation}\n\n"

            f"⚠️ Anti-chase protection active.\n\n"

            f"<code>{signal.signal_id}</code>\n\n"

            f"#LUMIAJ_XAU\n"
            f"{setup_tag}\n"
            f"#LX_{signal.side}"
        )