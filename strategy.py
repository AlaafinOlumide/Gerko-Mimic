from __future__ import annotations

import html
import inspect
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

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

logger = logging.getLogger(__name__)


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
# LOGGING
# ============================================================

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        result = float(value)
        if not np.isfinite(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _fmt(value: Any, digits: int = 2) -> str:
    try:
        value = float(value)
        if not np.isfinite(value):
            return "n/a"
        return f"{value:.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _get(obj: Any, *names: str, default: Any = None) -> Any:
    """
    Safely retrieve an attribute or dictionary key from an object.
    This makes the strategy tolerant of different sr_engine result
    dataclass implementations.
    """
    if obj is None:
        return default

    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]

        if hasattr(obj, name):
            try:
                return getattr(obj, name)
            except Exception:
                pass

    return default


def _result_dict(obj: Any) -> dict:
    if obj is None:
        return {}

    if isinstance(obj, dict):
        return obj

    if hasattr(obj, "__dict__"):
        try:
            return dict(obj.__dict__)
        except Exception:
            return {}

    return {}


# ============================================================
# SIGNAL DATA MODEL
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


def true_range(df: pd.DataFrame) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    previous_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - previous_close).abs()
    tr3 = (low - previous_close).abs()

    return pd.concat(
        [tr1, tr2, tr3],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).rolling(
        period,
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


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
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

    atr_value = tr.rolling(
        period,
        min_periods=period,
    ).mean()

    plus_di = (
        100
        * plus_dm.rolling(
            period,
            min_periods=period,
        ).mean()
        / atr_value.replace(0, np.nan)
    )

    minus_di = (
        100
        * minus_dm.rolling(
            period,
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

    return dx.rolling(
        period,
        min_periods=period,
    ).mean()


# ============================================================
# DATA ENRICHMENT
# ============================================================

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

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
    ).copy()

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    df["ema_20"] = ema(
        df["close"],
        20,
    )

    df["ema_50"] = ema(
        df["close"],
        50,
    )

    df["ema_100"] = ema(
        df["close"],
        100,
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    df["rsi"] = rsi(
        df["close"],
        14,
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    df["atr"] = atr(
        df,
        14,
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    fast_ema = ema(
        df["close"],
        12,
    )

    slow_ema = ema(
        df["close"],
        26,
    )

    df["macd_line"] = (
        fast_ema - slow_ema
    )

    df["macd_signal"] = ema(
        df["macd_line"],
        9,
    )

    df["macd_hist"] = (
        df["macd_line"]
        - df["macd_signal"]
    )

    # Compatibility with sr_engine
    df["macd_histogram"] = df["macd_hist"]

    # --------------------------------------------------------
    # ADX
    # --------------------------------------------------------

    df["adx"] = adx(
        df,
        14,
    )

    return df


# ============================================================
# PRICE / IMPULSE HELPERS
# ============================================================

def current_impulse(
    m5: pd.DataFrame,
) -> str:
    """
    Determines the immediate M5 direction.

    This is deliberately more important than stale higher-timeframe
    direction when deciding whether a signal is currently executable.
    """

    if m5 is None or len(m5) < 5:
        return "NEUTRAL"

    last = m5.iloc[-1]
    previous = m5.iloc[-2]
    recent = m5.iloc[-4:]

    close = _safe_float(last["close"])
    previous_close = _safe_float(previous["close"])

    ema20 = _safe_float(last.get("ema_20"))
    ema50 = _safe_float(last.get("ema_50"))

    recent_close = _safe_float(
        recent["close"].iloc[0]
    )

    short_move = close - previous_close
    four_candle_move = close - recent_close

    bullish_votes = 0
    bearish_votes = 0

    if close > ema20:
        bullish_votes += 1
    elif close < ema20:
        bearish_votes += 1

    if ema20 > ema50:
        bullish_votes += 1
    elif ema20 < ema50:
        bearish_votes += 1

    if short_move > 0:
        bullish_votes += 1
    elif short_move < 0:
        bearish_votes += 1

    if four_candle_move > 0:
        bullish_votes += 1
    elif four_candle_move < 0:
        bearish_votes += 1

    if bullish_votes >= 3:
        return "BUY"

    if bearish_votes >= 3:
        return "SELL"

    return "NEUTRAL"


def trigger_exhausted(
    m5: pd.DataFrame,
    side: str,
) -> tuple[bool, str]:
    """
    Detect oversized trigger candles.

    This is an anti-chase mechanism.
    """

    if m5 is None or len(m5) < 3:
        return False, "insufficient_data"

    last = m5.iloc[-1]

    high = _safe_float(last["high"])
    low = _safe_float(last["low"])
    close = _safe_float(last["close"])
    open_price = _safe_float(last["open"])

    candle_range = high - low

    atr_value = _safe_float(
        last.get("atr"),
        0.0,
    )

    if atr_value <= 0:
        return False, "atr_unavailable"

    body = abs(close - open_price)

    range_ratio = candle_range / atr_value
    body_ratio = body / atr_value

    if range_ratio >= 1.60:
        if side == "BUY" and close > open_price:
            return True, (
                f"bullish trigger oversized "
                f"range={range_ratio:.2f}ATR"
            )

        if side == "SELL" and close < open_price:
            return True, (
                f"bearish trigger oversized "
                f"range={range_ratio:.2f}ATR"
            )

    if body_ratio >= 1.25:
        if side == "BUY" and close > open_price:
            return True, (
                f"bullish trigger body oversized "
                f"body={body_ratio:.2f}ATR"
            )

        if side == "SELL" and close < open_price:
            return True, (
                f"bearish trigger body oversized "
                f"body={body_ratio:.2f}ATR"
            )

    return False, "not_exhausted"


def chase_distance(
    price: float,
    reference: float,
    atr_value: float,
) -> float:
    if atr_value <= 0:
        return 999.0

    return abs(price - reference) / atr_value


# ============================================================
# TREND CONTEXT
# ============================================================

def trend_context(
    h1: pd.DataFrame,
    m15: pd.DataFrame,
) -> str:

    if h1 is None or m15 is None:
        return "NEUTRAL"

    h1_last = h1.iloc[-1]
    m15_last = m15.iloc[-1]

    h1_close = _safe_float(h1_last["close"])
    h1_ema20 = _safe_float(h1_last.get("ema_20"))
    h1_ema50 = _safe_float(h1_last.get("ema_50"))

    m15_close = _safe_float(m15_last["close"])
    m15_ema20 = _safe_float(m15_last.get("ema_20"))
    m15_ema50 = _safe_float(m15_last.get("ema_50"))

    bullish = 0
    bearish = 0

    if h1_close > h1_ema50:
        bullish += 1
    elif h1_close < h1_ema50:
        bearish += 1

    if h1_ema20 > h1_ema50:
        bullish += 1
    elif h1_ema20 < h1_ema50:
        bearish += 1

    if m15_close > m15_ema50:
        bullish += 1
    elif m15_close < m15_ema50:
        bearish += 1

    if m15_ema20 > m15_ema50:
        bullish += 1
    elif m15_ema20 < m15_ema50:
        bearish += 1

    if bullish >= 3:
        return "BULLISH"

    if bearish >= 3:
        return "BEARISH"

    return "NEUTRAL"


# ============================================================
# GENERIC SR ENGINE CALLER
# ============================================================

def _call_sr_function(
    function,
    *,
    m5: Optional[pd.DataFrame] = None,
    m15: Optional[pd.DataFrame] = None,
    h1: Optional[pd.DataFrame] = None,
    levels: Any = None,
    atr_value: Optional[float] = None,
    price: Optional[float] = None,
    label: str = "SR",
):
    """
    Compatibility wrapper around sr_engine.

    The current engine signatures are:

        detect_support_bounce(
            m5,
            m15,
            levels,
            atr_value
        )

        detect_resistance_bounce(
            m5,
            m15,
            levels,
            atr_value
        )

        detect_break_retest(
            m5,
            levels,
            atr_value
        )
    """

    try:
        signature = inspect.signature(function)

        kwargs = {}

        for name, parameter in signature.parameters.items():

            lname = name.lower()

            if lname in {
                "m5",
                "data",
                "df",
                "candles",
            }:
                kwargs[name] = m5

            elif lname == "m15":
                kwargs[name] = m15

            elif lname == "h1":
                kwargs[name] = h1

            elif lname in {
                "levels",
                "sr_levels",
            }:
                kwargs[name] = levels

            elif lname in {
                "atr",
                "atr_value",
                "volatility",
            }:
                kwargs[name] = atr_value

            elif lname == "price":
                kwargs[name] = price

        logger.info(
            "%s CALL | args=%s | signature=%s",
            label,
            len(kwargs),
            signature,
        )

        result = function(**kwargs)

        return result

    except Exception as exc:
        logger.exception(
            "%s ERROR | %s",
            label,
            exc,
        )

        return None


# ============================================================
# LEVEL BUILDING
# ============================================================

def _call_level_builder(
    dataframe: pd.DataFrame,
    *,
    atr_value: float,
    timeframe: str,
):
    """
    Compatibility wrapper for different build_levels versions.
    """

    try:
        signature = inspect.signature(
            build_levels
        )

        kwargs = {}

        for name in signature.parameters:
            lname = name.lower()

            if lname in {
                "df",
                "data",
                "candles",
                "m5",
                "m15",
                "h1",
            }:
                kwargs[name] = dataframe

            elif lname in {
                "atr",
                "atr_value",
                "volatility",
            }:
                kwargs[name] = atr_value

            elif lname in {
                "timeframe",
                "tf",
            }:
                kwargs[name] = timeframe

        return build_levels(
            **kwargs
        )

    except Exception as exc:
        logger.exception(
            "LEVEL BUILD ERROR | timeframe=%s | %s",
            timeframe,
            exc,
        )
        return []


# ============================================================
# RESULT DIAGNOSTICS
# ============================================================

def _diagnose_sr_result(
    label: str,
    result: Any,
) -> dict:

    if result is None:
        logger.info(
            "%s RESULT | result=None",
            label,
        )

        return {
            "detected": False,
            "side": "NEUTRAL",
            "score": 0,
            "level": 0.0,
            "touches": 0,
            "strength": 0.0,
            "reasons": [],
        }

    detected = bool(
        _get(
            result,
            "detected",
            "valid",
            "qualified",
            default=False,
        )
    )

    side = str(
        _get(
            result,
            "side",
            "direction",
            default="NEUTRAL",
        )
    ).upper()

    score = _safe_int(
        _get(
            result,
            "score",
            "confidence",
            "quality_score",
            default=0,
        )
    )

    # sr_engine.BounceResult.level / BreakRetestResult.level is an
    # SRLevel object, not a numeric price. Unwrap it before converting.
    level_obj = _get(
        result,
        "level",
        "reference_level",
        "price_level",
        "reference",
        default=None,
    )

    if isinstance(level_obj, (int, float, str, np.number)):
        level = _safe_float(level_obj)
    else:
        level = _safe_float(
            _get(
                level_obj,
                "price",
                "level_price",
                "value",
                default=0.0,
            )
        )

    # Prefer explicitly provided result metadata when present; otherwise
    # take it from the nested SRLevel dataclass/dictionary.
    touches_value = _get(
        result,
        "touches",
        "level_touches",
        "touch_count",
        default=None,
    )
    if touches_value is None:
        touches_value = _get(
            level_obj,
            "touches",
            "level_touches",
            "touch_count",
            default=0,
        )
    touches = _safe_int(touches_value)

    strength_value = _get(
        result,
        "strength",
        "level_strength",
        default=None,
    )
    if strength_value is None:
        strength_value = _get(
            level_obj,
            "strength",
            "level_strength",
            default=0.0,
        )
    strength = _safe_float(strength_value)

    reasons = _get(
        result,
        "reasons",
        "reason",
        default=[],
    )

    if isinstance(reasons, str):
        reasons = [reasons]

    if reasons is None:
        reasons = []

    if detected and level <= 0:
        logger.warning(
            "%s RESULT WARNING | detected=True but no valid reference level; "
            "raw_level_type=%s",
            label,
            type(level_obj).__name__,
        )

    logger.info(
        "%s RESULT | detected=%s | side=%s | score=%s | "
        "level=%s | touches=%s | strength=%s | reasons=%s",
        label,
        detected,
        side,
        score,
        _fmt(level),
        touches,
        _fmt(strength),
        "; ".join(str(x) for x in reasons),
    )

    return {
        "detected": detected,
        "side": side,
        "score": score,
        "level": level,
        "touches": touches,
        "strength": strength,
        "reasons": reasons,
        "raw": result,
    }


# ============================================================
# RANGE ENGINE
# ============================================================

def range_regime(
    h1: pd.DataFrame,
    m15: pd.DataFrame,
) -> tuple[bool, dict]:

    if h1 is None or m15 is None:
        return False, {
            "reason": "missing_data"
        }

    if len(h1) < 60 or len(m15) < 60:
        return False, {
            "reason": "insufficient_history"
        }

    h1_last = h1.iloc[-1]
    m15_last = m15.iloc[-1]

    h1_atr = _safe_float(
        h1_last.get("atr")
    )

    m15_atr = _safe_float(
        m15_last.get("atr")
    )

    if h1_atr <= 0:
        return False, {
            "reason": "H1 ATR unavailable"
        }

    if m15_atr <= 0:
        return False, {
            "reason": "M15 ATR unavailable"
        }

    recent = m15.tail(48)

    range_high = _safe_float(
        recent["high"].max()
    )

    range_low = _safe_float(
        recent["low"].min()
    )

    width = range_high - range_low

    width_atr = width / m15_atr

    if width_atr < RANGE_MIN_WIDTH_ATR:
        return False, {
            "reason": (
                f"range too narrow "
                f"{width_atr:.2f}ATR"
            )
        }

    if width_atr > RANGE_MAX_WIDTH_ATR:
        return False, {
            "reason": (
                f"range too wide "
                f"{width_atr:.2f}ATR"
            )
        }

    h1_adx = _safe_float(
        h1_last.get("adx"),
        999.0,
    )

    m15_adx = _safe_float(
        m15_last.get("adx"),
        999.0,
    )

    if h1_adx > RANGE_ADX_MAX:
        return False, {
            "reason": (
                f"H1 ADX {h1_adx:.1f} "
                f"> {RANGE_ADX_MAX:.1f}"
            )
        }

    if m15_adx > 23:
        return False, {
            "reason": (
                f"M15 ADX {m15_adx:.1f} > 23"
            )
        }

    h1_ema50_slope = abs(
        _safe_float(h1.iloc[-1]["ema_50"])
        - _safe_float(h1.iloc[-6]["ema_50"])
    )

    m15_ema50_slope = abs(
        _safe_float(m15.iloc[-1]["ema_50"])
        - _safe_float(m15.iloc[-6]["ema_50"])
    )

    if h1_ema50_slope > h1_atr * 0.35:
        return False, {
            "reason": (
                f"H1 EMA50 slope too strong "
                f"{h1_ema50_slope / h1_atr:.2f}ATR"
            )
        }

    if m15_ema50_slope > m15_atr * 0.45:
        return False, {
            "reason": (
                f"M15 EMA50 slope too strong "
                f"{m15_ema50_slope / m15_atr:.2f}ATR"
            )
        }

    price = _safe_float(
        m15_last["close"]
    )

    buffer = m15_atr * RANGE_BREAK_BUFFER_ATR

    if price < range_low - buffer:
        return False, {
            "reason": "price below range"
        }

    if price > range_high + buffer:
        return False, {
            "reason": "price above range"
        }

    return True, {
        "reason": "qualified range regime",
        "range_high": range_high,
        "range_low": range_low,
        "width": width,
        "width_atr": width_atr,
        "h1_adx": h1_adx,
        "m15_adx": m15_adx,
        "atr": m15_atr,
    }


# ============================================================
# MODEL
# ============================================================

def range_edge_score(edge_distance_atr: float, room_atr: float) -> float:
    """Score a range-edge bounce (0-100). Components are capped so the score cannot
    be inflated by one condition: edge proximity 50, room to the opposite edge 30,
    confirmed range regime 20."""
    edge_pts = 50.0 * max(0.0, 1.0 - edge_distance_atr / RANGE_EDGE_ATR)
    room_pts = 30.0 * min(max(room_atr, 0.0) / 3.0, 1.0)
    return round(20.0 + edge_pts + room_pts, 1)


class GerkoXAUModel:

    def __init__(
        self,
        symbol: str = "XAU/USD",
    ):
        self.symbol = symbol

    # ========================================================
    # TREND SIGNAL
    # ========================================================

    def trend_signal(
        self,
        h1: pd.DataFrame,
        m15: pd.DataFrame,
        m5: pd.DataFrame,
        context: str,
        impulse: str,
    ) -> Optional[dict]:

        last_h1 = h1.iloc[-1]
        last_m15 = m15.iloc[-1]
        last_m5 = m5.iloc[-1]

        price = _safe_float(
            last_m5["close"]
        )

        atr_value = _safe_float(
            last_m5.get("atr")
        )

        rsi_value = _safe_float(
            last_m5.get("rsi"),
            50,
        )

        macd_hist = _safe_float(
            last_m5.get("macd_histogram")
        )

        macd_previous = _safe_float(
            m5.iloc[-2].get("macd_histogram")
        )

        score = 0
        reasons = []

        # ----------------------------------------------------
        # Context
        # ----------------------------------------------------

        if context == "NEUTRAL":
            logger.info(
                "TREND CHECK | REJECT=context neutral"
            )
            return None

        side = impulse

        if side not in {"BUY", "SELL"}:
            logger.info(
                "TREND CHECK | REJECT=impulse neutral"
            )
            return None

        # ----------------------------------------------------
        # H1 price position - 15
        # ----------------------------------------------------

        h1_close = _safe_float(
            last_h1["close"]
        )

        h1_ema50 = _safe_float(
            last_h1.get("ema_50")
        )

        if (
            side == "BUY"
            and h1_close > h1_ema50
        ):
            score += 15
            reasons.append(
                "H1 price above EMA50"
            )

        elif (
            side == "SELL"
            and h1_close < h1_ema50
        ):
            score += 15
            reasons.append(
                "H1 price below EMA50"
            )

        # ----------------------------------------------------
        # H1 EMA structure - 12
        # ----------------------------------------------------

        h1_ema20 = _safe_float(
            last_h1.get("ema_20")
        )

        if (
            side == "BUY"
            and h1_ema20 > h1_ema50
        ):
            score += 12
            reasons.append(
                "H1 EMA20 > EMA50"
            )

        elif (
            side == "SELL"
            and h1_ema20 < h1_ema50
        ):
            score += 12
            reasons.append(
                "H1 EMA20 < EMA50"
            )

        # ----------------------------------------------------
        # M15 price - 12
        # ----------------------------------------------------

        m15_close = _safe_float(
            last_m15["close"]
        )

        m15_ema50 = _safe_float(
            last_m15.get("ema_50")
        )

        if (
            side == "BUY"
            and m15_close > m15_ema50
        ):
            score += 12
            reasons.append(
                "M15 price above EMA50"
            )

        elif (
            side == "SELL"
            and m15_close < m15_ema50
        ):
            score += 12
            reasons.append(
                "M15 price below EMA50"
            )

        # ----------------------------------------------------
        # M15 short-term structure - 8
        # ----------------------------------------------------

        m15_ema20 = _safe_float(
            last_m15.get("ema_20")
        )

        if (
            side == "BUY"
            and m15_ema20 > m15_ema50
        ):
            score += 8
            reasons.append(
                "M15 EMA20 > EMA50"
            )

        elif (
            side == "SELL"
            and m15_ema20 < m15_ema50
        ):
            score += 8
            reasons.append(
                "M15 EMA20 < EMA50"
            )

        # ----------------------------------------------------
        # M5 RSI - 12
        # ----------------------------------------------------

        if side == "BUY":

            if 52 <= rsi_value <= 68:
                score += 12
                reasons.append(
                    f"M5 RSI constructive {rsi_value:.1f}"
                )

            elif 48 <= rsi_value < 52:
                score += 6
                reasons.append(
                    f"M5 RSI early turn {rsi_value:.1f}"
                )

        else:

            if 32 <= rsi_value <= 48:
                score += 12
                reasons.append(
                    f"M5 RSI constructive {rsi_value:.1f}"
                )

            elif 48 < rsi_value <= 52:
                score += 6
                reasons.append(
                    f"M5 RSI early turn {rsi_value:.1f}"
                )

        # ----------------------------------------------------
        # MACD - 10
        # ----------------------------------------------------

        if side == "BUY":

            if macd_hist > 0 and macd_hist >= macd_previous:
                score += 10
                reasons.append(
                    "MACD bullish and strengthening"
                )

            elif macd_hist >= macd_previous:
                score += 5
                reasons.append(
                    "MACD bullish turn"
                )

        else:

            if macd_hist < 0 and macd_hist <= macd_previous:
                score += 10
                reasons.append(
                    "MACD bearish and strengthening"
                )

            elif macd_hist <= macd_previous:
                score += 5
                reasons.append(
                    "MACD bearish turn"
                )

        # ----------------------------------------------------
        # Immediate impulse - 10
        # ----------------------------------------------------

        if impulse == side:
            score += 10
            reasons.append(
                f"M5 impulse agrees {impulse}"
            )

        # ----------------------------------------------------
        # MARKET STATE DIAGNOSTIC
        # ----------------------------------------------------

        logger.info(
            "MARKET STATE | context=%s | impulse=%s | "
            "price=%s | RSI=%s | MACD=%s | "
            "MACD_prev=%s | ATR=%s",
            context,
            impulse,
            _fmt(price),
            _fmt(rsi_value, 1),
            _fmt(macd_hist, 3),
            _fmt(macd_previous, 3),
            _fmt(atr_value),
        )

        logger.info(
            "TREND SCORE | side=%s | score=%s/79 | "
            "required=%s",
            side,
            score,
            TREND_MIN_SCORE,
        )

        # ----------------------------------------------------
        # Chase protection
        # ----------------------------------------------------

        ema20_distance = (
            abs(price - _safe_float(last_m5["ema_20"]))
            / atr_value
            if atr_value > 0
            else 999
        )

        logger.info(
            "TREND CHASE | side=%s | distance=%.2fATR | "
            "max=%.2fATR",
            side,
            ema20_distance,
            MAX_TREND_CHASE_ATR,
        )

        if ema20_distance > MAX_TREND_CHASE_ATR:
            logger.info(
                "TREND CHECK | REJECT=CHASE | "
                "distance=%.2fATR | max=%.2fATR",
                ema20_distance,
                MAX_TREND_CHASE_ATR,
            )
            return None

        # ----------------------------------------------------
        # Trigger exhaustion
        # ----------------------------------------------------

        exhausted, exhaustion_reason = trigger_exhausted(
            m5,
            side,
        )

        logger.info(
            "TRIGGER CHECK | side=%s | exhausted=%s | "
            "reason=%s",
            side,
            exhausted,
            exhaustion_reason,
        )

        if exhausted:
            logger.info(
                "TREND CHECK | REJECT=TRIGGER_EXHAUSTED | %s",
                exhaustion_reason,
            )
            return None

        # ----------------------------------------------------
        # Score
        # ----------------------------------------------------

        if score < TREND_MIN_SCORE:

            logger.info(
                "TREND CHECK | REJECT=SCORE | "
                "score=%s | required=%s | gap=%s",
                score,
                TREND_MIN_SCORE,
                TREND_MIN_SCORE - score,
            )

            if score >= TREND_MIN_SCORE - 5:
                logger.info(
                    "NEAR MISS | TREND | side=%s | "
                    "score=%s | required=%s",
                    side,
                    score,
                    TREND_MIN_SCORE,
                )

            return None

        logger.info(
            "TREND CHECK | QUALIFIED | side=%s | "
            "score=%s",
            side,
            score,
        )

        return {
            "setup_type": "TC",
            "setup_name": "TREND CONTINUATION",
            "side": side,
            "score": score,
            "reference_level": _safe_float(
                last_m5["ema_20"]
            ),
            "level_strength": 0.0,
            "level_touches": 0,
            "reasons": reasons,
        }

    # ========================================================
    # SUPPORT / RESISTANCE
    # ========================================================

    def sr_signal(
        self,
        result: Any,
        *,
        label: str,
        context: str,
        impulse: str,
        m5: pd.DataFrame,
        min_score: int,
    ) -> Optional[dict]:

        info = _diagnose_sr_result(
            label,
            result,
        )

        if not info["detected"]:
            logger.info(
                "%s CHECK | REJECT=NOT_DETECTED",
                label,
            )
            return None

        side = info["side"]

        if side not in {"BUY", "SELL"}:
            logger.info(
                "%s CHECK | REJECT=INVALID_SIDE | side=%s",
                label,
                side,
            )
            return None

        score = info["score"]

        # ----------------------------------------------------
        # Countertrend classification
        # ----------------------------------------------------

        countertrend = False

        if context == "BULLISH" and side == "SELL":
            countertrend = True

        elif context == "BEARISH" and side == "BUY":
            countertrend = True

        if impulse in {"BUY", "SELL"} and side != impulse:
            countertrend = True

        required_score = (
            COUNTERTREND_SR_MIN_SCORE
            if countertrend
            else min_score
        )

        logger.info(
            "%s CHECK | side=%s | countertrend=%s | "
            "score=%s | required=%s",
            label,
            side,
            countertrend,
            score,
            required_score,
        )

        if score < required_score:

            logger.info(
                "%s CHECK | REJECT=SCORE | "
                "score=%s | required=%s | gap=%s",
                label,
                score,
                required_score,
                required_score - score,
            )

            if score >= required_score - 5:
                logger.info(
                    "NEAR MISS | %s | score=%s | "
                    "required=%s",
                    label,
                    score,
                    required_score,
                )

            return None

        # ----------------------------------------------------
        # Price / level
        # ----------------------------------------------------

        last = m5.iloc[-1]

        price = _safe_float(
            last["close"]
        )

        atr_value = _safe_float(
            last.get("atr")
        )

        level = info["level"]

        distance = chase_distance(
            price,
            level,
            atr_value,
        )

        logger.info(
            "%s CHASE | price=%s | level=%s | "
            "distance=%.2fATR | max=%.2fATR",
            label,
            _fmt(price),
            _fmt(level),
            distance,
            MAX_SR_CHASE_ATR,
        )

        if distance > MAX_SR_CHASE_ATR:
            logger.info(
                "%s CHECK | REJECT=CHASE | "
                "distance=%.2fATR | max=%.2fATR",
                label,
                distance,
                MAX_SR_CHASE_ATR,
            )
            return None

        # ----------------------------------------------------
        # Exhaustion
        # ----------------------------------------------------

        exhausted, exhaustion_reason = trigger_exhausted(
            m5,
            side,
        )

        logger.info(
            "%s TRIGGER | exhausted=%s | reason=%s",
            label,
            exhausted,
            exhaustion_reason,
        )

        if exhausted:
            logger.info(
                "%s CHECK | REJECT=TRIGGER_EXHAUSTED",
                label,
            )
            return None

        logger.info(
            "%s CHECK | QUALIFIED | side=%s | "
            "score=%s | level=%s",
            label,
            side,
            score,
            _fmt(level),
        )

        return {
            "setup_type": (
                "SB"
                if label.upper().startswith("SUPPORT")
                else "RB"
            ),
            "setup_name": label,
            "side": side,
            "score": score,
            "reference_level": level,
            "level_strength": info["strength"],
            "level_touches": info["touches"],
            "reasons": info["reasons"],
            "raw": result,
        }

    # ========================================================
    # BREAK / RETEST
    # ========================================================

    def break_retest_signal(
        self,
        result: Any,
        *,
        context: str,
        impulse: str,
        m5: pd.DataFrame,
    ) -> Optional[dict]:

        info = _diagnose_sr_result(
            "BREAK RETEST",
            result,
        )

        if not info["detected"]:
            logger.info(
                "BREAK RETEST CHECK | REJECT=NOT_DETECTED"
            )
            return None

        side = info["side"]

        if side not in {"BUY", "SELL"}:
            logger.info(
                "BREAK RETEST CHECK | REJECT=INVALID_SIDE"
            )
            return None

        score = info["score"]

        # ----------------------------------------------------
        # Current impulse veto
        # ----------------------------------------------------

        if impulse in {"BUY", "SELL"}:
            if side != impulse:
                logger.info(
                    "BREAK RETEST CHECK | "
                    "REJECT=IMPULSE_CONFLICT | "
                    "side=%s | impulse=%s",
                    side,
                    impulse,
                )
                return None

        # ----------------------------------------------------
        # Higher-timeframe context veto
        # ----------------------------------------------------

        if context == "BULLISH" and side == "SELL":
            logger.info(
                "BREAK RETEST CHECK | "
                "REJECT=CONTEXT_CONFLICT | "
                "context=%s | side=%s",
                context,
                side,
            )
            return None

        if context == "BEARISH" and side == "BUY":
            logger.info(
                "BREAK RETEST CHECK | "
                "REJECT=CONTEXT_CONFLICT | "
                "context=%s | side=%s",
                context,
                side,
            )
            return None

        logger.info(
            "BREAK RETEST CHECK | side=%s | "
            "score=%s | required=%s",
            side,
            score,
            BREAK_RETEST_MIN_SCORE,
        )

        if score < BREAK_RETEST_MIN_SCORE:
            logger.info(
                "BREAK RETEST CHECK | REJECT=SCORE | "
                "score=%s | required=%s | gap=%s",
                score,
                BREAK_RETEST_MIN_SCORE,
                BREAK_RETEST_MIN_SCORE - score,
            )

            if score >= BREAK_RETEST_MIN_SCORE - 5:
                logger.info(
                    "NEAR MISS | BREAK RETEST | "
                    "score=%s | required=%s",
                    score,
                    BREAK_RETEST_MIN_SCORE,
                )

            return None

        # ----------------------------------------------------
        # Chase
        # ----------------------------------------------------

        last = m5.iloc[-1]

        price = _safe_float(
            last["close"]
        )

        atr_value = _safe_float(
            last.get("atr")
        )

        level = info["level"]

        distance = chase_distance(
            price,
            level,
            atr_value,
        )

        logger.info(
            "BREAK RETEST CHASE | price=%s | "
            "level=%s | distance=%.2fATR | max=%.2fATR",
            _fmt(price),
            _fmt(level),
            distance,
            MAX_TREND_CHASE_ATR,
        )

        if distance > MAX_TREND_CHASE_ATR:
            logger.info(
                "BREAK RETEST CHECK | REJECT=CHASE | "
                "distance=%.2fATR | max=%.2fATR",
                distance,
                MAX_TREND_CHASE_ATR,
            )
            return None

        # ----------------------------------------------------
        # Exhaustion
        # ----------------------------------------------------

        exhausted, exhaustion_reason = trigger_exhausted(
            m5,
            side,
        )

        logger.info(
            "BREAK RETEST TRIGGER | exhausted=%s | "
            "reason=%s",
            exhausted,
            exhaustion_reason,
        )

        if exhausted:
            logger.info(
                "BREAK RETEST CHECK | "
                "REJECT=TRIGGER_EXHAUSTED"
            )
            return None

        logger.info(
            "BREAK RETEST CHECK | QUALIFIED | "
            "side=%s | score=%s",
            side,
            score,
        )

        return {
            "setup_type": "BR",
            "setup_name": "BREAK + RETEST",
            "side": side,
            "score": score,
            "reference_level": level,
            "level_strength": info["strength"],
            "level_touches": info["touches"],
            "reasons": info["reasons"],
            "raw": result,
        }

    # ========================================================
    # RANGE SIGNAL
    # ========================================================

    def range_signal(
        self,
        m15: pd.DataFrame,
        regime_info: dict,
    ) -> Optional[dict]:

        if not regime_info:
            logger.info(
                "RANGE CHECK | REJECT=no_regime"
            )
            return None

        range_high = _safe_float(
            regime_info.get("range_high")
        )

        range_low = _safe_float(
            regime_info.get("range_low")
        )

        atr_value = _safe_float(
            regime_info.get("atr")
        )

        if range_high <= range_low:
            logger.info(
                "RANGE CHECK | REJECT=invalid_range"
            )
            return None

        price = _safe_float(
            m15.iloc[-1]["close"]
        )

        width = range_high - range_low

        if atr_value <= 0:
            logger.info(
                "RANGE CHECK | REJECT=ATR unavailable"
            )
            return None

        distance_low = (
            price - range_low
        )

        distance_high = (
            range_high - price
        )

        low_distance_atr = (
            distance_low / atr_value
        )

        high_distance_atr = (
            distance_high / atr_value
        )

        # ----------------------------------------------------
        # BUY from lower edge
        # ----------------------------------------------------

        if low_distance_atr <= RANGE_EDGE_ATR:

            opposite_room = (
                range_high - price
            ) / atr_value

            if opposite_room < 1.2:
                logger.info(
                    "RANGE CHECK | BUY REJECT="
                    "INSUFFICIENT_ROOM | room=%.2fATR",
                    opposite_room,
                )
                return None

            score = range_edge_score(
                low_distance_atr,
                opposite_room,
            )

            logger.info(
                "RANGE CHECK | BUY CANDIDATE | "
                "score=%s | lower_edge=%.2fATR | "
                "room=%.2fATR",
                score,
                low_distance_atr,
                opposite_room,
            )

            if score < RANGE_MIN_SCORE:
                logger.info(
                    "RANGE CHECK | BUY REJECT=SCORE"
                )
                return None

            return {
                "setup_type": "RG",
                "setup_name": "RANGE LOW BOUNCE",
                "side": "BUY",
                "score": score,
                "reference_level": range_low,
                "level_strength": 0.0,
                "level_touches": 0,
                "reasons": [
                    "Range regime active",
                    "Price near lower range edge",
                    "Room toward opposite range edge",
                ],
            }

        # ----------------------------------------------------
        # SELL from upper edge
        # ----------------------------------------------------

        if high_distance_atr <= RANGE_EDGE_ATR:

            # Room from price down to the lower edge (not the full range width).
            opposite_room = (
                price - range_low
            ) / atr_value

            if opposite_room < 1.2:
                logger.info(
                    "RANGE CHECK | SELL REJECT="
                    "INSUFFICIENT_ROOM | room=%.2fATR",
                    opposite_room,
                )
                return None

            score = range_edge_score(
                high_distance_atr,
                opposite_room,
            )

            logger.info(
                "RANGE CHECK | SELL CANDIDATE | "
                "score=%s | upper_edge=%.2fATR",
                score,
                high_distance_atr,
            )

            if score < RANGE_MIN_SCORE:
                logger.info(
                    "RANGE CHECK | SELL REJECT=SCORE"
                )
                return None

            return {
                "setup_type": "RG",
                "setup_name": "RANGE HIGH BOUNCE",
                "side": "SELL",
                "score": score,
                "reference_level": range_high,
                "level_strength": 0.0,
                "level_touches": 0,
                "reasons": [
                    "Range regime active",
                    "Price near upper range edge",
                    "Room toward opposite range edge",
                ],
            }

        logger.info(
            "RANGE CHECK | REJECT=NOT_AT_EDGE | "
            "low=%.2fATR | high=%.2fATR | "
            "required=%.2fATR",
            low_distance_atr,
            high_distance_atr,
            RANGE_EDGE_ATR,
        )

        return None

    # ========================================================
    # SIGNAL CREATION
    # ========================================================

    def create_signal(
        self,
        candidate: dict,
        m5: pd.DataFrame,
        *,
        context: str,
        impulse: str,
        signal_time: Optional[datetime] = None,
    ) -> Optional[Signal]:

        if not candidate:
            return None

        side = candidate["side"]

        score = _safe_int(
            candidate["score"]
        )

        setup_type = str(
            candidate.get(
                "setup_type",
                "LX",
            )
        )

        setup_name = str(
            candidate.get(
                "setup_name",
                "LUMIAJ SETUP",
            )
        )

        reference_level = _safe_float(
            candidate.get(
                "reference_level",
                0,
            )
        )

        last = m5.iloc[-1]

        entry_mid = _safe_float(
            last["close"]
        )

        atr_value = _safe_float(
            last.get("atr")
        )

        if atr_value <= 0:
            logger.info(
                "SIGNAL CREATION REJECT | ATR unavailable"
            )
            return None

        # ----------------------------------------------------
        # Entry zone
        # ----------------------------------------------------

        entry_buffer = max(
            atr_value * 0.08,
            0.30,
        )

        entry_low = entry_mid - entry_buffer
        entry_high = entry_mid + entry_buffer

        # ----------------------------------------------------
        # Stop
        # ----------------------------------------------------

        if side == "BUY":

            base_stop = min(
                reference_level
                - atr_value * 0.35,
                entry_mid
                - atr_value * 0.80,
            )

            stop_loss = base_stop

        else:

            base_stop = max(
                reference_level
                + atr_value * 0.35,
                entry_mid
                + atr_value * 0.80,
            )

            stop_loss = base_stop

        risk = abs(
            entry_mid - stop_loss
        )

        if risk <= 0:
            logger.info(
                "SIGNAL CREATION REJECT | invalid risk"
            )
            return None

        # ----------------------------------------------------
        # Targets
        # ----------------------------------------------------

        if side == "BUY":

            tp1 = entry_mid + risk * 1.0
            tp2 = entry_mid + risk * 2.0
            tp3 = entry_mid + risk * 3.0

        else:

            tp1 = entry_mid - risk * 1.0
            tp2 = entry_mid - risk * 2.0
            tp3 = entry_mid - risk * 3.0

        rr = abs(
            tp2 - entry_mid
        ) / risk

        minimum = {
            "TC": TREND_MIN_SCORE,
            "SB": SR_MIN_SCORE,
            "RB": SR_MIN_SCORE,
            "BR": BREAK_RETEST_MIN_SCORE,
            "RG": RANGE_MIN_SCORE,
        }.get(
            setup_type,
            TREND_MIN_SCORE,
        )

        decision = classify_signal(
            score,
            minimum,
        )

        if not decision.allowed:
            logger.info(
                "SIGNAL CREATION REJECT | "
                "risk_engine denied | score=%s | "
                "minimum=%s",
                score,
                minimum,
            )
            return None

        # ----------------------------------------------------
        # Signal ID
        # ----------------------------------------------------

        now = signal_time or datetime.now()

        code = setup_type

        signal_id = (
            f"LX-{code}-"
            f"{side[0]}-"
            f"{now:%Y%m%d-%H%M}-"
            f"{uuid.uuid4().hex[:4].upper()}"
        )

        reasons = candidate.get(
            "reasons",
            [],
        )

        if isinstance(reasons, str):
            reasons = [reasons]

        reasons = [
            str(reason)
            for reason in reasons
        ]

        invalidation = (
            f"{side} invalid if price "
            f"decisively breaks the setup reference "
            f"against the position."
        )

        return Signal(
            signal_id=signal_id,
            created_at=now.isoformat(),
            symbol=self.symbol,
            setup_type=setup_type,
            setup_name=setup_name,
            side=side,
            score=score,
            grade=decision.grade,
            quality=decision.quality,
            trend_context=context,
            impulse=impulse,
            reference_level=reference_level,
            level_strength=_safe_float(
                candidate.get(
                    "level_strength",
                    0,
                )
            ),
            level_touches=_safe_int(
                candidate.get(
                    "level_touches",
                    0,
                )
            ),
            entry_low=entry_low,
            entry_high=entry_high,
            entry_mid=entry_mid,
            stop_loss=stop_loss,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            rr=rr,
            lot_size=decision.lot_size,
            reasons=reasons,
            invalidation=invalidation,
        )

    # ========================================================
    # MAIN ANALYSIS
    # ========================================================

    def analyse(
        self,
        h1: Optional[pd.DataFrame] = None,
        m15: Optional[pd.DataFrame] = None,
        m5: Optional[pd.DataFrame] = None,
        candles=None,
        signal_time=None,
        **kwargs,
    ) -> Optional[Signal]:

        # ----------------------------------------------------
        # Compatibility with previous main.py interfaces
        # ----------------------------------------------------

        if m5 is None and candles is not None:

            if isinstance(candles, dict):

                if h1 is None:
                    h1 = candles.get("h1")
                if m15 is None:
                    m15 = candles.get("m15")
                if m5 is None:
                    m5 = candles.get("m5")

        if h1 is None:
            h1 = kwargs.get("H1")

        if m15 is None:
            m15 = kwargs.get("M15")

        if m5 is None:
            m5 = kwargs.get("M5")

        if h1 is None or m15 is None or m5 is None:
            logger.info(
                "ANALYSIS REJECT | missing timeframe data"
            )
            return None

        started = datetime.now()

        try:

            # ------------------------------------------------
            # Enrich
            # ------------------------------------------------

            h1 = enrich(h1)
            m15 = enrich(m15)
            m5 = enrich(m5)

            if (
                len(h1) < 100
                or len(m15) < 100
                or len(m5) < 100
            ):
                logger.info(
                    "ANALYSIS REJECT | insufficient candles | "
                    "H1=%s | M15=%s | M5=%s",
                    len(h1),
                    len(m15),
                    len(m5),
                )
                return None

            # ------------------------------------------------
            # Current market state
            # ------------------------------------------------

            context = trend_context(
                h1,
                m15,
            )

            impulse = current_impulse(
                m5,
            )

            atr_value = _safe_float(
                m5.iloc[-1].get("atr")
            )

            price = _safe_float(
                m5.iloc[-1]["close"]
            )

            rsi_value = _safe_float(
                m5.iloc[-1].get("rsi"),
                50,
            )

            macd_hist = _safe_float(
                m5.iloc[-1].get(
                    "macd_histogram"
                )
            )

            logger.info(
                "MARKET STATE | context=%s | "
                "impulse=%s | price=%s | "
                "RSI=%s | MACD=%s | ATR=%s",
                context,
                impulse,
                _fmt(price),
                _fmt(rsi_value, 1),
                _fmt(macd_hist, 3),
                _fmt(atr_value),
            )

            # ------------------------------------------------
            # Build S/R levels
            # ------------------------------------------------

            h1_levels = _call_level_builder(
                h1,
                atr_value=_safe_float(
                    h1.iloc[-1].get("atr")
                ),
                timeframe="H1",
            )

            m15_levels = _call_level_builder(
                m15,
                atr_value=_safe_float(
                    m15.iloc[-1].get("atr")
                ),
                timeframe="M15",
            )

            # Combine H1 and M15 levels using the current sr_engine
            # signature: combine_levels(h1_levels, m15_levels, atr_value).
            # Do not retry with the same invalid arguments if this fails.
            try:
                levels = combine_levels(
                    h1_levels,
                    m15_levels,
                    atr_value,
                )
                levels = levels if levels is not None else []
                logger.info(
                    "LEVELS COMBINED | H1=%s | M15=%s | combined=%s | ATR=%s",
                    len(h1_levels) if h1_levels is not None else 0,
                    len(m15_levels) if m15_levels is not None else 0,
                    len(levels),
                    _fmt(atr_value),
                )
            except Exception as exc:
                logger.exception(
                    "LEVEL COMBINATION ERROR | %s",
                    exc,
                )
                levels = []

            # ------------------------------------------------
            # Candidates
            # ------------------------------------------------

            candidates = []

            # =================================================
            # TREND CONTINUATION
            # =================================================

            trend_candidate = self.trend_signal(
                h1,
                m15,
                m5,
                context,
                impulse,
            )

            if trend_candidate:
                candidates.append(
                    trend_candidate
                )

            # =================================================
            # SUPPORT BOUNCE
            # =================================================

            support_result = _call_sr_function(
                detect_support_bounce,
                m5=m5,
                m15=m15,
                h1=h1,
                levels=levels,
                atr_value=atr_value,
                price=price,
                label="SUPPORT BOUNCE",
            )

            support_candidate = self.sr_signal(
                support_result,
                label="SUPPORT BOUNCE",
                context=context,
                impulse=impulse,
                m5=m5,
                min_score=SR_MIN_SCORE,
            )

            if support_candidate:
                candidates.append(
                    support_candidate
                )

            # =================================================
            # RESISTANCE BOUNCE
            # =================================================

            resistance_result = _call_sr_function(
                detect_resistance_bounce,
                m5=m5,
                m15=m15,
                h1=h1,
                levels=levels,
                atr_value=atr_value,
                price=price,
                label="RESISTANCE BOUNCE",
            )

            resistance_candidate = self.sr_signal(
                resistance_result,
                label="RESISTANCE BOUNCE",
                context=context,
                impulse=impulse,
                m5=m5,
                min_score=SR_MIN_SCORE,
            )

            if resistance_candidate:
                candidates.append(
                    resistance_candidate
                )

            # =================================================
            # RANGE
            # =================================================

            regime, regime_info = range_regime(
                h1,
                m15,
            )

            if regime:

                logger.info(
                    "RANGE CHECK | regime=True | "
                    "width=%.2fATR | H1_ADX=%.1f | "
                    "M15_ADX=%.1f",
                    regime_info.get(
                        "width_atr",
                        0,
                    ),
                    regime_info.get(
                        "h1_adx",
                        0,
                    ),
                    regime_info.get(
                        "m15_adx",
                        0,
                    ),
                )

                range_candidate = self.range_signal(
                    m15,
                    regime_info,
                )

                if range_candidate:
                    candidates.append(
                        range_candidate
                    )

            else:

                logger.info(
                    "RANGE CHECK | regime=False | "
                    "reason=%s",
                    regime_info.get(
                        "reason",
                        "unknown",
                    ),
                )

            # =================================================
            # BREAK / RETEST
            # =================================================

            break_result = _call_sr_function(
                detect_break_retest,
                m5=m5,
                m15=m15,
                h1=h1,
                levels=levels,
                atr_value=atr_value,
                price=price,
                label="BREAK/RETEST",
            )

            break_candidate = self.break_retest_signal(
                break_result,
                context=context,
                impulse=impulse,
                m5=m5,
            )

            if break_candidate:
                candidates.append(
                    break_candidate
                )

            # =================================================
            # CANDIDATE SUMMARY
            # =================================================

            logger.info(
                "CANDIDATE SUMMARY | count=%s",
                len(candidates),
            )

            for candidate in candidates:
                logger.info(
                    "CANDIDATE | type=%s | name=%s | "
                    "side=%s | score=%s | level=%s",
                    candidate.get(
                        "setup_type"
                    ),
                    candidate.get(
                        "setup_name"
                    ),
                    candidate.get(
                        "side"
                    ),
                    candidate.get(
                        "score"
                    ),
                    _fmt(
                        candidate.get(
                            "reference_level"
                        )
                    ),
                )

            if not candidates:
                logger.info(
                    "ANALYSIS RESULT | candidates=0 | "
                    "NO QUALIFIED LUMIAJ SETUP"
                )

                elapsed = (
                    datetime.now()
                    - started
                ).total_seconds()

                logger.info(
                    "ANALYSIS COMPLETE | %.2fs",
                    elapsed,
                )

                return None

            # =================================================
            # PRIORITY
            # =================================================

            priority = {
                "RG": 5,
                "SB": 4,
                "RB": 4,
                "BR": 3,
                "TC": 2,
            }

            candidates.sort(
                key=lambda item: (
                    _safe_int(
                        item.get(
                            "score"
                        )
                    ),
                    priority.get(
                        item.get(
                            "setup_type"
                        ),
                        0,
                    ),
                ),
                reverse=True,
            )

            selected = candidates[0]

            logger.info(
                "SELECTED CANDIDATE | type=%s | "
                "side=%s | score=%s | "
                "context=%s | impulse=%s",
                selected.get(
                    "setup_type"
                ),
                selected.get(
                    "side"
                ),
                selected.get(
                    "score"
                ),
                context,
                impulse,
            )

            # =================================================
            # FINAL DIRECTION SAFETY
            # =================================================

            selected_type = selected.get(
                "setup_type"
            )

            selected_side = selected.get(
                "side"
            )

            if selected_type in {
                "TC",
                "BR",
            }:

                if (
                    impulse in {
                        "BUY",
                        "SELL",
                    }
                    and selected_side != impulse
                ):
                    logger.info(
                        "FINAL SAFETY REJECT | "
                        "setup=%s | side=%s | "
                        "impulse=%s",
                        selected_type,
                        selected_side,
                        impulse,
                    )
                    return None

            # ------------------------------------------------
            # Create final signal
            # ------------------------------------------------

            signal = self.create_signal(
                selected,
                m5,
                context=context,
                impulse=impulse,
                signal_time=signal_time,
            )

            if signal is None:
                logger.info(
                    "ANALYSIS RESULT | "
                    "candidate_selected_but_signal_creation_failed"
                )
                return None

            elapsed = (
                datetime.now()
                - started
            ).total_seconds()

            logger.info(
                "ANALYSIS COMPLETE | %.2fs | "
                "SIGNAL=%s | %s | %s | score=%s",
                elapsed,
                signal.signal_id,
                signal.setup_type,
                signal.side,
                signal.score,
            )

            return signal

        except Exception as exc:

            logger.exception(
                "ANALYSIS ERROR | %s",
                exc,
            )

            return None

    # ========================================================
    # TELEGRAM FORMATTER
    # ========================================================

    def format_signal(
        self,
        signal: Signal,
    ) -> str:

        side_emoji = (
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

        side_hashtag = (
            "#LX_BUY"
            if signal.side == "BUY"
            else "#LX_SELL"
        )

        reason_text = "\n".join(
            f"• {html.escape(str(reason))}"
            for reason in signal.reasons[:6]
        )

        return (
            "🧠⚡ <b>LumiAj-XAU</b>\n"
            "\n"
            f"<b>{side_emoji} {signal.side}</b> "
            f"<b>{signal.symbol}</b>\n"
            "\n"
            f"<b>Premium LumiAj Setup</b>\n"
            f"Setup: <b>{html.escape(str(signal.setup_name))}</b>\n"
            f"Score: <b>{signal.score}/100</b>\n"
            f"Grade: <b>{signal.grade}</b>\n"
            f"Quality: <b>{signal.quality}</b>\n"
            "\n"
            f"<b>Entry Zone</b>\n"
            f"{signal.entry_low:.2f} – "
            f"{signal.entry_high:.2f}\n"
            f"Mid: <b>{signal.entry_mid:.2f}</b>\n"
            "\n"
            f"<b>Stop Loss</b>\n"
            f"{signal.stop_loss:.2f}\n"
            "\n"
            f"<b>Take Profit</b>\n"
            f"TP1: {signal.tp1:.2f}\n"
            f"TP2: {signal.tp2:.2f}\n"
            f"TP3: {signal.tp3:.2f}\n"
            "\n"
            f"<b>RR</b>: 1:{signal.rr:.2f}\n"
            f"<b>Lot</b>: {signal.lot_size:.2f}\n"
            "\n"
            f"<b>Trend Context</b>: "
            f"{html.escape(str(signal.trend_context))}\n"
            f"<b>M5 Impulse</b>: "
            f"{html.escape(str(signal.impulse))}\n"
            "\n"
            "<b>WHY LumiAj TRIGGERED</b>\n"
            f"{reason_text}\n"
            "\n"
            f"<b>Reference Level</b>: "
            f"{signal.reference_level:.2f}\n"
            f"<b>Level Strength</b>: "
            f"{signal.level_strength:.2f}\n"
            f"<b>Touches</b>: "
            f"{signal.level_touches}\n"
            "\n"
            f"<b>Invalidation</b>\n"
            f"{html.escape(str(signal.invalidation))}\n"
            "\n"
            f"🆔 <code>{signal.signal_id}</code>\n"
            "\n"
            f"{setup_hashtag} "
            f"{side_hashtag} "
            f"#LUMIAJ_XAU"
        )


# ============================================================
# MODULE-LEVEL COMPATIBILITY FUNCTIONS
# ============================================================

def calculate_rr(
    entry: float,
    stop_loss: float,
    take_profit: float,
) -> float:

    risk = abs(
        entry - stop_loss
    )

    if risk <= 0:
        return 0.0

    reward = abs(
        take_profit - entry
    )

    return reward / risk


def format_signal(
    signal: Signal,
) -> str:

    model = GerkoXAUModel(
        symbol=signal.symbol
    )

    return model.format_signal(
        signal
    )