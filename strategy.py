from __future__ import annotations

import math
import secrets
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from risk_engine import (
    RiskEngine,
    calculate_rr,
)


@dataclass(frozen=True)
class ScoreBreakdown:
    trend: int
    liquidity: int
    momentum: int
    volatility: int
    execution: int

    @property
    def total(self) -> int:
        return min(
            self.trend
            + self.liquidity
            + self.momentum
            + self.volatility
            + self.execution,
            100,
        )


@dataclass(frozen=True)
class Signal:
    signal_id: str
    created_at: datetime
    symbol: str
    side: str
    score: int
    grade: str
    quality: str

    trend_score: int
    liquidity_score: int
    momentum_score: int
    volatility_score: int
    execution_score: int

    entry_low: float
    entry_high: float
    entry_mid: float

    stop_loss: float
    tp1: float
    tp2: float
    tp3: float

    lot_size: float
    rr: float

    regime: str
    setup_name: str
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

    gain = delta.clip(
        lower=0,
    )

    loss = -delta.clip(
        upper=0,
    )

    average_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()

    average_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()

    relative_strength = (
        average_gain
        / average_loss.replace(
            0,
            math.nan,
        )
    )

    return 100 - (
        100
        / (
            1
            + relative_strength
        )
    )


def true_range(
    dataframe: pd.DataFrame,
) -> pd.Series:
    high_low = (
        dataframe["high"]
        - dataframe["low"]
    )

    high_previous_close = (
        dataframe["high"]
        - dataframe["close"].shift(1)
    ).abs()

    low_previous_close = (
        dataframe["low"]
        - dataframe["close"].shift(1)
    ).abs()

    return pd.concat(
        [
            high_low,
            high_previous_close,
            low_previous_close,
        ],
        axis=1,
    ).max(axis=1)


def atr(
    dataframe: pd.DataFrame,
    period: int = 14,
) -> pd.Series:
    range_series = true_range(
        dataframe
    )

    return range_series.ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()


def adx(
    dataframe: pd.DataFrame,
    period: int = 14,
) -> pd.Series:
    up_move = dataframe["high"].diff()
    down_move = -dataframe["low"].diff()

    plus_dm = up_move.where(
        (
            up_move > down_move
        )
        & (
            up_move > 0
        ),
        0.0,
    )

    minus_dm = down_move.where(
        (
            down_move > up_move
        )
        & (
            down_move > 0
        ),
        0.0,
    )

    average_true_range = atr(
        dataframe,
        period,
    ).replace(
        0,
        math.nan,
    )

    plus_di = (
        100
        * plus_dm.ewm(
            alpha=1 / period,
            adjust=False,
        ).mean()
        / average_true_range
    )

    minus_di = (
        100
        * minus_dm.ewm(
            alpha=1 / period,
            adjust=False,
        ).mean()
        / average_true_range
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


def macd_histogram(
    series: pd.Series,
) -> pd.Series:
    fast_ema = ema(
        series,
        12,
    )

    slow_ema = ema(
        series,
        26,
    )

    macd_line = fast_ema - slow_ema

    signal_line = macd_line.ewm(
        span=9,
        adjust=False,
    ).mean()

    return macd_line - signal_line


def bollinger_width(
    series: pd.Series,
    period: int = 20,
) -> pd.Series:
    middle = series.rolling(
        period
    ).mean()

    standard_deviation = series.rolling(
        period
    ).std()

    upper_band = (
        middle
        + (
            2
            * standard_deviation
        )
    )

    lower_band = (
        middle
        - (
            2
            * standard_deviation
        )
    )

    return (
        upper_band
        - lower_band
    ) / middle.replace(
        0,
        math.nan,
    )


def enrich(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = dataframe.copy()

    for column in [
        "open",
        "high",
        "low",
        "close",
    ]:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    dataframe = dataframe.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close",
        ]
    )

    dataframe["ema20"] = ema(
        dataframe["close"],
        20,
    )

    dataframe["ema50"] = ema(
        dataframe["close"],
        50,
    )

    dataframe["ema200"] = ema(
        dataframe["close"],
        200,
    )

    dataframe["rsi"] = rsi(
        dataframe["close"],
        14,
    )

    dataframe["atr"] = atr(
        dataframe,
        14,
    )

    dataframe["adx"] = adx(
        dataframe,
        14,
    )

    dataframe["macd_histogram"] = macd_histogram(
        dataframe["close"]
    )

    dataframe["bb_width"] = bollinger_width(
        dataframe["close"],
        20,
    )

    dataframe["ema20_slope"] = (
        dataframe["ema20"]
        .diff(3)
    )

    dataframe["ema50_slope"] = (
        dataframe["ema50"]
        .diff(5)
    )

    dataframe["candle_range"] = (
        dataframe["high"]
        - dataframe["low"]
    )

    dataframe["body_size"] = (
        dataframe["close"]
        - dataframe["open"]
    ).abs()

    dataframe["upper_wick"] = (
        dataframe["high"]
        - dataframe[
            [
                "open",
                "close",
            ]
        ].max(axis=1)
    )

    dataframe["lower_wick"] = (
        dataframe[
            [
                "open",
                "close",
            ]
        ].min(axis=1)
        - dataframe["low"]
    )

    return dataframe.dropna().reset_index(
        drop=True
    )


# ============================================================
# GERKO MODEL
# ============================================================

class GerkoXAUModel:
    """
    Retail Gerko-inspired probability model for XAUUSD.

    This does not reproduce XTX Markets or Alexander Gerko's
    proprietary systems. It uses a transparent, rules-based
    scoring engine designed for retail market data.
    """

    def __init__(
        self,
        min_score: int = 80,
    ) -> None:
        self.risk_engine = RiskEngine(
            min_score=min_score
        )

    @staticmethod
    def generate_signal_id(
        signal_time: datetime,
        side: str,
    ) -> str:
        timestamp = signal_time.strftime(
            "%Y%m%d-%H%M"
        )

        random_suffix = secrets.token_hex(
            2
        ).upper()

        direction_code = (
            "B"
            if side.upper() == "BUY"
            else "S"
        )

        return (
            f"GX-{direction_code}-"
            f"{timestamp}-{random_suffix}"
        )

    @staticmethod
    def determine_regime(
        h1_last: pd.Series,
    ) -> str:
        adx_value = float(
            h1_last["adx"]
        )

        bullish_alignment = (
            h1_last["close"]
            > h1_last["ema50"]
            > h1_last["ema200"]
        )

        bearish_alignment = (
            h1_last["close"]
            < h1_last["ema50"]
            < h1_last["ema200"]
        )

        if bullish_alignment and adx_value >= 25:
            return "Strong Bullish Trend"

        if bearish_alignment and adx_value >= 25:
            return "Strong Bearish Trend"

        if bullish_alignment:
            return "Developing Bullish Trend"

        if bearish_alignment:
            return "Developing Bearish Trend"

        if adx_value < 18:
            return "Range / Low Trend Strength"

        return "Mixed Market Structure"

    def analyse(
        self,
        candles: Dict[str, pd.DataFrame],
        signal_time: Optional[datetime] = None,
    ) -> Optional[Signal]:
        required_timeframes = {
            "h1",
            "m15",
            "m5",
        }

        if not required_timeframes.issubset(
            candles
        ):
            raise ValueError(
                "Candles must include h1, m15 and m5 dataframes."
            )

        signal_time = (
            signal_time
            or datetime.now()
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
            len(h1),
            len(m15),
            len(m5),
        ) < 80:
            return None

        h1_last = h1.iloc[-1]
        m15_last = m15.iloc[-1]
        m5_last = m5.iloc[-1]

        price = float(
            m5_last["close"]
        )

        current_atr = float(
            m15_last["atr"]
        )

        if (
            not math.isfinite(current_atr)
            or current_atr <= 0
        ):
            return None

        bullish_trend = (
            h1_last["close"]
            > h1_last["ema50"]
            > h1_last["ema200"]
            and h1_last["ema50_slope"] > 0
        )

        bearish_trend = (
            h1_last["close"]
            < h1_last["ema50"]
            < h1_last["ema200"]
            and h1_last["ema50_slope"] < 0
        )

        if bullish_trend:
            side = "BUY"
        elif bearish_trend:
            side = "SELL"
        else:
            return None

        regime = self.determine_regime(
            h1_last
        )

        reasons: List[str] = []

        trend_score = self._score_trend(
            side=side,
            h1_last=h1_last,
            m15_last=m15_last,
            reasons=reasons,
        )

        liquidity_score = self._score_liquidity(
            side=side,
            m15=m15,
            reasons=reasons,
        )

        momentum_score = self._score_momentum(
            side=side,
            m15=m15,
            m5=m5,
            reasons=reasons,
        )

        volatility_score = self._score_volatility(
            m15=m15,
            reasons=reasons,
        )

        levels = self._calculate_trade_levels(
            side=side,
            price=price,
            current_atr=current_atr,
            m15=m15,
        )

        rr = calculate_rr(
            entry=levels["entry_mid"],
            stop_loss=levels["stop_loss"],
            take_profit=levels["tp2"],
            side=side,
        )

        execution_score = self._score_execution(
            side=side,
            price=price,
            current_atr=current_atr,
            m5_last=m5_last,
            rr=rr,
            reasons=reasons,
        )

        breakdown = ScoreBreakdown(
            trend=trend_score,
            liquidity=liquidity_score,
            momentum=momentum_score,
            volatility=volatility_score,
            execution=execution_score,
        )

        score = breakdown.total

        decision = self.risk_engine.decide(
            score
        )

        if not decision.trade_allowed:
            return None

        setup_name = self._setup_name(
            side=side,
            liquidity_score=liquidity_score,
            momentum_score=momentum_score,
        )

        if side == "BUY":
            invalidation = (
                f"An M15 candle close below "
                f"{levels['stop_loss']:.2f} "
                "invalidates this Gerko buy setup."
            )
        else:
            invalidation = (
                f"An M15 candle close above "
                f"{levels['stop_loss']:.2f} "
                "invalidates this Gerko sell setup."
            )

        return Signal(
            signal_id=self.generate_signal_id(
                signal_time=signal_time,
                side=side,
            ),
            created_at=signal_time,
            symbol="XAUUSD",
            side=side,
            score=score,
            grade=decision.grade,
            quality=decision.quality,
            trend_score=trend_score,
            liquidity_score=liquidity_score,
            momentum_score=momentum_score,
            volatility_score=volatility_score,
            execution_score=execution_score,
            entry_low=round(
                levels["entry_low"],
                2,
            ),
            entry_high=round(
                levels["entry_high"],
                2,
            ),
            entry_mid=round(
                levels["entry_mid"],
                2,
            ),
            stop_loss=round(
                levels["stop_loss"],
                2,
            ),
            tp1=round(
                levels["tp1"],
                2,
            ),
            tp2=round(
                levels["tp2"],
                2,
            ),
            tp3=round(
                levels["tp3"],
                2,
            ),
            lot_size=decision.lot_size,
            rr=rr,
            regime=regime,
            setup_name=setup_name,
            reasons=reasons[:9],
            invalidation=invalidation,
        )

    @staticmethod
    def _score_trend(
        side: str,
        h1_last: pd.Series,
        m15_last: pd.Series,
        reasons: List[str],
    ) -> int:
        score = 0

        if side == "BUY":
            if (
                h1_last["close"]
                > h1_last["ema50"]
                > h1_last["ema200"]
            ):
                score += 18
                reasons.append(
                    "H1 price, EMA50 and EMA200 are bullishly aligned."
                )

            if h1_last["ema50_slope"] > 0:
                score += 4
                reasons.append(
                    "H1 EMA50 slope is rising."
                )

            if m15_last["ema20"] > m15_last["ema50"]:
                score += 3
                reasons.append(
                    "M15 EMA20 remains above EMA50."
                )

        else:
            if (
                h1_last["close"]
                < h1_last["ema50"]
                < h1_last["ema200"]
            ):
                score += 18
                reasons.append(
                    "H1 price, EMA50 and EMA200 are bearishly aligned."
                )

            if h1_last["ema50_slope"] < 0:
                score += 4
                reasons.append(
                    "H1 EMA50 slope is falling."
                )

            if m15_last["ema20"] < m15_last["ema50"]:
                score += 3
                reasons.append(
                    "M15 EMA20 remains below EMA50."
                )

        adx_value = float(
            h1_last["adx"]
        )

        if adx_value >= 30:
            score += 5
            reasons.append(
                f"H1 ADX shows strong trend strength at {adx_value:.1f}."
            )
        elif adx_value >= 22:
            score += 3
            reasons.append(
                f"H1 ADX confirms an active trend at {adx_value:.1f}."
            )

        return min(
            score,
            30,
        )

    @staticmethod
    def _score_liquidity(
        side: str,
        m15: pd.DataFrame,
        reasons: List[str],
    ) -> int:
        score = 0

        recent = m15.tail(
            14
        )

        latest = recent.iloc[-1]
        previous = recent.iloc[:-1]

        previous_high = float(
            previous["high"].max()
        )

        previous_low = float(
            previous["low"].min()
        )

        candle_range = max(
            float(
                latest["candle_range"]
            ),
            0.01,
        )

        body_size = float(
            latest["body_size"]
        )

        body_ratio = (
            body_size
            / candle_range
        )

        if side == "BUY":
            swept_low = (
                latest["low"]
                <= previous_low
            )

            bullish_close = (
                latest["close"]
                > latest["open"]
            )

            recovered_low = (
                latest["close"]
                > previous_low
            )

            if (
                swept_low
                and bullish_close
                and recovered_low
            ):
                score += 14
                reasons.append(
                    "M15 swept sell-side liquidity and recovered above the prior low."
                )
            elif bullish_close:
                score += 4
                reasons.append(
                    "Latest M15 candle closed bullish."
                )

            lower_wick_ratio = (
                float(
                    latest["lower_wick"]
                )
                / candle_range
            )

            if lower_wick_ratio >= 0.35:
                score += 6
                reasons.append(
                    "M15 shows a strong lower rejection wick."
                )

        else:
            swept_high = (
                latest["high"]
                >= previous_high
            )

            bearish_close = (
                latest["close"]
                < latest["open"]
            )

            recovered_high = (
                latest["close"]
                < previous_high
            )

            if (
                swept_high
                and bearish_close
                and recovered_high
            ):
                score += 14
                reasons.append(
                    "M15 swept buy-side liquidity and rejected below the prior high."
                )
            elif bearish_close:
                score += 4
                reasons.append(
                    "Latest M15 candle closed bearish."
                )

            upper_wick_ratio = (
                float(
                    latest["upper_wick"]
                )
                / candle_range
            )

            if upper_wick_ratio >= 0.35:
                score += 6
                reasons.append(
                    "M15 shows a strong upper rejection wick."
                )

        if body_ratio >= 0.55:
            score += 2
            reasons.append(
                "The confirmation candle has a decisive body."
            )

        return min(
            score,
            20,
        )

    @staticmethod
    def _score_momentum(
        side: str,
        m15: pd.DataFrame,
        m5: pd.DataFrame,
        reasons: List[str],
    ) -> int:
        score = 0

        m15_last = m15.iloc[-1]
        m5_last = m5.iloc[-1]

        m15_previous = m15.iloc[-2]
        m5_previous = m5.iloc[-2]

        if side == "BUY":
            if (
                m15_last["rsi"] >= 52
                and m5_last["rsi"] >= 52
            ):
                score += 7
                reasons.append(
                    "M15 and M5 RSI are aligned above 52."
                )
            elif (
                m15_last["rsi"] > 50
                and m5_last["rsi"] > 50
            ):
                score += 4

            if (
                m15_last["macd_histogram"] > 0
                and m5_last["macd_histogram"] > 0
            ):
                score += 6
                reasons.append(
                    "M15 and M5 MACD histogram are positive."
                )

            if (
                m5_last["macd_histogram"]
                > m5_previous["macd_histogram"]
            ):
                score += 3
                reasons.append(
                    "M5 bullish momentum is increasing."
                )

            if (
                m15_last["close"]
                > m15_last["ema20"]
                and m5_last["close"]
                > m5_last["ema20"]
            ):
                score += 4
                reasons.append(
                    "Price is holding above EMA20 on M15 and M5."
                )

        else:
            if (
                m15_last["rsi"] <= 48
                and m5_last["rsi"] <= 48
            ):
                score += 7
                reasons.append(
                    "M15 and M5 RSI are aligned below 48."
                )
            elif (
                m15_last["rsi"] < 50
                and m5_last["rsi"] < 50
            ):
                score += 4

            if (
                m15_last["macd_histogram"] < 0
                and m5_last["macd_histogram"] < 0
            ):
                score += 6
                reasons.append(
                    "M15 and M5 MACD histogram are negative."
                )

            if (
                m5_last["macd_histogram"]
                < m5_previous["macd_histogram"]
            ):
                score += 3
                reasons.append(
                    "M5 bearish momentum is increasing."
                )

            if (
                m15_last["close"]
                < m15_last["ema20"]
                and m5_last["close"]
                < m5_last["ema20"]
            ):
                score += 4
                reasons.append(
                    "Price is holding below EMA20 on M15 and M5."
                )

        return min(
            score,
            20,
        )

    @staticmethod
    def _score_volatility(
        m15: pd.DataFrame,
        reasons: List[str],
    ) -> int:
        score = 0

        latest = m15.iloc[-1]

        atr_value = float(
            latest["atr"]
        )

        atr_median = float(
            m15["atr"]
            .tail(80)
            .median()
        )

        bb_width_value = float(
            latest["bb_width"]
        )

        bb_width_median = float(
            m15["bb_width"]
            .tail(80)
            .median()
        )

        if (
            atr_median > 0
            and atr_value >= atr_median
        ):
            score += 8
            reasons.append(
                "M15 ATR is at or above its recent median."
            )
        elif (
            atr_median > 0
            and atr_value >= atr_median * 0.80
        ):
            score += 5
            reasons.append(
                "M15 ATR is sufficient for a tradable move."
            )

        if (
            bb_width_median > 0
            and bb_width_value >= bb_width_median
        ):
            score += 7
            reasons.append(
                "Bollinger width confirms active price expansion."
            )
        elif (
            bb_width_median > 0
            and bb_width_value
            >= bb_width_median * 0.75
        ):
            score += 4

        return min(
            score,
            15,
        )

    @staticmethod
    def _score_execution(
        side: str,
        price: float,
        current_atr: float,
        m5_last: pd.Series,
        rr: float,
        reasons: List[str],
    ) -> int:
        score = 0

        if rr >= 2.0:
            score += 9
            reasons.append(
                f"Projected TP2 risk-to-reward is {rr}:1."
            )
        elif rr >= 1.7:
            score += 6
            reasons.append(
                f"Projected risk-to-reward is acceptable at {rr}:1."
            )

        distance_from_ema20 = abs(
            price
            - float(
                m5_last["ema20"]
            )
        )

        if distance_from_ema20 <= current_atr * 0.45:
            score += 6
            reasons.append(
                "Entry is close to M5 value and not excessively extended."
            )
        elif distance_from_ema20 <= current_atr * 0.75:
            score += 3

        if side == "BUY":
            if float(m5_last["rsi"]) >= 75:
                score = max(
                    score - 3,
                    0,
                )
        else:
            if float(m5_last["rsi"]) <= 25:
                score = max(
                    score - 3,
                    0,
                )

        return min(
            score,
            15,
        )

    @staticmethod
    def _calculate_trade_levels(
        side: str,
        price: float,
        current_atr: float,
        m15: pd.DataFrame,
    ) -> Dict[str, float]:
        latest = m15.iloc[-1]

        if side == "BUY":
            entry_low = (
                price
                - current_atr * 0.15
            )

            entry_high = (
                price
                + current_atr * 0.08
            )

            entry_mid = (
                entry_low
                + entry_high
            ) / 2

            structural_stop = float(
                latest["low"]
            ) - current_atr * 0.10

            atr_stop = (
                entry_mid
                - current_atr * 0.95
            )

            stop_loss = min(
                structural_stop,
                atr_stop,
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

        else:
            entry_low = (
                price
                - current_atr * 0.08
            )

            entry_high = (
                price
                + current_atr * 0.15
            )

            entry_mid = (
                entry_low
                + entry_high
            ) / 2

            structural_stop = float(
                latest["high"]
            ) + current_atr * 0.10

            atr_stop = (
                entry_mid
                + current_atr * 0.95
            )

            stop_loss = max(
                structural_stop,
                atr_stop,
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

        return {
            "entry_low": float(entry_low),
            "entry_high": float(entry_high),
            "entry_mid": float(entry_mid),
            "stop_loss": float(stop_loss),
            "tp1": float(tp1),
            "tp2": float(tp2),
            "tp3": float(tp3),
        }

    @staticmethod
    def _setup_name(
        side: str,
        liquidity_score: int,
        momentum_score: int,
    ) -> str:
        direction = (
            "Bullish"
            if side == "BUY"
            else "Bearish"
        )

        if (
            liquidity_score >= 14
            and momentum_score >= 14
        ):
            return (
                f"{direction} Liquidity Sweep "
                "and Momentum Continuation"
            )

        if liquidity_score >= 14:
            return (
                f"{direction} Liquidity Reversal"
            )

        return (
            f"{direction} Trend Continuation"
        )


# ============================================================
# TELEGRAM SIGNAL FORMAT
# ============================================================

def format_signal(
    signal: Signal,
) -> str:
    if signal.side == "BUY":
        direction_header = (
            "🟢🧠 <b>GERKO-XAU BUY SIGNAL</b>"
        )

        direction_tag = "#GX_BUY"

    else:
        direction_header = (
            "🔴🧠 <b>GERKO-XAU SELL SIGNAL</b>"
        )

        direction_tag = "#GX_SELL"

    reason_text = "\n".join(
        f"• {reason}"
        for reason in signal.reasons
    )

    created_at = signal.created_at.strftime(
        "%d %b %Y, %H:%M"
    )

    return f"""
{direction_header}
━━━━━━━━━━━━━━━━━━━━

<b>MODEL IDENTITY</b>
Model: GERKO-MIMIC XAU
Signal ID: <code>{signal.signal_id}</code>
Symbol: {signal.symbol}
Generated: {created_at}

<b>TRADE DECISION</b>
Direction: <b>{signal.side}</b>
Grade: <b>{signal.grade}</b>
Confidence: <b>{signal.score}/100</b>
Quality: {signal.quality}
Regime: {signal.regime}
Setup: {signal.setup_name}

<b>GERKO SCORECARD</b>
Trend: {signal.trend_score}/30
Liquidity: {signal.liquidity_score}/20
Momentum: {signal.momentum_score}/20
Volatility: {signal.volatility_score}/15
Execution: {signal.execution_score}/15

<b>EXECUTION ZONE</b>
Entry: <b>{signal.entry_low:.2f} – {signal.entry_high:.2f}</b>
Entry Midpoint: {signal.entry_mid:.2f}
Stop Loss: <b>{signal.stop_loss:.2f}</b>

<b>PROFIT TARGETS</b>
TP1: {signal.tp1:.2f} — 1R
TP2: {signal.tp2:.2f} — 2R
TP3: {signal.tp3:.2f} — 3R

<b>RISK PROFILE</b>
Suggested Lot: <b>{signal.lot_size:.2f}</b>
TP2 Risk-to-Reward: <b>1:{signal.rr:.2f}</b>

<b>MODEL EVIDENCE</b>
{reason_text}

<b>INVALIDATION</b>
{signal.invalidation}

⚠️ Wait for price to enter the stated zone. Do not chase
the signal after a large move away from entry.

#GERKO_XAU {direction_tag} #GERKO_MIMIC
""".strip()