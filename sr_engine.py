from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd


@dataclass
class SRLevel:
    price: float
    level_type: str
    touches: int
    strength: int
    timeframe: str


@dataclass
class BounceResult:
    detected: bool
    side: str
    level: Optional[SRLevel]
    score: int
    reasons: List[str]


@dataclass
class BreakRetestResult:
    detected: bool
    side: str
    level: Optional[SRLevel]
    score: int
    reasons: List[str]


def find_swing_lows(
    df: pd.DataFrame,
    strength: int = 2,
) -> List[float]:
    lows: List[float] = []

    for i in range(
        strength,
        len(df) - strength,
    ):
        centre = float(
            df.iloc[i]["low"]
        )

        left = df.iloc[
            i - strength:i
        ]["low"]

        right = df.iloc[
            i + 1:i + strength + 1
        ]["low"]

        if (
            centre <= float(left.min())
            and centre <= float(right.min())
        ):
            lows.append(
                centre
            )

    return lows


def find_swing_highs(
    df: pd.DataFrame,
    strength: int = 2,
) -> List[float]:
    highs: List[float] = []

    for i in range(
        strength,
        len(df) - strength,
    ):
        centre = float(
            df.iloc[i]["high"]
        )

        left = df.iloc[
            i - strength:i
        ]["high"]

        right = df.iloc[
            i + 1:i + strength + 1
        ]["high"]

        if (
            centre >= float(left.max())
            and centre >= float(right.max())
        ):
            highs.append(
                centre
            )

    return highs


def cluster_prices(
    prices: List[float],
    tolerance: float,
) -> List[List[float]]:
    if not prices:
        return []

    prices = sorted(
        prices
    )

    clusters: List[List[float]] = [
        [prices[0]]
    ]

    for price in prices[1:]:
        mean_price = (
            sum(clusters[-1])
            / len(clusters[-1])
        )

        if (
            abs(
                price - mean_price
            )
            <= tolerance
        ):
            clusters[-1].append(
                price
            )

        else:
            clusters.append(
                [price]
            )

    return clusters


def strength_from_touches(
    touches: int,
) -> int:
    if touches >= 6:
        return 20

    if touches == 5:
        return 19

    if touches == 4:
        return 17

    if touches == 3:
        return 14

    if touches == 2:
        return 10

    return 5


def build_levels(
    df: pd.DataFrame,
    atr_value: float,
    timeframe: str,
) -> List[SRLevel]:
    tolerance = max(
        atr_value * 0.25,
        0.20,
    )

    lows = find_swing_lows(
        df
    )

    highs = find_swing_highs(
        df
    )

    levels: List[SRLevel] = []

    for cluster in cluster_prices(
        lows,
        tolerance,
    ):
        touches = len(
            cluster
        )

        levels.append(
            SRLevel(
                price=sum(cluster) / touches,
                level_type="SUPPORT",
                touches=touches,
                strength=strength_from_touches(
                    touches
                ),
                timeframe=timeframe,
            )
        )

    for cluster in cluster_prices(
        highs,
        tolerance,
    ):
        touches = len(
            cluster
        )

        levels.append(
            SRLevel(
                price=sum(cluster) / touches,
                level_type="RESISTANCE",
                touches=touches,
                strength=strength_from_touches(
                    touches
                ),
                timeframe=timeframe,
            )
        )

    return levels


def combine_levels(
    h1_levels: List[SRLevel],
    m15_levels: List[SRLevel],
    atr_value: float,
) -> List[SRLevel]:
    tolerance = max(
        atr_value * 0.30,
        0.30,
    )

    merged: List[SRLevel] = []

    for level in (
        h1_levels
        + m15_levels
    ):
        matched: Optional[
            SRLevel
        ] = None

        for existing in merged:
            if (
                existing.level_type
                == level.level_type
                and abs(
                    existing.price
                    - level.price
                )
                <= tolerance
            ):
                matched = existing
                break

        if matched is None:
            merged.append(
                SRLevel(
                    price=level.price,
                    level_type=level.level_type,
                    touches=level.touches,
                    strength=level.strength,
                    timeframe=level.timeframe,
                )
            )

            continue

        total_touches = (
            matched.touches
            + level.touches
        )

        matched.price = (
            (
                matched.price
                * matched.touches
            )
            +
            (
                level.price
                * level.touches
            )
        ) / total_touches

        matched.touches = (
            total_touches
        )

        matched.strength = min(
            20,
            max(
                matched.strength,
                level.strength,
            )
            + 2,
        )

        if (
            matched.timeframe
            != level.timeframe
        ):
            matched.timeframe = (
                "H1+M15"
            )

    return merged


def nearest_level(
    price: float,
    levels: List[SRLevel],
    level_type: str,
) -> Optional[SRLevel]:
    filtered = [
        level
        for level in levels
        if level.level_type
        == level_type
    ]

    if not filtered:
        return None

    return min(
        filtered,
        key=lambda level: abs(
            price - level.price
        ),
    )


def resistance_is_breaking(
    m5: pd.DataFrame,
    resistance: float,
    atr_value: float,
) -> bool:
    latest = m5.iloc[-1]
    previous = m5.iloc[-2]

    buffer = (
        atr_value * 0.08
    )

    return bool(
        float(latest["close"])
        > resistance + buffer

        and float(previous["close"])
        > resistance

        and float(latest["close"])
        > float(latest["open"])

        and float(
            latest["macd_histogram"]
        )
        > float(
            previous["macd_histogram"]
        )
    )


def support_is_breaking(
    m5: pd.DataFrame,
    support: float,
    atr_value: float,
) -> bool:
    latest = m5.iloc[-1]
    previous = m5.iloc[-2]

    buffer = (
        atr_value * 0.08
    )

    return bool(
        float(latest["close"])
        < support - buffer

        and float(previous["close"])
        < support

        and float(latest["close"])
        < float(latest["open"])

        and float(
            latest["macd_histogram"]
        )
        < float(
            previous["macd_histogram"]
        )
    )


def detect_support_bounce(
    m5: pd.DataFrame,
    m15: pd.DataFrame,
    levels: List[SRLevel],
    atr_value: float,
) -> BounceResult:
    latest = m5.iloc[-1]
    previous = m5.iloc[-2]

    price = float(
        latest["close"]
    )

    support = nearest_level(
        price,
        levels,
        "SUPPORT",
    )

    if support is None:
        return BounceResult(
            False,
            "",
            None,
            0,
            [],
        )

    if support_is_breaking(
        m5,
        support.price,
        atr_value,
    ):
        return BounceResult(
            False,
            "",
            support,
            0,
            [
                "Support is breaking, "
                "so BUY bounce is vetoed."
            ],
        )

    zone = max(
        atr_value * 0.30,
        0.30,
    )

    distance = abs(
        price
        - support.price
    )

    if distance > (
        atr_value * 0.45
    ):
        return BounceResult(
            False,
            "",
            support,
            0,
            [
                "Support bounce is already "
                "too extended from the level."
            ],
        )

    score = min(
        support.strength,
        20,
    )

    reasons = [
        (
            f"{support.timeframe} support "
            f"{support.price:.2f} has "
            f"{support.touches} historical touches."
        )
    ]

    touched = (
        float(latest["low"])
        <= support.price + zone
    )

    reclaimed = (
        float(latest["close"])
        > support.price
    )

    bullish = (
        float(latest["close"])
        > float(latest["open"])
    )

    if touched:
        score += 10

        reasons.append(
            "Price entered the support zone."
        )

    if (
        float(latest["low"])
        < support.price
        and reclaimed
    ):
        score += 15

        reasons.append(
            "Price swept below support "
            "and reclaimed it."
        )

    if bullish:
        score += 10

        reasons.append(
            "M5 closed bullish after "
            "testing support."
        )

    candle_range = max(
        float(
            latest["high"]
            - latest["low"]
        ),
        0.01,
    )

    lower_wick = (
        min(
            float(latest["open"]),
            float(latest["close"]),
        )
        - float(latest["low"])
    )

    if (
        lower_wick
        / candle_range
        >= 0.30
    ):
        score += 10

        reasons.append(
            "Strong lower rejection wick "
            "confirms buying response."
        )

    if (
        float(latest["rsi"])
        > float(previous["rsi"])
    ):
        score += 10

        reasons.append(
            "M5 RSI is turning upward."
        )

    if (
        float(
            latest["macd_histogram"]
        )
        >
        float(
            previous["macd_histogram"]
        )
    ):
        score += 10

        reasons.append(
            "M5 MACD momentum is improving."
        )

    if (
        float(m15.iloc[-1]["close"])
        > float(m15.iloc[-1]["open"])
    ):
        score += 5

        reasons.append(
            "M15 supports the bullish reaction."
        )

    if (
        support.timeframe
        == "H1+M15"
    ):
        score += 5

        reasons.append(
            "Support is confirmed on "
            "both H1 and M15."
        )

    score = min(
        score,
        100,
    )

    detected = (
        touched
        and reclaimed
        and bullish
        and score >= 68
    )

    return BounceResult(
        detected,
        "BUY",
        support,
        score,
        reasons,
    )


def detect_resistance_bounce(
    m5: pd.DataFrame,
    m15: pd.DataFrame,
    levels: List[SRLevel],
    atr_value: float,
) -> BounceResult:
    latest = m5.iloc[-1]
    previous = m5.iloc[-2]

    price = float(
        latest["close"]
    )

    resistance = nearest_level(
        price,
        levels,
        "RESISTANCE",
    )

    if resistance is None:
        return BounceResult(
            False,
            "",
            None,
            0,
            [],
        )

    if resistance_is_breaking(
        m5,
        resistance.price,
        atr_value,
    ):
        return BounceResult(
            False,
            "",
            resistance,
            0,
            [
                "Resistance is breaking, "
                "so SELL bounce is vetoed."
            ],
        )

    zone = max(
        atr_value * 0.30,
        0.30,
    )

    distance = abs(
        price
        - resistance.price
    )

    if distance > (
        atr_value * 0.45
    ):
        return BounceResult(
            False,
            "",
            resistance,
            0,
            [
                "Resistance bounce is already "
                "too extended from the level."
            ],
        )

    score = min(
        resistance.strength,
        20,
    )

    reasons = [
        (
            f"{resistance.timeframe} resistance "
            f"{resistance.price:.2f} has "
            f"{resistance.touches} historical touches."
        )
    ]

    touched = (
        float(latest["high"])
        >= resistance.price - zone
    )

    reclaimed_below = (
        float(latest["close"])
        < resistance.price
    )

    bearish = (
        float(latest["close"])
        < float(latest["open"])
    )

    if touched:
        score += 10

        reasons.append(
            "Price entered the resistance zone."
        )

    if (
        float(latest["high"])
        > resistance.price
        and reclaimed_below
    ):
        score += 15

        reasons.append(
            "Price swept above resistance "
            "and rejected back below."
        )

    if bearish:
        score += 10

        reasons.append(
            "M5 closed bearish from resistance."
        )

    candle_range = max(
        float(
            latest["high"]
            - latest["low"]
        ),
        0.01,
    )

    upper_wick = (
        float(latest["high"])
        -
        max(
            float(latest["open"]),
            float(latest["close"]),
        )
    )

    if (
        upper_wick
        / candle_range
        >= 0.30
    ):
        score += 10

        reasons.append(
            "Strong upper rejection wick "
            "confirms selling response."
        )

    if (
        float(latest["rsi"])
        < float(previous["rsi"])
    ):
        score += 10

        reasons.append(
            "M5 RSI is turning downward."
        )

    if (
        float(
            latest["macd_histogram"]
        )
        <
        float(
            previous["macd_histogram"]
        )
    ):
        score += 10

        reasons.append(
            "M5 MACD momentum is weakening."
        )

    if (
        float(m15.iloc[-1]["close"])
        < float(m15.iloc[-1]["open"])
    ):
        score += 5

        reasons.append(
            "M15 supports the bearish reaction."
        )

    if (
        resistance.timeframe
        == "H1+M15"
    ):
        score += 5

        reasons.append(
            "Resistance is confirmed on "
            "both H1 and M15."
        )

    score = min(
        score,
        100,
    )

    detected = (
        touched
        and reclaimed_below
        and bearish
        and score >= 68
    )

    return BounceResult(
        detected,
        "SELL",
        resistance,
        score,
        reasons,
    )


def detect_break_retest(
    m5: pd.DataFrame,
    levels: List[SRLevel],
    atr_value: float,
) -> BreakRetestResult:
    latest = m5.iloc[-1]
    previous = m5.iloc[-2]
    third = m5.iloc[-3]

    current_price = float(
        latest["close"]
    )

    zone = max(
        atr_value * 0.25,
        0.25,
    )

    resistance = nearest_level(
        current_price,
        levels,
        "RESISTANCE",
    )

    if resistance:
        broke_up = (
            float(third["close"])
            <= resistance.price

            and float(previous["close"])
            > resistance.price
            + zone * 0.20
        )

        retested = (
            float(latest["low"])
            <= resistance.price + zone

            and float(latest["close"])
            > resistance.price
        )

        bullish = (
            float(latest["close"])
            > float(latest["open"])
        )

        momentum_up = (
            float(
                latest["macd_histogram"]
            )
            >=
            float(
                previous["macd_histogram"]
            )
        )

        if (
            broke_up
            and retested
            and bullish
            and momentum_up
        ):
            score = (
                58
                + min(
                    resistance.strength,
                    20,
                )
            )

            reasons = [
                (
                    f"Resistance at "
                    f"{resistance.price:.2f} "
                    "was broken."
                ),
                (
                    "Price retested former "
                    "resistance from above."
                ),
                (
                    "M5 closed bullish "
                    "after the retest."
                ),
            ]

            if (
                float(latest["rsi"])
                > float(previous["rsi"])
            ):
                score += 8

                reasons.append(
                    "RSI is recovering during "
                    "the retest."
                )

            if (
                float(latest["close"])
                > float(latest["ema_20"])
            ):
                score += 7

                reasons.append(
                    "Price is back above M5 EMA20."
                )

            return BreakRetestResult(
                True,
                "BUY",
                resistance,
                min(
                    score,
                    100,
                ),
                reasons,
            )

    support = nearest_level(
        current_price,
        levels,
        "SUPPORT",
    )

    if support:
        broke_down = (
            float(third["close"])
            >= support.price

            and float(previous["close"])
            < support.price
            - zone * 0.20
        )

        retested = (
            float(latest["high"])
            >= support.price - zone

            and float(latest["close"])
            < support.price
        )

        bearish = (
            float(latest["close"])
            < float(latest["open"])
        )

        momentum_down = (
            float(
                latest["macd_histogram"]
            )
            <=
            float(
                previous["macd_histogram"]
            )
        )

        if (
            broke_down
            and retested
            and bearish
            and momentum_down
        ):
            score = (
                58
                + min(
                    support.strength,
                    20,
                )
            )

            reasons = [
                (
                    f"Support at "
                    f"{support.price:.2f} "
                    "was broken."
                ),
                (
                    "Price retested former "
                    "support from below."
                ),
                (
                    "M5 closed bearish "
                    "after the retest."
                ),
            ]

            if (
                float(latest["rsi"])
                < float(previous["rsi"])
            ):
                score += 8

                reasons.append(
                    "RSI is weakening during "
                    "the retest."
                )

            if (
                float(latest["close"])
                < float(latest["ema_20"])
            ):
                score += 7

                reasons.append(
                    "Price is back below M5 EMA20."
                )

            return BreakRetestResult(
                True,
                "SELL",
                support,
                min(
                    score,
                    100,
                ),
                reasons,
            )

    return BreakRetestResult(
        False,
        "",
        None,
        0,
        [],
    )
