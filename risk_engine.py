from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    grade: str
    quality: str
    lot_size: float


def classify_signal(
    score: int,
    minimum: int,
) -> RiskDecision:

    score = max(
        0,
        min(
            int(score),
            100,
        ),
    )

    minimum = max(
        0,
        min(
            int(minimum),
            100,
        ),
    )

    if score < minimum:
        return RiskDecision(
            allowed=False,
            grade="NO TRADE",
            quality="Insufficient Confirmation",
            lot_size=0.00,
        )

    if score >= 94:
        return RiskDecision(
            allowed=True,
            grade="A+",
            quality="Elite Gerko Setup",
            lot_size=0.25,
        )

    if score >= 88:
        return RiskDecision(
            allowed=True,
            grade="A",
            quality="Premium Gerko Setup",
            lot_size=0.20,
        )

    if score >= 83:
        return RiskDecision(
            allowed=True,
            grade="A-",
            quality="Strong Gerko Setup",
            lot_size=0.15,
        )

    if score >= 78:
        return RiskDecision(
            allowed=True,
            grade="B+",
            quality="Qualified Gerko Setup",
            lot_size=0.10,
        )

    return RiskDecision(
        allowed=True,
        grade="B",
        quality="Early Gerko Setup",
        lot_size=0.10,
    )


def calculate_rr(
    entry: float,
    stop: float,
    target: float,
    side: str,
) -> float:

    side = side.upper().strip()

    if side == "BUY":

        risk = entry - stop
        reward = target - entry

    elif side == "SELL":

        risk = stop - entry
        reward = entry - target

    else:

        raise ValueError(
            "side must be BUY or SELL"
        )

    if risk <= 0 or reward <= 0:
        return 0.0

    return round(
        reward / risk,
        2,
    )