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
            False,
            "NO TRADE",
            "Insufficient Confirmation",
            0.00,
        )

    if score >= 94:
        return RiskDecision(
            True,
            "A+",
            "Elite Gerko Setup",
            0.25,
        )

    if score >= 88:
        return RiskDecision(
            True,
            "A",
            "Premium Gerko Setup",
            0.20,
        )

    if score >= 83:
        return RiskDecision(
            True,
            "A-",
            "Strong Gerko Setup",
            0.15,
        )

    if score >= 78:
        return RiskDecision(
            True,
            "B+",
            "Qualified Gerko Setup",
            0.10,
        )

    return RiskDecision(
        True,
        "B",
        "Early Gerko Setup",
        0.10,
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