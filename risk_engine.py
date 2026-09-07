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
            quality="Elite LumiAj Setup",
            lot_size=0.25,
        )

    if score >= 88:
        return RiskDecision(
            allowed=True,
            grade="A",
            quality="Premium LumiAj Setup",
            lot_size=0.20,
        )

    if score >= 83:
        return RiskDecision(
            allowed=True,
            grade="A-",
            quality="Strong LumiAj Setup",
            lot_size=0.15,
        )

    if score >= 78:
        return RiskDecision(
            allowed=True,
            grade="B+",
            quality="Qualified LumiAj Setup",
            lot_size=0.10,
        )

    return RiskDecision(
        allowed=True,
        grade="B",
        quality="Early LumiAj Setup",
        lot_size=0.10,
    )